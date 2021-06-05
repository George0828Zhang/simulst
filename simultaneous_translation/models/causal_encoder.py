#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from fairseq.data.data_utils import lengths_to_padding_mask

from fairseq.models.transformer import (
    TransformerEncoder,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerEncoder as S2TTransformerEncoderProto,
)

# user
from simultaneous_translation.modules import (
    CausalConv1dSubsampler,
    CausalTransformerEncoderLayer,
)

logger = logging.getLogger(__name__)

class CausalTransformerEncoder(TransformerEncoder):
    """Transformer encoder that consists of causal attention.
    """
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [CausalTransformerEncoderLayer(args) for i in range(args.encoder_layers)]
        )

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,  # not used
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,  # not used
    ):
        """ Same as parent but with incremental_states """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        # x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                src_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            src_tokens = src_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = encoder_embedding = self.embed_scale * self.embed_tokens(src_tokens)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_padding_mask=encoder_padding_mask if has_pads else None,
                incremental_state=incremental_state,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

class S2TCausalEncoder(S2TTransformerEncoderProto):
    """Speech-to-text Transformer encoder that consists of causal input subsampler
    and causal attention.
    """
    def __init__(self, args):
        super().__init__(args)
        self.subsample = CausalConv1dSubsampler(
            args.input_feat_per_channel * args.input_channels,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )
        self.transformer_layers = nn.ModuleList([])
        self.transformer_layers.extend(
            [CausalTransformerEncoderLayer(args) for i in range(args.encoder_layers)]
        )

    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False,):
        """ Same as prototype but returns hidden states """
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = self.dropout_module(x)

        encoder_states = []
        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any() else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def slice_encoder_out(self, encoder_out, context_size):
        """ Slice encoder output according to *context_size*.
        encoder_out:
            (S, N, E) -> (context_size, N, E)
        encoder_padding_mask:
            (N, S) -> (N, context_size)
        encoder_embedding:
            (N, S, E) -> (N, context_size, E)
        encoder_states:
            List(S, N, E) -> List(context_size, N, E)
        """
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.clone()[:context_size] for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.clone()[:, :context_size] for x in encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x.clone()[:, :context_size] for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.clone()[:context_size]

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }
