#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/elbayadm/attn2d/blob/master/examples/waitk
# Implementation of the papers:
#   *SimulMT to SimulST: Adapting Simultaneous Text Translation to
#       End-to-End Simultaneous Speech Translation
#       https://www.aclweb.org/anthology/2020.aacl-main.58.pdf
#   *Efficient Wait-k Models for Simultaneous Machine Translation
#       http://www.interspeech2020.org/uploadfile/pdf/Tue-1-1-2.pdf

import logging
import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask

from torch import Tensor

from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairseq.models.transformer import (
    TransformerDecoder,
    Embedding,
    Linear,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerModel,
    S2TTransformerEncoder as S2TTransformerEncoderProto,
    s2t_transformer_s,
)

# user
from simultaneous_translation.modules import (
    WaitkTransformerDecoderLayer,
    CausalConv1dSubsampler,
    CausalTransformerEncoderLayer,
)

logger = logging.getLogger(__name__)

@register_model("waitk_s2t_transformer")
class S2TWaitkTransformerModel(S2TTransformerModel):
    """
    S2TTransformer with a uni-directional encoder and wait-k decoder
    """
    @property
    def pre_decision_ratio(self):
        return self.decoder.pre_decision_ratio

    @property
    def waitk(self):
        return self.decoder.waitk

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(S2TWaitkTransformerModel, S2TWaitkTransformerModel).add_args(parser)
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )
        parser.add_argument('--waitk', type=int, required=True,
                            help='wait-k for incremental reading')
        parser.add_argument(
            "--pre-decision-ratio",
            type=int,
            required=True,
            help=(
                "Ratio for the fixed pre-decision,"
                "indicating how many encoder steps will start"
                "simultaneous decision making process."
            ),
        )
        parser.add_argument(
            "--encoder-log-penalty", action="store_true",
            help=(
                'add logrithmic distance penalty in speech encoder.'
            ),
        )

    @classmethod
    def build_encoder(cls, args):
        encoder = S2TCausalEncoder(args)
        encoder.apply(init_bert_params)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        decoder = WaitkTransformerDecoder(args, task.target_dictionary, embed_tokens)
        decoder.apply(init_bert_params)
        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
            logger.info(
                f"loaded pretrained decoder from: "
                f"{args.load_pretrained_decoder_from}"
            )
        return decoder

    def forward_embeddings(self, tokens):
        """ convenient function for sinkhorn loss """
        return F.embedding(
            tokens,
            self.decoder.output_projection.weight
        )

    def output_projection(self, x):
        """ convenient function for sinkhorn loss """
        return self.decoder.output_projection(x)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """ convenient override for sinkhorn loss """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        x, extra = self.decoder(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            features_only=True,
        )
        extra["decoder_states"] = x
        logits = self.decoder.output_projection(x)
        return logits, extra

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

class WaitkTransformerDecoder(TransformerDecoder):
    """
    1. Adds wait-k encoder_masks in training.
    """
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        **kwargs,
    ):
        super().__init__(args, dictionary, embed_tokens, **kwargs)

        self.waitk = args.waitk
        self.pre_decision_ratio = args.pre_decision_ratio

    def build_decoder_layer(self, args, no_encoder_attn=False):
        # change to waitk layer.
        return WaitkTransformerDecoderLayer(args, no_encoder_attn)

    def get_attention_mask(self, x, src_len, waitk=None, pre_decision_ratio=None):
        if waitk is None:
            waitk = self.waitk

        if pre_decision_ratio is None:
            pre_decision_ratio = self.pre_decision_ratio

        pooled_src_len = src_len // pre_decision_ratio + 1

        if waitk >= pooled_src_len:
            return None

        encoder_attn_mask = torch.triu(
            utils.fill_with_neg_inf(
                x.new(x.size(0), pooled_src_len)
            ), waitk
        )
        if waitk <= 0:
            encoder_attn_mask[:, 0] = 0

        # upsample
        encoder_attn_mask = encoder_attn_mask.repeat_interleave(
            pre_decision_ratio, dim=1)[:, :src_len]

        return encoder_attn_mask

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        add encoder_attn_mask (wait-k masking) at training time. otherwise is the same as original.
        """

        bs, slen = prev_output_tokens.size()
        encoder_states: Optional[Tensor] = None
        encoder_padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            encoder_states = encoder_out["encoder_out"][0]
            assert (
                encoder_states.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {encoder_states.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            encoder_padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None:
                # training time
                self_attn_mask = self.buffered_future_mask(x)
                encoder_attn_mask = self.get_attention_mask(x, encoder_states.size(0))
            else:
                # inference time
                self_attn_mask = None
                encoder_attn_mask = None

            x, attn = layer(
                x,
                encoder_states,
                encoder_padding_mask,
                encoder_attn_mask=encoder_attn_mask,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

@register_model_architecture(
    "waitk_s2t_transformer", "waitk_s2t_transformer_s"
)
def waitk_s2t_transformer_s(args):
    s2t_transformer_s(args)
    args.waitk = getattr(args, 'waitk', 60000)  # default is wait-until-end
    args.encoder_freezing_updates = 0  # disable this feature.
    args.share_decoder_input_output_embed = True  # force embed sharing
    args.encoder_log_penalty = True  # force log penalty
