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
    FairseqEncoderModel, 
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
from simultaneous_translation.modules import WaitkTransformerDecoderLayer

logger = logging.getLogger(__name__)

@register_model("waitk_s2t_transformer")
class S2TWaitkTransformerModel(S2TTransformerModel):
    """
    Waitk S2TTransformer with a bi-directional encoder
    """
    @property
    def pre_decision_ratio(self):
        return self.decoder.pre_decision_ratio

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
        parser.add_argument('--min-waitk', type=int, 
                            help='wait-k for incremental reading')
        parser.add_argument('--max-waitk', type=int, 
                            help='wait-k for incremental reading')
        parser.add_argument('--multi-waitk', action='store_true',  default=False,)
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

    @classmethod
    def build_encoder(cls, args):
        encoder = S2TFullContextEncoder(args)
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
        return decoder


class S2TFullContextEncoder(S2TTransformerEncoderProto):
    """
    1. Return encoder hidden states to be used in downstream modules.
    2. add sliced_encoder_out
    """
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
        self.min_waitk = args.min_waitk
        self.max_waitk = args.max_waitk
        self.multi_waitk = args.multi_waitk
        self.pre_decision_ratio = args.pre_decision_ratio

    def build_decoder_layer(self, args, no_encoder_attn=False):
        # change to waitk layer. 
        return WaitkTransformerDecoderLayer(args, no_encoder_attn)

    def get_attention_mask(self, x, src_len, waitk=None, pre_decision_ratio=None):
        if waitk is None:
            if self.multi_waitk:
                assert self.min_waitk <= self.max_waitk
                waitk = random.randint(min(self.min_waitk, src_len),
                                       min(src_len, self.max_waitk))
            else:
                waitk = self.waitk

        if pre_decision_ratio is None:
            pre_decision_ratio = self.pre_decision_ratio
        
        fake_waitk = waitk * pre_decision_ratio

        if fake_waitk < src_len:
            encoder_attn_mask = torch.triu(
                utils.fill_with_neg_inf(
                    x.new(x.size(0), src_len)
                ), fake_waitk
            )
            if fake_waitk <= 0:
                encoder_attn_mask[:, 0] = 0

        else:
            encoder_attn_mask = None
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
                encoder_attn_mask=None,
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
    args.waitk = getattr(args, 'waitk', 1024) # wait-until-end
    args.min_waitk = getattr(args, 'min_waitk', 1)
    args.max_waitk = getattr(args, 'max_waitk', 1024)
