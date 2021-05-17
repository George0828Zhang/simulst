#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Any

import logging
import pdb
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
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerModel,
    s2t_transformer_s,
)

# user
from simultaneous_translation.models.nat_generate import generate
from simultaneous_translation.models.waitk_s2t_transformer import (
    S2TCausalEncoder,
)
from simultaneous_translation.modules import SinkhornTransformerDecoderLayer

logger = logging.getLogger(__name__)

@register_model("simple_nat")
class S2TSimpleNATransformerModel(S2TTransformerModel):
    """
    S2TTransformer with a uni-directional encoder and reorder decoder
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.one_pass_decoding = True  # must implement generate()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(S2TSimpleNATransformerModel, S2TSimpleNATransformerModel).add_args(parser)
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
        decoder = NATransformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True)
        decoder.apply(init_bert_params)
        return decoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens):

        encoder_out = self.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
        )
        x, extra = self.decoder.extract_features(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        logits = self.decoder.output_layer(x)
        extra["decoder_states"] = x

        # padding mask for speech
        padding_mask = encoder_out["encoder_padding_mask"][0] \
            if len(encoder_out["encoder_padding_mask"]) > 0 else None

        extra["encoder_out"] = encoder_out

        # in this model, encoder and decoder padding masks are the same
        extra["padding_mask"] = padding_mask
        return logits, extra

    def forward_embeddings(self, tokens):
        return F.embedding(
            tokens,
            self.decoder.output_projection.weight
        )

    def output_layer(self, x):
        return self.decoder.output_layer(x)

    def generate(self, src_tokens, src_lengths, blank_idx=0, from_encoder=False, **unused):
        if not from_encoder:
            return generate(self, src_tokens, src_lengths, blank_idx=blank_idx)
        _logits, extra = self.forward(src_tokens, src_lengths, None)
        encoder_out = extra["encoder_out"]
        encoder_states = encoder_out["encoder_out"][0]
        encoder_states = encoder_states.permute(1, 0, 2)  # (N, S, E)
        logits = self.output_layer(
            encoder_states
        )
        return generate(self, src_tokens, src_lengths, net_output=(logits, extra), blank_idx=blank_idx)

class NATransformerDecoder(TransformerDecoder):

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out,
        **unused,
    ):
        # input
        x = encoder_out["encoder_out"][0]
        decoder_padding_mask = encoder_out["encoder_padding_mask"][0] \
            if len(encoder_out["encoder_padding_mask"]) > 0 else None

        # T x B x C
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        encoder_states, encoder_padding_mask = None, None
        self_attn_mask = None

        # decoder layers
        for i, layer in enumerate(self.layers):

            x, attn, _ = layer(
                x,
                encoder_states,
                encoder_padding_mask,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=decoder_padding_mask,
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
    "simple_nat", "simple_nat_s"
)
def simple_nat_s(args):
    s2t_transformer_s(args)
    args.share_decoder_input_output_embed = True  # force embed sharing
    args.encoder_log_penalty = True  # force log penalty
