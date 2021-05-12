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

@register_model("sinkhorn_nat")
class S2TSinkhornNATransformerModel(S2TTransformerModel):
    """
    S2TTransformer with a uni-directional encoder and reorder decoder
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.one_pass_decoding = True  # must implement generate()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(S2TSinkhornNATransformerModel, S2TSinkhornNATransformerModel).add_args(parser)
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )
        parser.add_argument('--sinkhorn-tau', type=float, required=True,
                            help='temperature for gumbel sinkhorn.')
        parser.add_argument(
            "--sinkhorn-iters",
            type=int,
            required=True,
            help=(
                'iters of sinkhorn normalization to perform.'
            ),
        )
        parser.add_argument(
            "--sinkhorn-bucket-size",
            type=int,
            required=True,
            help=(
                'number of elements to group before performing sinkhorn sorting.'
            ),
        )
        parser.add_argument(
            "--sinkhorn-energy",
            type=str,
            required=True,
            choices=["dot", "cos", "L2"],
            help=(
                'type of energy function to use to calculate attention. available: dot, cos, L2'
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
        decoder = SinkhornNATransformerDecoder(args, task.target_dictionary, embed_tokens)
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
            prev_output_tokens=None,  # prev_output_tokens,
            encoder_out=encoder_out,
            features_only=True,
        )
        # extra["decoder_states"] = x
        logits = self.decoder.output_projection(x)

        # padding mask for speech
        padding_mask = None
        if len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # in this model, encoder and decoder padding masks are the same
        extra["padding_mask"] = padding_mask
        extra["encoder_out"] = encoder_out
        return logits, extra

    def generate(self, src_tokens, src_lengths, blank_idx=0, from_encoder=False, **unused):
        if not from_encoder:
            return generate(self, src_tokens, src_lengths, blank_idx=blank_idx)
        _logits, extra = self.forward(src_tokens, src_lengths, None)
        encoder_out = extra["encoder_out"]
        encoder_states = encoder_out["encoder_out"][0]
        # when decoding with encoder out, need to match decoder
        encoder_states = self.decoder.forward_fc(encoder_states)
        encoder_states = encoder_states.permute(1, 0, 2)  # (N, S, E)
        logits = self.output_layer(
            encoder_states
        )
        return generate(self, src_tokens, src_lengths, net_output=(logits, extra), blank_idx=blank_idx)

class SinkhornNATransformerDecoder(TransformerDecoder):
    """
    https://github.com/lucidrains/sinkhorn-transformer/blob/master/sinkhorn_transformer/sinkhorn_transformer.py
    """
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
    ):
        super().__init__(args, dictionary, embed_tokens)
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [SinkhornTransformerDecoderLayer(args) for i in range(args.decoder_layers - 1)] +
            [SinkhornTransformerDecoderLayer(args, no_fc=True)]
        )

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out,
        **unused,
    ):
        """
        Note that prev_output_tokens is treated as embeddings i.e.
        has shape (T, N, E)
        """
        # input
        x = encoder_out["encoder_out"][0]
        decoder_padding_mask = encoder_out["encoder_padding_mask"][0] \
            if len(encoder_out["encoder_padding_mask"]) > 0 else None
        encoder_states = x
        encoder_padding_mask = decoder_padding_mask

        # position
        T, N, E = x.size()
        positions = self.embed_positions(x.new_empty(N, T, dtype=torch.long))
        x = self.embed_scale * x + positions.transpose(0, 1)
        x = self.dropout_module(x)

        # T x B x C
        attn: Optional[Tensor] = None
        log_alpha: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        self_attn_mask = None

        # decoder layers
        for i, layer in enumerate(self.layers):

            x, attn, log_alpha = layer(
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

        return x, {"attn": [attn], "log_alpha": [log_alpha]}  # , "inner_states": inner_states,}

    def forward_fc(self, x):
        """ when decoding with encoder out, and training is with fc,
        need to do fc on states before prediction.
        """        
        return self.layers[-1].forward_fc(x)

@register_model_architecture(
    "sinkhorn_nat", "sinkhorn_nat_s"
)
def sinkhorn_nat_s(args):
    s2t_transformer_s(args)
