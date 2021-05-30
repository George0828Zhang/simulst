#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Dict, List, Optional, Any

import logging
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask

from torch import Tensor

from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
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
from simultaneous_translation.modules import SinkhornTransformerDecoderLayer, NonCausalTransformerEncoderLayer

logger = logging.getLogger(__name__)

@register_model("sinkhorn_encoder")
class S2TSinkhornEncoderModel(FairseqEncoderModel):
    """
    S2TTransformer with a causal encoder and reorder module cascaded.
    """
    def __init__(self, encoder, output_projection):
        super().__init__(encoder)
        self.output_projection = output_projection
        self.one_pass_decoding = True  # must implement generate()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(S2TSinkhornEncoderModel, S2TSinkhornEncoderModel).add_args(parser)
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--encoder-log-penalty", action="store_true",
            help=(
                'add logrithmic distance penalty in speech encoder.'
            ),
        )
        parser.add_argument(
            "--non-causal-layers",
            type=int,
            help=(
                'number of layers for non-causal encoder.'
            ),
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
            "--sinkhorn-noise-factor",
            type=float,
            required=True,
            help=(
                'represents how many gumbel randomness in training.'
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
            choices=["dot", "cos", "l2"],
            help=(
                'type of energy function to use to calculate attention. available: dot, cos, L2'
            ),
        )
        parser.add_argument(
            "--fusion-factor",
            type=float,
            required=True,
            help=(
                'represents how many future information glanced in training.'
            ),
        )
        parser.add_argument(
            "--fusion-type",
            type=str,
            required=True,
            choices=["add", "inter", "prob"],
            help=(
                'type of operation to fuse causal and non-causal information. available options:'
                'add: additive, simply adds them together.'
                'inter: interpolation by factor controlled with --fusion-factor.'
                'prob: randomly selected using bernoulli test with probability --fusion-factor.'
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
        cascade = S2TSinkhornCascadedEncoder(args, encoder)
        return cascade

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        output_projection = nn.Linear(
            args.encoder_embed_dim, len(task.target_dictionary), bias=False
        )
        nn.init.normal_(
            output_projection.weight, mean=0, std=args.encoder_embed_dim ** -0.5
        )

        encoder = cls.build_encoder(args)
        # decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, output_projection)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        logits = net_output[0]

        if torch.is_tensor(logits):
            # syntactic sugar for simple models which don't have a decoder
            # (e.g., the classification tutorial)
            logits_f = logits.float()
            if log_probs:
                lprobs = F.log_softmax(logits_f, dim=-1)
            else:
                lprobs = F.softmax(logits_f, dim=-1)
        else:
            raise NotImplementedError

        return lprobs

    def forward_causal(self, src_tokens, src_lengths, return_all_hiddens: bool = False, **unused):

        encoder_out = self.encoder.forward_causal(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
        )
        x = self.output_projection(encoder_out["encoder_out"][0])
        x = x.transpose(1, 0)  # force batch first

        padding_mask = encoder_out["encoder_padding_mask"][0] \
            if len(encoder_out["encoder_padding_mask"]) > 0 else None
        extra = {
            "padding_mask": padding_mask,
            "encoder_out": encoder_out,
            "attn": encoder_out["attn"],
            "log_alpha": encoder_out["log_alpha"],
        }
        return x, extra

    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False, **unused):

        encoder_out = self.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens
        )
        x = self.output_projection(encoder_out["encoder_out"][0])
        x = x.transpose(1, 0)  # force batch first

        padding_mask = encoder_out["encoder_padding_mask"][0] \
            if len(encoder_out["encoder_padding_mask"]) > 0 else None
        extra = {
            "padding_mask": padding_mask,
            "encoder_out": encoder_out,
            "attn": encoder_out["attn"],
            "log_alpha": encoder_out["log_alpha"],
        }
        return x, extra

    def generate(self, src_tokens, src_lengths, blank_idx=0, from_encoder=False, **unused):
        if not from_encoder:
            return generate(self, src_tokens, src_lengths, blank_idx=blank_idx)
        logits, extra = self.forward_causal(src_tokens, src_lengths, None)
        return generate(self, src_tokens, src_lengths, net_output=(logits, extra), blank_idx=blank_idx)

    def max_decoder_positions(self):
        """Used by sequence generator."""
        return self.encoder.max_positions()

class S2TSinkhornCascadedEncoder(FairseqEncoder):
    """
    Add following layers to the causal encoder,
    1) several non-causal encoder layers
    2) 1 sinkhorn layer (reorder layer)
    """
    def __init__(self, args, causal_encoder):
        super().__init__(None)
        self.causal_encoder = causal_encoder
        self.non_causal_layers = nn.ModuleList([
            NonCausalTransformerEncoderLayer(args) for i in range(args.non_causal_layers)
        ])
        self.sinkhorn_layers = nn.ModuleList([
            SinkhornTransformerDecoderLayer(args, no_fc=True)
        ])
        self.fusion_type = args.fusion_type
        self.register_buffer(
            "fusion_factor",
            torch.tensor(args.fusion_factor)
        )

    def forward_causal(self, src_tokens, src_lengths, return_all_hiddens: bool = False):
        causal_out = self.causal_encoder(src_tokens, src_lengths, return_all_hiddens=return_all_hiddens)
        causal_out.update({
            "attn": [],
            "log_alpha": [],
        })
        return causal_out

    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False):
        """ Added non-causal forwards and sinkhorn fusion """
        causal_out = self.causal_encoder(src_tokens, src_lengths, return_all_hiddens=return_all_hiddens)

        # causal outputs
        causal_states = x = causal_out["encoder_out"][0]
        encoder_padding_mask = causal_out["encoder_padding_mask"][0] \
            if len(causal_out["encoder_padding_mask"]) > 0 else None
        encoder_states = causal_out["encoder_states"]

        # forward non-causal layers
        for layer in self.non_causal_layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)
        non_causal_states = x

        # reorder using sinkhorn layers
        # (q,k,v) = (non-causal, causal, causal)
        attn: Optional[Tensor] = None
        log_alpha: Optional[Tensor] = None
        for layer in self.sinkhorn_layers:
            x, attn, log_alpha = layer(
                x.detach(),  # this is non_causal_states
                causal_states,
                encoder_padding_mask,
                self_attn_mask=None,
                self_attn_padding_mask=encoder_padding_mask,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        # fusion of non_causal_states and reordered_states
        if self.fusion_type == "add":
            x = non_causal_states + x
        elif self.fusion_type == "inter":
            x = (1. - self.fusion_factor) * non_causal_states + self.fusion_factor * x
        elif self.fusion_type == "prob":
            factor = torch.bernoulli(self.fusion_factor)
            x = (1. - factor) * non_causal_states + factor * x
        else:
            raise NotImplementedError(f"{self.fusion_type} unkown fusion type.")

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask is not None else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "attn": [attn],
            "log_alpha": [log_alpha],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """ This reorder is fairseq's reordering of batch dimension,
        different from our reorder.
        """
        return self.causal_encoder.reorder_encoder_out(encoder_out, new_order)

@register_model_architecture(
    "sinkhorn_encoder", "sinkhorn_encoder_s"
)
def sinkhorn_encoder_s(args):
    s2t_transformer_s(args)
    args.share_decoder_input_output_embed = True  # force embed sharing
    args.encoder_log_penalty = True  # force log penalty
    args.non_causal_layers = getattr(args, "non_causal_layers", 5)  # 5 non-causal, 1 sinkhorn
