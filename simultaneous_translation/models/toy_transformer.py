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
from fairseq import checkpoint_utils, utils

from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqEncoder,
    FairseqEncoderModel
)
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
    Embedding,
    Linear,
    base_architecture,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params

# user
from .causal_encoder import CausalTransformerEncoder
from simultaneous_translation.models.nat_generate import generate
from simultaneous_translation.modules import (
    CausalTransformerEncoderLayer,
    NonCausalTransformerEncoderLayer,
    SinkhornAttention,
)

logger = logging.getLogger(__name__)

@register_model("toy_transformer")
class ToySinkhornEncoderModel(FairseqEncoderModel):
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
        FairseqEncoderModel.add_args(parser)
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
            "--upsample-ratio",
            type=int,
            help=(
                'number of upsampling factor before ctc loss. used for mt.'
            ),
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = CausalTransformerEncoder(args, src_dict, embed_tokens)
        encoder.apply(init_bert_params)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        cascade = SinkhornCascadedEncoder(args, encoder)
        return cascade

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        src_dict = task.source_dictionary
        encoder_embed_tokens = cls.build_embedding(
            args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
        )

        output_projection = nn.Linear(
            args.encoder_embed_dim, len(task.target_dictionary), bias=False
        )
        nn.init.normal_(
            output_projection.weight, mean=0, std=args.encoder_embed_dim ** -0.5
        )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        # decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, output_projection)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

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

    @property
    def output_layer(self):
        return self.output_projection

    def generate(self, src_tokens, src_lengths, blank_idx=0, from_encoder=False, **unused):
        if not from_encoder:
            return generate(self, src_tokens, src_lengths, blank_idx=blank_idx, collapse=True)
        logits, extra = self.forward_causal(src_tokens, src_lengths, None)
        return generate(self, src_tokens, src_lengths, net_output=(logits, extra), blank_idx=blank_idx, collapse=True)

    def max_decoder_positions(self):
        """Used by sequence generator."""
        return self.encoder.max_positions()

class SinkhornCascadedEncoder(FairseqEncoder):
    """
    Add following layers to the causal encoder,
    1) several non-causal encoder layers
    2) 1 sinkhorn layer attention
    """
    def __init__(self, args, causal_encoder):
        super().__init__(None)
        self.causal_encoder = causal_encoder
        self.non_causal_layers = nn.ModuleList([
            NonCausalTransformerEncoderLayer(args) for i in range(args.non_causal_layers)
        ])
        self.sinkhorn_layer = SinkhornAttention(
            args.encoder_embed_dim,
            bucket_size=args.sinkhorn_bucket_size,
            dropout=0,  # args.attention_dropout, already have gumbel noise
            no_query_proj=True,
            no_key_proj=True,
            no_value_proj=True,
            no_out_proj=True,
            sinkhorn_tau=args.sinkhorn_tau,
            sinkhorn_iters=args.sinkhorn_iters,
            sinkhorn_noise_factor=args.sinkhorn_noise_factor,
            energy_fn=args.sinkhorn_energy,
        )
        self.upsample_ratio = args.upsample_ratio
        if self.upsample_ratio > 1:
            self.upsampler = Linear(
                args.encoder_embed_dim, args.encoder_embed_dim * self.upsample_ratio)

    def upsample(self, x, encoder_padding_mask):
        if self.upsample_ratio == 1:
            return x, encoder_padding_mask

        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.repeat_interleave(
                self.upsample_ratio, dim=1)

        T, B, C = x.size()
        # T x B x C
        # -> T x B x C*U
        # -> U * (T x B x C)
        # -> T x U x B x C
        # -> T*U x B x C
        x = torch.stack(
            torch.chunk(
                self.upsampler(x),
                self.upsample_ratio,
                dim=-1
            ),
            dim=1
        ).view(-1, B, C)
        return x, encoder_padding_mask

    def forward_causal(
        self,
        src_tokens,
        src_lengths,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        return_all_hiddens: bool = False
    ):
        causal_out = self.causal_encoder(
            src_tokens, src_lengths, incremental_state=incremental_state, return_all_hiddens=return_all_hiddens)
        x = causal_out["encoder_out"][0]
        encoder_padding_mask = causal_out["encoder_padding_mask"][0] \
            if len(causal_out["encoder_padding_mask"]) > 0 else None
        x, encoder_padding_mask = self.upsample(x, encoder_padding_mask)
        causal_out.update({
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask is not None else [],  # B x T
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
        x, attn, log_alpha = self.sinkhorn_layer(
            x,  # this is non_causal_states
            causal_states,
            causal_states,
            encoder_padding_mask,
        )

        # upsample
        x, encoder_padding_mask = self.upsample(x, encoder_padding_mask)
        causal_states, _ = self.upsample(causal_states, None)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask is not None else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "attn": [attn],
            "log_alpha": [log_alpha],
            "causal_out": [causal_states],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """ This reorder is fairseq's reordering of batch dimension,
        different from our reorder.
        """
        return self.causal_encoder.reorder_encoder_out(encoder_out, new_order)

@register_model_architecture(
    "toy_transformer", "toy_transformer"
)
def toy_transformer(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    # args.decoder_layers = getattr(args, "decoder_layers", 2)
    # args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    # args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    # args.share_decoder_input_output_embed = True  # force embed sharing
    args.max_target_positions = getattr(args, "max_target_positions", 512)

    args.non_causal_layers = getattr(args, "non_causal_layers", 2)  # 2 non-causal, 1 sinkhorn

    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)

    args.encoder_log_penalty = False

@register_model_architecture(
    "toy_transformer", "toy_transformer_mt"
)
def toy_transformer_mt(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)

    args.non_causal_layers = getattr(args, "non_causal_layers", 6)

    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)

    args.encoder_log_penalty = False
    args.upsample_ratio = getattr(args, "upsample_ratio", 2)
