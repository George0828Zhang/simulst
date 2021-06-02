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
# from fairseq.data.data_utils import lengths_to_padding_mask

# from torch import Tensor

from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqEncoder
)
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
    Embedding,
    Linear,
    base_architecture,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
# from fairseq.models.speech_to_text.s2t_transformer import (
#     S2TTransformerModel,
#     S2TTransformerEncoder as S2TTransformerEncoderProto,
#     s2t_transformer_s,
# )

# user
from simultaneous_translation.models.nat_generate import generate
from simultaneous_translation.models.sinkhorn_encoders import (
    S2TSinkhornEncoderModel,
    # S2TSinkhornCascadedEncoder,
)
from simultaneous_translation.modules import (
    CausalTransformerEncoderLayer,
    NonCausalTransformerEncoderLayer,
    # SinkhornTransformerDecoderLayer
    SinkhornAttention,
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

@register_model("toy_transformer")
class ToySinkhornEncoderModel(S2TSinkhornEncoderModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        S2TSinkhornEncoderModel.add_args(parser)
        parser.add_argument(
            "--upsample-ratio",
            type=int,
            help=(
                'number of upsampling factor before ctc loss. used for mt.'
            ),
        )

    @property
    def output_layer(self):
        return self.output_projection

    def generate(self, src_tokens, src_lengths, blank_idx=0, from_encoder=False, **unused):
        if not from_encoder:
            return generate(self, src_tokens, src_lengths, blank_idx=blank_idx, collapse=True)
        logits, extra = self.forward_causal(src_tokens, src_lengths, None)
        return generate(self, src_tokens, src_lengths, net_output=(logits, extra), blank_idx=blank_idx, collapse=True)

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
        x, attn, log_alpha = self.sinkhorn_layer(
            x,  # this is non_causal_states
            causal_states,
            causal_states,
            encoder_padding_mask,
        )

        # upsample
        x = x.repeat_interleave(
            self.upsample_ratio, dim=0)  # batch middle
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.repeat_interleave(
                self.upsample_ratio, dim=1)

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
