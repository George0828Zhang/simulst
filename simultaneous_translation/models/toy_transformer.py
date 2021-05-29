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
    register_model_architecture
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
    S2TSinkhornCascadedEncoder,
)
from simultaneous_translation.modules import (
    CausalTransformerEncoderLayer,
    NonCausalTransformerEncoderLayer,
    SinkhornTransformerDecoderLayer
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
    @property
    def output_layer(self):
        return self.output_projection

    def generate(self, src_tokens, src_lengths, blank_idx=0, from_encoder=False, **unused):
        if not from_encoder:
            return generate(self, src_tokens, src_lengths, blank_idx=blank_idx, collapse=False)
        logits, extra = self.forward_causal(src_tokens, src_lengths, None)
        return generate(self, src_tokens, src_lengths, net_output=(logits, extra), blank_idx=blank_idx, collapse=False)

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
        cascade = S2TSinkhornCascadedEncoder(args, encoder)
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

@register_model_architecture(
    "toy_transformer", "toy_transformer"
)
def toy_transformer(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.max_source_positions = getattr(args, "max_source_positions", 512)
    # args.decoder_layers = getattr(args, "decoder_layers", 2)
    # args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    # args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    # args.share_decoder_input_output_embed = True  # force embed sharing
    args.max_target_positions = getattr(args, "max_target_positions", 512)

    args.non_causal_layers = getattr(args, "non_causal_layers", 2)  # 2 non-causal, 1 sinkhorn

    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)

    args.encoder_log_penalty = False
