#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/lucidrains/sinkhorn-transformer
# Implementation of the papers:
#   *Sparse Sinkhorn Attention
#       https://arxiv.org/pdf/2002.11296.pdf

import logging
from typing import Dict, List, Optional

import torch

from fairseq import utils
from fairseq.modules import TransformerDecoderLayer
from torch import Tensor

# user
from simultaneous_translation.modules import SinkhornAttention

logger = logging.getLogger(__name__)

class SinkhornTransformerDecoderLayer(TransformerDecoderLayer):
    """Sinkhorn Decoder layer block.
    """
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, no_fc=False
    ):
        super().__init__(
            args,
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn
        )
        self.no_fc = no_fc
        if no_fc:
            self.fc1 = self.fc2 = None

    def build_encoder_attention(self, embed_dim, args):
        return SinkhornAttention(
            self.embed_dim,
            bucket_size=args.sinkhorn_bucket_size,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=0,  # args.attention_dropout,
            no_out_proj=True,
            sinkhorn_tau=args.sinkhorn_tau,
            sinkhorn_iters=args.sinkhorn_iters,
            energy_fn=args.sinkhorn_energy,
        )

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=None,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        log_alpha = None
        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            x, attn, log_alpha = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        if self.no_fc:
            return x, attn, log_alpha

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn, log_alpha

    def forward_fc(self, x):
        """ when decoding with encoder out, and training is with fc,
        need to do fc on states before prediction.
        """
        if self.encoder_attn.out_proj is not None:
            x = self.encoder_attn.out_proj(x)

        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        if self.no_fc:
            return x

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)  # need this?
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn
