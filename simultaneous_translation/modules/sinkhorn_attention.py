
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter

from fairseq.modules import MultiheadAttention

def sample_gumbel(proto, eps=1e-6):
    u = torch.empty_like(proto).uniform_(0, 1)
    return -torch.log(-torch.log(u + eps) + eps)

def log_sinkhorn_norm(log_alpha: torch.Tensor, n_iter: int = 20) -> (torch.Tensor,):
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
    return log_alpha.exp()

def gumbel_sinkhorn(log_alpha: torch.Tensor, tau: float = 0.7, n_iter: int = 20, noise: bool = True) -> (torch.Tensor,):
    if noise:
        gumbel_noise = sample_gumbel(log_alpha)
        log_alpha = (log_alpha + gumbel_noise) / tau
    sampled_perm_mat = log_sinkhorn_norm(log_alpha, n_iter)
    return sampled_perm_mat

class SinkhornAttention(MultiheadAttention):
    """ single head sorting attention that preserves value, without projection. """

    def __init__(
        self,
        embed_dim,
        # num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        # self_attention=False,
        # encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        sinkhorn_tau=0.75,
        sinkhorn_iters=8,
    ):
        super().__init__(
            embed_dim,
            num_heads=1,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=False,
            encoder_decoder_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        self.tau = sinkhorn_tau
        self.iters = sinkhorn_iters

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Get unnormalized attention weights by:
         incremental_state=None,
         static_kv=True,
         before_softmax=True,
        """

        log_alpha, _ = super().forward(
            query=query,
            key=key,
            value=value,
            key_padding_mask=None,
            incremental_state=None,
            need_weights=need_weights,
            static_kv=True,
            attn_mask=attn_mask,
            before_softmax=True,
            need_head_weights=need_head_weights,
        )

        # log softmax ?
        # attn_weights_float = utils.log_softmax(
        #     attn_weights, dim=-1, onnx_trace=self.onnx_trace
        # )
        # attn_weights = attn_weights_float.type_as(attn_weights)

        attn_weights_float = gumbel_sinkhorn(
            log_alpha.float(),
            tau=self.tau,
            n_iter=self.iters,
            noise=True
        )

        # perform masking after sinkhorn
        if key_padding_mask is not None:
            # mask out non-pad -> pad attentions.
            attn_weights_float = attn_weights_float.masked_fill(
                key_padding_mask.unsqueeze(1),
                0
            )

        attn_weights = attn_weights_float.type_as(log_alpha)
        attn_probs = self.dropout_module(attn_weights)

        v = value.transpose(0, 1)
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous()

        return attn, attn_weights, log_alpha
