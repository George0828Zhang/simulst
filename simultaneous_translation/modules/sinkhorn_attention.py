
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import pdb
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import Tensor, nn
from torch.nn import Parameter


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


class SinkhornAttention(nn.Module):
    """Single head attention with sinkhorn normalization.
    """
    ENERGY_FNS = ["dot", "cos", "L2"]

    def __init__(
        self,
        embed_dim,
        bucket_size,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        no_out_proj=False,
        sinkhorn_tau=0.75,
        sinkhorn_iters=8,
        energy_fn='dot',
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.bucket_size = bucket_size
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.scaling = self.embed_dim ** -0.5

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if no_out_proj:
            self.out_proj = None
        else:
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.tau = sinkhorn_tau
        self.iters = sinkhorn_iters
        self.energy_fn = energy_fn
        assert self.energy_fn in self.ENERGY_FNS, f"{energy_fn} not in {self.ENERGY_FNS}"

        self.reset_parameters()

    def reset_parameters(self):
        # Empirically observed the convergence to be much better with
        # the scaled initialization
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))

        if self.out_proj is not None:
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def pad_to_multiple(self, q, k, v, key_padding_mask):
        """Input shape
            q: (B, T, E),
            k, v: (B, S, E),
            key_padding_mask: (B, S),
        """
        B, T, E = q.size()
        B, S, E = k.size()

        new_q = q
        new_k = k
        new_v = v
        new_key_padding_mask = key_padding_mask

        buckets = math.ceil(T / self.bucket_size)
        kv_buckets = math.ceil(S / self.bucket_size)

        # pad query
        new_T = buckets * self.bucket_size
        if new_T != T:
            new_q = torch.cat([
                q,
                q.new_zeros((B, new_T - T, E)),
            ], dim=1)
            # if attn_mask is not None:
            #     new_attn_mask = attn_mask.new_zeros((new_T, new_T))
            #     new_attn_mask[:T, :T] = attn_mask
            #     new_attn_mask[:, T:].fill_(float("-inf"))

        # pad key value
        new_S = kv_buckets * self.bucket_size
        if new_S != S:
            new_k = torch.cat([
                k,
                k.new_zeros((B, new_S - S, E)),
            ], dim=1)
            new_v = torch.cat([
                v,
                v.new_zeros((B, new_S - S, E)),
            ], dim=1)
            if key_padding_mask is not None:
                new_key_padding_mask = torch.cat([
                    key_padding_mask,
                    key_padding_mask.new_ones((B, new_S - S)),
                ], dim=1)

        return (
            new_q,
            new_k,
            new_v,
            new_key_padding_mask,
            new_T - T,
            new_S - S
        )

    def aggregate_buckets(self, q, k, v, key_padding_mask):
        """Input shape
            q: (B, T, E),
            k, v: (B, S, E),
            key_padding_mask: (B, S),
        """
        B, T, E = q.size()
        B, S, E = k.size()
        buckets = T // self.bucket_size
        kv_buckets = S // self.bucket_size

        # aggregate query & key by meaning (summing in paper?) each buckets
        new_q = q.view(B, buckets, self.bucket_size, E).mean(dim=2)
        new_k = k.view(B, kv_buckets, self.bucket_size, E).mean(dim=2)

        # aggregate value by concatenating into single vector
        new_v = v.contiguous().view(B, kv_buckets, self.bucket_size * E)

        # aggregate padding mask by: if a bucket is all pad then it is masked.
        new_key_padding_mask = key_padding_mask
        if key_padding_mask is not None:
            new_key_padding_mask = key_padding_mask.view(B, kv_buckets, self.bucket_size).prod(dim=2)

        return (
            new_q,
            new_k,
            new_v,
            new_key_padding_mask
        )

    def undo_aggregate_buckets(self, v, tail_v):
        """Input shape
            v: (B, new_S, E),
        """
        B, kv_buckets, bucket_size_E = v.size()
        E = bucket_size_E // self.bucket_size
        new_v = v.view(B, kv_buckets * self.bucket_size, E)
        return new_v[:, :-tail_v, :] if tail_v > 0 else new_v

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        # attn_mask: Optional[Tensor] = None,
        **unused,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len, key_bsz, _ = key.size()

        assert embed_dim == self.embed_dim
        assert key is not None and value is not None
        assert key_bsz == bsz
        assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = value

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            # if attn_mask is not None:
            #     attn_mask = torch.cat(
            #         [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
            #     )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        # q = q.contiguous().transpose(0, 1)
        # k = k.contiguous().transpose(0, 1)
        # v = v.contiguous().transpose(0, 1)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        q, k, v, key_padding_mask, q_tail, v_tail = self.pad_to_multiple(
            q, k, v, key_padding_mask)

        q, k, v, new_key_padding_mask = self.aggregate_buckets(
            q, k, v, key_padding_mask)

        tgt_len = q.size(1)
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if new_key_padding_mask is not None and new_key_padding_mask.dim() == 0:
            new_key_padding_mask = None

        if self.energy_fn == "dot":
            attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scaling
        elif self.energy_fn == "cos":
            attn_weights = F.cosine_similarity(
                q.unsqueeze(2),  # (bsz, tgt_len, 1, embed_dim)
                k.unsqueeze(1),  # (bsz, 1, src_len, embed_dim)
                dim=-1,
            )
        elif self.energy_fn == "L2":
            attn_weights = -torch.cdist(q, k, p=2) * self.scaling

        log_alpha = attn_weights

        assert list(attn_weights.size()) == [bsz, tgt_len, src_len]

        # if attn_mask is not None:
        #     attn_mask = attn_mask.unsqueeze(0)
        #     attn_weights += attn_mask

        if new_key_padding_mask is not None:
            assert list(new_key_padding_mask.size()) == [bsz, src_len]
            new_key_padding_mask = new_key_padding_mask.bool()

            final_mask = new_key_padding_mask.unsqueeze(1) & (~new_key_padding_mask).unsqueeze(2)

            # mask out normal -> pad attentions
            attn_weights = attn_weights.masked_fill(
                final_mask,
                float("-inf"),
            )
            # mask out pad -> normal attentions
            attn_weights = attn_weights.masked_fill(
                final_mask.transpose(2, 1),
                float("-inf"),
            )

        attn_weights_float = gumbel_sinkhorn(
            attn_weights.float(),
            tau=self.tau,
            n_iter=self.iters,
            noise=True
        )

        attn_weights = attn_weights_float.type_as(log_alpha)
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)

        attn = self.undo_aggregate_buckets(attn, v_tail)

        attn = attn.transpose(0, 1)  # .contiguous()

        if self.out_proj is not None:
            attn = self.out_proj(attn)

        return attn, attn_weights, log_alpha
