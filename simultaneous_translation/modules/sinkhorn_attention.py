
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


def sample_gumbel(proto, eps=1e-8):
    u = torch.rand_like(proto, dtype=torch.float32)
    return -torch.log(-torch.log(u + eps) + eps)


def log_sinkhorn_norm(log_alpha: torch.Tensor, n_iter: int = 20) -> (torch.Tensor,):
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
    return log_alpha.exp()


def gumbel_sinkhorn(
    log_alpha: torch.Tensor,
    tau: float = 0.7,
    n_iter: int = 20,
    noise_factor: float = 1.0
) -> (torch.Tensor,):
    if noise_factor > 0:
        noise = noise_factor * sample_gumbel(log_alpha)
        log_alpha = log_alpha + noise.type_as(log_alpha)
    log_alpha = log_alpha / tau
    sampled_perm_mat = log_sinkhorn_norm(log_alpha, n_iter)
    return sampled_perm_mat

class GaussianBlur(nn.Conv2d):
    """ Blur the attention map before sinkhorn normalization """
    def __init__(self, kernel_size=3):
        super().__init__(
            1, 1, kernel_size,
            padding=kernel_size // 2,
            bias=False,
            padding_mode='replicate',
        )
        mu = (kernel_size - 1) / 2.
        var = (kernel_size / 2.)**2
        grid = torch.arange(kernel_size) - mu
        grid_x, grid_y = torch.meshgrid(grid, grid)
        grid_xy = grid_x**2 + grid_y**2

        gaussian = torch.exp(
            -grid_xy / (2 * var)
        ).view(1, 1, kernel_size, kernel_size)
        gaussian = gaussian / gaussian.sum()

        self.weight.data = gaussian
        self.weight.data.requires_grad = False

    def forward(self, x):
        return super().forward(
            x.unsqueeze(1)
        ).squeeze(1)

# class DepthwiseConv1dTBC(nn.Conv1d):
#     """ Makes conv1d a drop-in replacement for Linear """
#     def __init__(self, in_channels, out_channels, kernel_size, bias=True):
#         super().__init__(
#             in_channels, out_channels, kernel_size,
#             padding=kernel_size // 2,
#             groups=in_channels,
#             bias=bias,
#             padding_mode='replicate',
#         )
#         # nn.utils.weight_norm(self, dim=2)

#     def forward(self, x):
#         return super().forward(
#             x.permute(1, 2, 0)  # TBC -> BCT
#         ).permute(2, 0, 1)  # BCT -> TBC

# class GaussianBlur(DepthwiseConv1dTBC):
#     """ Makes conv1d a drop-in replacement for Linear """
#     def __init__(self, in_channels, out_channels, kernel_size, bias=False):
#         super().__init__(
#             in_channels, out_channels, kernel_size,
#             bias=bias,
#         )
#         mu = (kernel_size - 1) / 2.
#         var = (kernel_size / 2.)**2
#         gaussian = (1. / (2. * math.pi * var))**0.5 * torch.exp(
#             -(torch.arange(kernel_size) - mu)**2. / (2 * var)
#         ).expand_as(self.weight)

#         self.weight.data = gaussian
#         self.weight.data.requires_grad = False

class SinkhornAttention(nn.Module):
    """Single head attention with sinkhorn normalization.
    """
    ENERGY_FNS = ["dot", "cos", "l2"]

    def __init__(
        self,
        embed_dim,
        bucket_size,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        no_query_proj=False,
        no_key_proj=False,
        no_value_proj=False,
        no_out_proj=False,
        blurr_kernel=1,
        sinkhorn_tau=0.75,
        sinkhorn_iters=8,
        sinkhorn_noise_factor=1.0,
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

        if no_query_proj:
            self.q_proj = None
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if no_key_proj:
            self.k_proj = None
        else:
            self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)

        if no_value_proj:
            self.v_proj = None
        else:
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

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
        self.noise_factor = sinkhorn_noise_factor
        self.energy_fn = energy_fn
        assert self.energy_fn in self.ENERGY_FNS, f"{energy_fn} not in {self.ENERGY_FNS}"

        if blurr_kernel > 1:
            self.blurr = GaussianBlur(blurr_kernel)
        else:
            self.blurr = None

        self.reset_parameters()

    def reset_parameters(self):
        # Empirically observed the convergence to be much better with
        # the scaled initialization
        if self.q_proj is not None:
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        if self.k_proj is not None:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        if self.v_proj is not None:
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))

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
            if key_padding_mask is None:
                key_padding_mask = k.new_zeros((B, S), dtype=torch.bool)

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
            new_key_padding_mask = key_padding_mask.view(
                B, kv_buckets, self.bucket_size).prod(dim=2).type_as(key_padding_mask)

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

        q = query
        k = key
        v = value

        if self.q_proj is not None:
            q = self.q_proj(query)

        if self.k_proj is not None:
            k = self.k_proj(key)

        if self.v_proj is not None:
            v = self.v_proj(value)

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])

            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

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
            # serious underflow for half.
            q = q.float()
            k = k.float()
            attn_weights = F.cosine_similarity(
                q.unsqueeze(2),  # (bsz, tgt_len, 1, embed_dim)
                k.unsqueeze(1),  # (bsz, 1, src_len, embed_dim)
                dim=-1,
            ).type_as(v)
        elif self.energy_fn == "l2":
            # cdist not inplemented for half.
            q = q.float()
            k = k.float()
            attn_weights = -torch.cdist(q, k, p=2).type_as(v)
        else:
            raise NotImplementedError()

        # add blurring
        if self.blurr is not None:
            attn_weights = self.blurr(attn_weights)

        # save a copy before masking
        log_alpha = attn_weights.type_as(v)

        assert list(attn_weights.size()) == [bsz, tgt_len, src_len]

        if new_key_padding_mask is not None:
            assert list(new_key_padding_mask.size()) == [bsz, src_len]
            new_key_padding_mask = new_key_padding_mask.bool()

            final_mask = new_key_padding_mask.unsqueeze(1) & (~new_key_padding_mask).unsqueeze(2)
            neg_inf = -torch.finfo(attn_weights.dtype).max
            # mask out normal -> pad attentions
            attn_weights = attn_weights.masked_fill(
                final_mask,
                neg_inf,
            )
            # mask out pad -> normal attentions
            attn_weights = attn_weights.masked_fill(
                final_mask.transpose(2, 1),
                neg_inf,
            )

        attn_weights_float = gumbel_sinkhorn(
            attn_weights,
            tau=self.tau,
            n_iter=self.iters,
            noise_factor=self.noise_factor if self.training else 0,
        )

        # convert back to half/float
        attn_weights = attn_weights_float.type_as(v)
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)

        attn = self.undo_aggregate_buckets(attn, v_tail)

        attn = attn.transpose(0, 1)

        if self.out_proj is not None:
            attn = self.out_proj(attn)

        return attn, attn_weights, log_alpha
