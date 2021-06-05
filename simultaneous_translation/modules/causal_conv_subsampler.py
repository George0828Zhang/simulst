#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Dict
import math
import torch
import torch.nn as nn
from torch import Tensor
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler
from fairseq.modules.linearized_convolution import LinearizedConvolution
import pdb

class CausalConv1dSubsampler(nn.Module):
    """Causal Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(CausalConv1dSubsampler, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.n_layers = len(kernel_sizes)
        self.dilation = 1
        self.stride = 2
        self.left_pads = [(k - 1) * self.dilation for k in kernel_sizes]

        self.conv_layers = nn.ModuleList(
            LinearizedConv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                # stride=self.stride,
                padding=self.left_pads[i],  # was k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for pad, ksz in zip(self.left_pads, self.kernel_sizes):
            out = ((out.float() - 1 + 1 * pad - self.dilation * (ksz - 1)) / self.stride + 1).floor().long()
        return out

    def forward(
        self,
        src_tokens,
        src_lengths,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
        Args:
            incremental_state: Used to buffer signal; if not None, then input is
                expected to contain a single frame. If the input order changes
                between time steps, call reorder_incremental_state.
        Input:
            Batch x Time x Channel

        Reshape Note:
            Batch x Time x Channel during inference
            Time x Batch x Channel during training
        """
        if incremental_state is not None:
            # inference
            x = src_tokens[:, -1:]  # B x 1 x (C x D)

            for conv in self.conv_layers:
                x = conv(x)
                x = x[:, ::self.stride, :]
                x = nn.functional.glu(x, dim=-1)

            x = x.transpose(0, 1).contiguous()  # B x T x (C x D) -> T x B x (C x D)
        else:
            # training
            x = src_tokens.transpose(0, 1).contiguous()  # B x T x (C x D) -> T x B x (C x D)

            for conv in self.conv_layers:
                x = conv(x)
                x = x[::self.stride, ...]
                x = nn.functional.glu(x, dim=-1)

        return x, self.get_out_seq_lens_tensor(src_lengths)

def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0.0, **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)

class OldCausalConv1dSubsampler(nn.Module):
    """Causal Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(CausalConv1dSubsampler, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.n_layers = len(kernel_sizes)
        self.dilation = 1
        self.stride = 2
        self.left_pads = [(k - 1) * self.dilation for k in kernel_sizes]

        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=self.stride,
                padding=self.left_pads[i],  # was k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for pad, ksz in zip(self.left_pads, self.kernel_sizes):
            out = ((out.float() - 1 + 2 * pad - self.dilation * (ksz - 1)) / self.stride + 1).floor().long()
        return out

    def forward(
        self,
        src_tokens,
        src_lengths,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)