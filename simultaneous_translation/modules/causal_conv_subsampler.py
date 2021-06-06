#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional
import torch.nn as nn
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler

class CausalConv1dSubsampler(nn.Module):
    """Causal Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
        leakage (int): the number of future frames allowed to be seen. a leakage
            of 1 with 2 layers results in a 2 frame delay in inference.
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

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)
