import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules import ConvTBC


def make_causal(klass):
    @with_incremental_state
    class Name(klass):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.padding = (0, *self.padding[1:])  # 1. remove automatic temporal pad
            self.Tdim, self.Bdim, self._manual_pad = self._infer_causal_params()

        @property
        def manual_padding(self):
            return max(self._manual_pad)

        # 2. add incremental helper funcs
        def _get_input_buffer(
            self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
        ) -> Dict[str, Optional[Tensor]]:
            result = self.get_incremental_state(incremental_state, "conv_state")
            if result is not None:
                return result
            else:
                empty_result: Dict[str, Optional[Tensor]] = {}
                return empty_result

        def _set_input_buffer(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            buffer: Dict[str, Optional[Tensor]],
        ):
            return self.set_incremental_state(incremental_state, "conv_state", buffer)

        def reorder_incremental_state(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            new_order: Tensor,
        ):
            """Reorder buffered internal state (for incremental generation)."""
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is not None:
                for k in input_buffer.keys():
                    input_buffer_k = input_buffer[k]
                    if input_buffer_k is not None:
                        input_buffer[k] = input_buffer_k.index_select(self.Bdim, new_order)
                incremental_state = self._set_input_buffer(incremental_state, input_buffer)
            return incremental_state

        # 3. make causal
        def forward(self, x, incremental_state=None):
            # B x C x T x D
            k = self.kernel_size[0]
            cur_len = x.size(self.Tdim)
            if incremental_state is not None:
                saved_state = self._get_input_buffer(incremental_state)
                if saved_state is not None and "prev_feat" in saved_state:
                    prev_feat = saved_state["prev_feat"]
                    assert prev_feat is not None
                    x = torch.cat([prev_feat, x], dim=self.Tdim)
                    saved_state["prev_feat"] = x
                    self._set_input_buffer(incremental_state, saved_state)

            x = F.pad(x, self._manual_pad)  # left pad kernel - 1
            # x = x[:, :, -(cur_len + k - 1):]  # keep k - 1 left context
            x = x.narrow(self.Tdim, -(cur_len + k - 1), (cur_len + k - 1))  # dim, start, length
            return super().forward(x)

    Name.__name__ = klass.__name__
    return Name


@make_causal
class CausalConv1d(nn.Conv1d):
    def _infer_causal_params(self):
        """ B, C, T, padding at last."""
        return 2, 0, (self.kernel_size[0] - 1, 0)


@make_causal
class CausalConv2d(nn.Conv2d):
    def _infer_causal_params(self):
        """ B, C, T, D, padding at 2nd to last."""
        return 2, 0, (0, 0, self.kernel_size[0] - 1, 0)


@make_causal
class CausalConvTBC(ConvTBC):
    def _infer_causal_params(self):
        """ T B C, padding at 3rd to last."""
        return 0, 1, (0, 0, 0, 0, self.kernel_size[0] - 1, 0)


class CausalConv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
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
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            CausalConv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                # padding=k // 2
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for c in self.conv_layers:
            padding = getattr(c, "manual_padding", 2 * c.padding[0])
            out = ((out.float() + padding - c.dilation[0] * (c.kernel_size[0] - 1) - 1) / c.stride[0] + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None):
        # bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = F.glu(x, dim=1)
        # _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)
