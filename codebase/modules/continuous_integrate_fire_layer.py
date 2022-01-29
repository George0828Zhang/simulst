from typing import Optional, Dict
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch import Tensor
from fairseq.incremental_decoding_utils import with_incremental_state


@with_incremental_state
class CIFLayer(nn.Module):
    def __init__(
        self,
        in_features,
        beta,
    ):
        r""" Continuous integrate and fire described in
        https://arxiv.org/abs/1905.11235
        https://arxiv.org/pdf/2101.06699.pdf

        Args:
            ctc_proj: The out-projection layer (a.k.a nn.Linear) for ctc if wanted.
        """
        super().__init__()

        self.out_proj = nn.Linear(in_features - 1, in_features)
        self.beta = beta

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor] = None,
        target_lengths: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        r"""
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if incremental_state is not None:
            raise NotImplementedError("This functionality is for streaming. You be implemented soon.")

        x = x.transpose(1, 0).contiguous()
        B, S, C = x.size()
        # show("BSC:", (B,S,C))

        # calculate integration weights
        alpha = x[..., -1]  # B, S
        if encoder_padding_mask is not None:
            alpha[encoder_padding_mask] = -1e4
        alpha = torch.sigmoid(alpha)

        # show("alpha:", alpha)
        if target_lengths is not None:
            T = target_lengths.max().long()

            # B, 1
            target_lengths = target_lengths.unsqueeze(1).type_as(x)
            # for quantity loss & scaling
            alpha_sum = alpha.sum(1)  # B
            alpha = alpha * self.beta * target_lengths / alpha_sum.unsqueeze(1)
        else:
            alpha_sum = alpha.sum(1)
            T = (alpha_sum.max() / self.beta + 1e-4).floor().long()

        # show("x:", x)
        # show("tgt:", target_lengths)
        # show("alpha':", alpha)

        # aggregate and integrate
        csum = alpha.cumsum(-1)
        # show("csum:", csum)
        # dest_idx[:, :-1] is the scatter index of left weight
        # dest_idx[:, 1:] is that of right weight
        dest_idx = torch.cat((
            x.new_zeros((B, 1), dtype=torch.long),
            (csum / self.beta + 1e-4).floor().long()
        ), dim=1)
        # show("dest_idx:", dest_idx)

        # number of fires on each position, typically at most 1
        # other rare case (extra_weight) is handled below
        # NOTE: extra_weights is discrete, this may have incorrect gradient?
        fire_num = dest_idx[:, 1:] - dest_idx[:, :-1]
        extra_weights = (fire_num - 1).clip(min=0).type_as(alpha) * self.beta
        # show("fire_num:", fire_num)
        # show("extra_weights:", extra_weights)

        attn_weights = alpha.new_full((B, T + 1, S), 0)

        # right scatter
        right_mask = fire_num > 0
        right_weight = (csum - dest_idx[:, 1:] * self.beta) * right_mask + alpha * (~right_mask)
        # show("right_weight:", right_weight)
        attn_weights.scatter_add_(
            1,
            dest_idx[:, 1:].unsqueeze(1),
            right_weight.unsqueeze(1)
        )

        # left scatter
        left_weight = alpha - right_weight - extra_weights
        # show("left_weight:", left_weight)
        attn_weights.scatter_add_(
            1,
            dest_idx[:, :-1].unsqueeze(1),
            left_weight.unsqueeze(1)
        )

        # extra
        if extra_weights.any():
            extra_idx = (dest_idx[:, :-1] + 1).clip(max=T - 1)
            attn_weights.scatter_add_(
                1,
                extra_idx.unsqueeze(1),
                extra_weights.clip(max=self.beta).unsqueeze(1)
            )
            extra_weights -= self.beta
            if extra_weights.any():
                # for more than 1, manually scatter
                extra_idx += 1
                for uid, pos in extra_weights.nonzero():
                    n = extra_weights[uid, pos].long()
                    tpos = extra_idx[uid, pos].long()
                    attn_weights[uid, tpos:tpos + n, pos] = self.beta

        # trim
        if target_lengths is not None:
            attn_weights = attn_weights[:, :T, :]
        else:
            # a size (B,) mask that removes non-firing position
            tail_mask = attn_weights[:, T, :].sum(-1) < (self.beta / 2.0001)
            if tail_mask.all():
                attn_weights = attn_weights[:, :T, :]
            else:
                attn_weights[tail_mask, T, :] = 0

        # cif operation
        # (B, T, S) x (B, S, C-1) -> (B, T, C-1)
        feats = x[..., :-1]  # B, S, C-1
        feats = torch.bmm(attn_weights, feats)

        # project back and (B, T, C-1) -> (T, B, C-1)
        feats = self.out_proj(feats)
        feats = feats.transpose(0, 1).contiguous()

        return feats, alpha_sum

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool = False,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
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
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "cif_state")
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
        return self.set_incremental_state(incremental_state, "cif_state", buffer)
