# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.modules import TransformerDecoderLayer, TransformerEncoderLayer

from typing import List, Dict, Optional

from torch import Tensor
import torch


class CausalTransformerEncoderLayer(TransformerEncoderLayer):
    """ TransformerEncoderLayer but adds the following
    1. future masking
    2. incremental states for incremental encoding in inference
    """
    def __init__(self, args):
        super().__init__(args)
        self._future_mask = torch.empty(0)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0 or (
                not self._future_mask.device == tensor.device
            ) or self._future_mask.size(0) < dim
        ):
            neg_inf = -torch.finfo(tensor.dtype).max
            self._future_mask = torch.triu(
                torch.full([dim, dim], neg_inf), 1
            )
        self._future_mask = self._future_mask.type_as(tensor)
        return self._future_mask[:dim, :dim]

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        """
        In inference, prev states are cached so we need to
        compute mask from cached states and input x together.
        """
        if incremental_state is not None:
            proto = x
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            if "prev_key" in saved_state:
                prev_key = saved_state["prev_key"]
                assert prev_key is not None
                prev_len = prev_key.size(2)
                new_len = x.size(0)
                proto = x.new_zeros(
                    (prev_len + new_len, 1))  # only dim 0 is used.
            attn_mask = self.buffered_future_mask(
                proto)[-x.size(0):]  # keep mask for x only
        else:
            attn_mask = self.buffered_future_mask(x)

        #################################################
        # below is same as original + incremental_State #
        #################################################

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

    def prune_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        keep: Optional[int] = None,
    ):
        if keep is None:
            return

        input_buffer = self.self_attn._get_input_buffer(incremental_state)
        for key in ["prev_key", "prev_value"]:
            input_buffer_key = input_buffer[key]
            assert input_buffer_key is not None
            # if input_buffer_key.size(2) > prune:
            if keep > 0:
                input_buffer[key] = input_buffer_key[:, :, :keep, :]
            else:
                typed_empty_dict: Dict[str, Optional[Tensor]] = {}
                input_buffer = typed_empty_dict
                break
        assert incremental_state is not None
        self.self_attn._set_input_buffer(incremental_state, input_buffer)


class WaitkTransformerDecoderLayer(TransformerDecoderLayer):
    """Wait-k Decoder layer block.
    1. added encoder_attn_mask for wait-k masking
    2. for simul trans, we CANNOT cache encoder states! in inference,
        the encoder states dicts should be constantly updated.
    """
    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        encoder_attn_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        cache_encoder: bool = True,
        cache_decoder: bool = True,
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
        if prev_self_attn_state is not None:
            # if incremental_state is None:
            #     incremental_state = {}
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state if cache_decoder else None,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                # if incremental_state is None:
                #     incremental_state = {}
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            # for simul trans, you CANNOT cache encoder states! in inference,
            # the encoder should be constantly updated.
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                attn_mask=encoder_attn_mask,
                incremental_state=None,
                static_kv=False,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

    def prune_incremental_state(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]):
        input_buffer = self.self_attn._get_input_buffer(incremental_state)
        for key in ["prev_key", "prev_value"]:
            input_buffer_key = input_buffer[key]
            assert input_buffer_key is not None
            if input_buffer_key.size(2) > 1:
                input_buffer[key] = input_buffer_key[:, :, :-1, :]
            else:
                typed_empty_dict: Dict[str, Optional[Tensor]] = {}
                input_buffer = typed_empty_dict
                break
        assert incremental_state is not None
        self.self_attn._set_input_buffer(incremental_state, input_buffer)
