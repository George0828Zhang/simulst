#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
from torch import Tensor
from typing import List, Dict, Optional
from fairseq import checkpoint_utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import TransformerDecoder

# user
from simultaneous_translation.modules.monotonic_transformer_layer import (
    WaitkTransformerDecoderLayer,
)
from simultaneous_translation.models.seq2seq import (
    ST2TTransformerModel,
    st2t_transformer_s
)

logger = logging.getLogger(__name__)


@register_model("st2t_transformer_waitk")
class WaitkST2TTransformerModel(ST2TTransformerModel):
    """
    causal encoder (+ semantic encoder) + waitk decoder
    """
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(WaitkST2TTransformerModel, WaitkST2TTransformerModel).add_args(parser)
        # parser.add_argument(
        #     '--waitk-max',
        #     type=int,
        #     required=True,
        #     help='maximum wait-k for incremental reading. k is sampled from [1, waitk_max]')
        parser.add_argument(
            '--waitk-list',
            type=str,
            required=True,
            help='list of choices of wait-k for incremental reading.'
            'k is sampled from this list. e.g. 3,9,15,21,27')

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        tgt_dict = task.target_dictionary

        decoder = WaitkTransformerDecoder(args, tgt_dict, embed_tokens)

        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
        return decoder

    @property
    def waitk(self):
        return max(self.decoder.waitk_list)

    @property
    def pre_decision_ratio(self):
        return self.decoder.pre_decision_ratio


class WaitkTransformerDecoder(TransformerDecoder):
    """
    1. Adds wait-k encoder_masks in training.
    """
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        **kwargs,
    ):
        super().__init__(args, dictionary, embed_tokens, **kwargs)

        self.waitk_list = [int(k) for k in args.waitk_list.split(',')]
        self.pre_decision_ratio = 1  # 1 for text model, > 1 for speech.
        for k in self.waitk_list:
            if k < 1:
                raise ValueError("waitk lagging can only be positive.")

    def build_decoder_layer(self, args, no_encoder_attn=False):
        # change to waitk layer.
        return WaitkTransformerDecoderLayer(args, no_encoder_attn)

    def get_attention_mask(self, x, src_len):

        # waitk = torch.randint(1, self.waitk_max + 1, [1]).item()
        choice = torch.randint(0, len(self.waitk_list), [1]).item()
        waitk = self.waitk_list[choice]
        pre_decision_ratio = self.pre_decision_ratio

        pooled_src_len = src_len // pre_decision_ratio + 1

        if waitk >= pooled_src_len:
            return None

        neg_inf = -torch.finfo(x.dtype).max
        encoder_attn_mask = torch.triu(
            x.new_full(
                (x.size(0), pooled_src_len),
                neg_inf
            ), waitk
        )
        if waitk <= 0:
            encoder_attn_mask[:, 0] = 0

        # upsample
        encoder_attn_mask = encoder_attn_mask.repeat_interleave(
            pre_decision_ratio, dim=1)[:, :src_len]

        return encoder_attn_mask

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        add encoder_attn_mask (wait-k masking) at training time. otherwise is the same as original.
        """

        bs, slen = prev_output_tokens.size()
        encoder_states: Optional[Tensor] = None
        encoder_padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            encoder_states = encoder_out["encoder_out"][0]
            assert (
                encoder_states.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {encoder_states.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            encoder_padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            # full_length = prev_output_tokens.size(1)
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None:
                # training time
                self_attn_mask = self.buffered_future_mask(x)
                encoder_attn_mask = self.get_attention_mask(x, encoder_states.size(0))
            else:
                # inference time
                self_attn_mask = None
                encoder_attn_mask = None
                # encoder_attn_mask = self.get_attention_mask(
                #     x.expand(full_length, -1, -1), encoder_states.size(0))
                # encoder_attn_mask = encoder_attn_mask[-1:, :] if encoder_attn_mask is not None else None

            x, attn = layer(
                x,
                encoder_states,
                encoder_padding_mask,
                encoder_attn_mask=encoder_attn_mask,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def clear_cache(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        end_id: Optional[int] = None,
    ):
        """
        Clear cache in the monotonic layers.
        The cache is generated because of a forward pass of decode but no prediction.
        end_id is the last idx of the layers
        """
        if end_id is None:
            end_id = len(self.layers)

        for index, layer in enumerate(self.layers):
            if index < end_id:
                layer.prune_incremental_state(incremental_state)


@register_model_architecture(
    "st2t_transformer_waitk", "st2t_transformer_waitk_s"
)
def st2t_transformer_waitk_s(args):
    st2t_transformer_s(args)
