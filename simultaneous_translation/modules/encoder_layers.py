#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/elbayadm/attn2d/blob/master/examples/waitk
# Implementation of the papers:
#   *SimulMT to SimulST: Adapting Simultaneous Text Translation to
#       End-to-End Simultaneous Speech Translation
#       https://www.aclweb.org/anthology/2020.aacl-main.58.pdf
#   *Efficient Wait-k Models for Simultaneous Machine Translation
#       http://www.interspeech2020.org/uploadfile/pdf/Tue-1-1-2.pdf

import logging
from typing import Dict, List, Optional

import torch

from fairseq import utils
from fairseq.modules import TransformerEncoderLayer, TransformerDecoderLayer
from torch import Tensor

logger = logging.getLogger(__name__)

class NonCausalTransformerEncoderLayer(TransformerEncoderLayer):
    """ Enhance encoder layers by adding log-distance penalty """
    def __init__(self, args):
        super().__init__(args)
        self._future_mask = torch.empty(0)
        self.log_penalty = args.encoder_log_penalty

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.zeros([dim, dim])
            if self.log_penalty:
                penalty = torch.arange(dim).type_as(self._future_mask)
                penalty = torch.abs(
                    penalty.unsqueeze(1) - penalty
                ).clamp(min=1)
                self._future_mask -= penalty.log()
        self._future_mask = self._future_mask.to(tensor)
        if self._future_mask.any():
            return self._future_mask[:dim, :dim]
        else:
            return None

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):

        attn_mask = self.buffered_future_mask(x)

        ######################################
        # below is same as original          #
        ######################################

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,  # add incremental state
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

class CausalTransformerEncoderLayer(NonCausalTransformerEncoderLayer):
    """ Same as NonCausal, but adds future masking """

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            neg_inf = -torch.finfo(tensor.dtype).max
            self._future_mask = torch.triu(
                torch.full([dim, dim], neg_inf), 1
                # utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
            if self.log_penalty:
                penalty = torch.arange(dim).type_as(self._future_mask)
                penalty = torch.abs(
                    penalty.unsqueeze(1) - penalty
                ).clamp(min=1)
                self._future_mask -= penalty.log()
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]
