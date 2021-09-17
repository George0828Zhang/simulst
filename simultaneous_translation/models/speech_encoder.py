#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import logging
from typing import Dict, List, Optional, Tuple
# from collections import OrderedDict
# import re
# import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from fairseq import checkpoint_utils
from fairseq.data.data_utils import lengths_to_padding_mask


from fairseq.models import (
    # FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
# from fairseq.modules import LayerNorm
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.speech_to_text.s2t_transformer import (
    # S2TTransformerModel,
    S2TTransformerEncoder,
    s2t_transformer_s,
)

# user
from simultaneous_translation.models.nat_utils import generate
from simultaneous_translation.modules import (
    CausalTransformerEncoderLayer,
)

logger = logging.getLogger(__name__)


@register_model("speech_encoder")
class SpeechEncoderModel(FairseqEncoderModel):
    """
    causal encoder + output projection
    """
    def __init__(self, encoder):
        super().__init__(encoder)
        self.one_pass_decoding = True  # must implement generate()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(SpeechEncoderModel, SpeechEncoderModel).add_args(parser)
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )

    @classmethod
    def build_encoder(cls, args, src_dict, ctc_projection):
        encoder = CausalSpeechEncoder(
            args, src_dict, ctc_projection)
        encoder.apply(init_bert_params)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        return encoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        src_dict = task.source_dictionary
        ctc_projection = nn.Linear(
            args.encoder_embed_dim,
            len(src_dict),
            bias=False
        )
        encoder = cls.build_encoder(
            args, src_dict, ctc_projection)
        return cls(encoder)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        logits = net_output[0]

        if torch.is_tensor(logits):
            # syntactic sugar for simple models which don't have a decoder
            # (e.g., the classification tutorial)
            logits_f = logits.float()
            if log_probs:
                lprobs = F.log_softmax(logits_f, dim=-1)
            else:
                lprobs = F.softmax(logits_f, dim=-1)
        else:
            raise NotImplementedError

        return lprobs

    def forward(
        self, src_tokens, src_lengths,
        return_all_hiddens: bool = False,
        **unused,
    ):
        encoder_out = self.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens
        )
        x = encoder_out["encoder_out"][0]  # speech hidden states
        x = self.encoder.ctc_projection(x)  # ctc projection
        x = x.transpose(1, 0)  # force batch first

        padding_mask = encoder_out["encoder_padding_mask"][0] \
            if len(encoder_out["encoder_padding_mask"]) > 0 else None
        extra = {
            "padding_mask": padding_mask,
            "encoder_out": encoder_out
        }
        return x, extra

    @property
    def output_layer(self):
        """ convenient function for accuracy calculation """
        return self.encoder.ctc_projection

    def generate(self, src_tokens, src_lengths, blank_idx=0, **unused):
        return generate(self, src_tokens, src_lengths, blank_idx=blank_idx)

    def max_decoder_positions(self):
        """Used by sequence generator."""
        return self.encoder.max_positions()


class CausalSpeechEncoder(S2TTransformerEncoder):
    """Transformer encoder that consists of causal attention.
    """
    def __init__(self, args, src_dict, ctc_projection):
        super().__init__(args)
        self.transformer_layers = nn.ModuleList([
            CausalTransformerEncoderLayer(args) for i in range(args.encoder_layers)
        ])
        self.src_dict = src_dict
        self.ctc_projection = ctc_projection

    def forward(self, src_tokens, src_lengths, return_all_hiddens=False, **unused):
        """ identical to original S2TEncoder (for now) """
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = self.dropout_module(x)

        encoder_states = []

        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any() else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def clear_cache(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        end_id: Optional[int] = None,
        keep: Optional[int] = None,
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
                layer.prune_incremental_state(incremental_state, keep)

    def load_state_dict(self, state_dict, strict=True):
        """
        1. ignores ctc_projection if not available
        """
        ignores = ["ctc_projection.weight", ]
        cur_state_dict = self.state_dict()

        for w in ignores:
            if w not in state_dict:
                logger.warning("Ignoring CTC projection weights! Make sure this is intended...")
                state_dict[w] = cur_state_dict[w]

        return super().load_state_dict(state_dict, strict=strict)


@register_model_architecture(
    "speech_encoder", "speech_encoder_s"
)
def speech_encoder_s(args):
    s2t_transformer_s(args)
