#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
# from typing import Dict, List, Optional, Tuple
# from collections import OrderedDict
# import re
# import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import Tensor
from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models.transformer import Embedding

from fairseq.models import (
    register_model,
    register_model_architecture,
)
# from fairseq.modules import LayerNorm
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerModel,
    s2t_transformer_s,
)

# user
from simultaneous_translation.models.speech_encoder import CausalSpeechEncoder

logger = logging.getLogger(__name__)


@register_model("ws_transformer")
class WeightedShrinkingTransformerModel(S2TTransformerModel):
    """
    causal encoder (+ semantic encoder) + monotonic decoder
    """
    # def __init__(self, encoder, d):
    #     super().__init__(encoder)

    # @staticmethod
    # def add_args(parser):
    #     """Add model-specific arguments to the parser."""
    #     super(WeightedShrinkingTransformerModel,
    #           WeightedShrinkingTransformerModel).add_args(parser)

    @classmethod
    def build_encoder(cls, args, task, ctc_projection):
        encoder = CausalSpeechEncoder(
            args, task.source_dictionary, ctc_projection)
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
        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        ctc_projection = nn.Linear(
            args.encoder_embed_dim,
            len(task.source_dictionary),
            bias=False
        )
        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder = cls.build_encoder(args, task, ctc_projection)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    def weighted_shrinking(self, encoder_out):
        blank = self.encoder.src_dict.bos()

        encoder_states = encoder_out["encoder_states"]
        # speech mask
        speech_padding_mask = encoder_out["encoder_padding_mask"][0] \
            if len(encoder_out["encoder_padding_mask"]) > 0 else None

        # speech hidden states
        speech_states = encoder_out["encoder_out"][0]
        encoder_states.append(speech_states)

        # ctc projection
        logits = self.encoder.ctc_projection(
            speech_states).transpose(0, 1)
        B, S, C = logits.size()
        if speech_padding_mask is not None:
            """ replace padding positions' logits so
            that they predicts blank after argmax. """
            one_hot = F.one_hot(torch.full((1,), blank), C).type_as(logits)
            logits[speech_padding_mask] = one_hot

        with torch.no_grad():
            # predict CTC tokens
            ctc_pred = logits.argmax(-1)
            # segment the speech
            pre_pred = torch.cat(
                (
                    ctc_pred[:, :1],
                    ctc_pred[:, :-1],
                ), dim=1
            )
            segment_ids = (
                (pre_pred != blank) & (ctc_pred != pre_pred)  # boundary
            ).cumsum(dim=1)
            # prepare attention to shrink speech states
            shrink_lengths = segment_ids[..., -1] + 1
            attn_weights = utils.fill_with_neg_inf(
                logits.new_empty((B, shrink_lengths.max(), S))
            )
            # compute non-blank confidence
            probs = logits.softmax(-1)
            confidence = 1 - probs[..., blank]
            attn_weights.scatter_(
                1,
                segment_ids.unsqueeze(1),
                confidence.unsqueeze(1)
            )
            attn_weights = utils.softmax(
                attn_weights, dim=-1
            ).type_as(confidence).nan_to_num(nan=0.)
        # shrink speech states (S, B, D) -> (B, S', D)
        shrinked_states = torch.bmm(
            attn_weights,
            speech_states.transpose(0, 1)
        ).transpose(0, 1)

        encoder_padding_mask = lengths_to_padding_mask(shrink_lengths)
        # calculate shrink rate
        src_lengths = shrink_lengths.new_full(
            (B,), S) if speech_padding_mask is None else (~speech_padding_mask).sum(-1).type_as(shrink_lengths)
        shrink_rate = (src_lengths / shrink_lengths).sum()
        return {
            "encoder_out": [shrinked_states],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any() else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "encoder_logits": [logits],
            "speech_padding_mask": [speech_padding_mask],
            "src_tokens": [],
            "src_lengths": [],
            "shrink_rate": [shrink_rate]
        }

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        src_txt_tokens,
        src_txt_lengths,
    ):
        """
        """
        encoder_out = self.encoder(
            src_tokens=src_tokens, src_lengths=src_lengths)
        encoder_out = self.weighted_shrinking(encoder_out)
        logits, extra = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        if extra is None:
            extra = {}
        extra.update({
            # "padding_mask": padding_mask,
            "encoder_out": encoder_out
        })
        return logits, extra

    @property
    def output_layer(self):
        """ convenient function for accuracy calculation """
        return self.encoder.ctc_projection


@register_model_architecture(
    "ws_transformer", "ws_transformer_s"
)
def ws_transformer_s(args):
    # args.encoder_log_penalty = True  # force log penalty
    s2t_transformer_s(args)
