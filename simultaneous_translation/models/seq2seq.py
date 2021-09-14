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
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser.
        """
        super(WeightedShrinkingTransformerModel,
              WeightedShrinkingTransformerModel).add_args(parser)
        parser.add_argument(
            "--do-weighted-shrink",
            action="store_true",
            default=False,
            help="shrink the encoder states based on ctc output.",
        )

    @classmethod
    def build_encoder(cls, args, task, ctc_projection):
        encoder = WSCausalSpeechEncoder(
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
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            src_txt_tokens=src_txt_tokens,
            src_txt_lengths=src_txt_lengths,
        )
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


class WSCausalSpeechEncoder(CausalSpeechEncoder):
    """Causal SpeechEncoder + weighted shrinking
    """
    def __init__(self, args, src_dict, ctc_projection):
        super().__init__(args, src_dict, ctc_projection)
        self.do_weighted_shrink = args.do_weighted_shrink

    def forward(
        self,
        src_tokens,
        src_lengths,
        src_txt_tokens,  # unused
        src_txt_lengths,  # unused
    ):
        # encode speech
        encoder_out = super().forward(src_tokens=src_tokens, src_lengths=src_lengths)
        return self.weighted_shrinking(encoder_out)

    def weighted_shrinking(self, encoder_out, return_all_hiddens=False):
        blank = self.src_dict.bos()
        pad = self.src_dict.pad()

        # speech mask
        speech_padding_mask = encoder_out["encoder_padding_mask"][0] \
            if len(encoder_out["encoder_padding_mask"]) > 0 else None

        # speech hidden states
        speech_states = encoder_out["encoder_out"][0]
        encoder_states = encoder_out["encoder_states"]
        if return_all_hiddens:
            encoder_states.append(speech_states)

        # ctc projection
        logits = self.ctc_projection(speech_states).transpose(0, 1)

        if not self.do_weighted_shrink:
            encoder_out.update({
                "encoder_logits": [logits],
                "encoder_states": encoder_states,  # List[T x B x C]
                "speech_padding_mask": encoder_out["encoder_padding_mask"]
            })
            return encoder_out

        B, S, C = logits.size()
        if speech_padding_mask is not None:
            # replace paddings' logits s.t. they predict pads
            one_hot = F.one_hot(torch.LongTensor([pad]), C).type_as(logits)
            logits[speech_padding_mask] = one_hot

        with torch.no_grad():
            # predict CTC tokens
            ctc_pred = logits.argmax(-1)
            # previos predictions
            pre_pred = torch.cat(
                (
                    ctc_pred[:, :1],  # dup first
                    ctc_pred[:, :-1],
                ), dim=1
            )
            # boundary condition & aggregate to segment id
            segment_ids = ((pre_pred != blank) & (ctc_pred != pre_pred)) | (
                (ctc_pred == pad) & (ctc_pred != pre_pred))
            segment_ids = segment_ids.cumsum(dim=1)
            # prepare attn matrix with max len
            shrink_lengths = segment_ids.max(dim=-1)[0] + 1
            attn_weights = utils.fill_with_neg_inf(
                logits.new_empty((B, shrink_lengths.max(), S))
            )
            # compute non-blank confidence
            confidence = 1 - logits.softmax(-1)[..., blank]
            # compute attn to shrink speech states
            attn_weights.scatter_(
                1,
                segment_ids.unsqueeze(1),
                confidence.unsqueeze(1)
            )
            attn_weights = utils.softmax(
                attn_weights, dim=-1
            ).type_as(confidence).nan_to_num(nan=0.)

        # shrink speech states
        # (B, S', S) x (B, S, D) -> (B, S', D) -> (S', B, D)
        shrinked_states = torch.bmm(
            attn_weights,
            speech_states.transpose(0, 1)
        ).transpose(0, 1)

        assert shrinked_states.size(1) == B
        if speech_padding_mask is not None:
            # pad states are shrunk to a segment
            # remove this 'pad segment'
            segment_ids[speech_padding_mask] = -1
            shrink_lengths = segment_ids.max(dim=-1)[0] + 1
            shrinked_states = shrinked_states[:shrink_lengths.max(), ...]

        encoder_padding_mask = lengths_to_padding_mask(shrink_lengths)
        # calculate shrink rate
        src_lengths = shrink_lengths.new_full(
            (B,), S) if speech_padding_mask is None else (~speech_padding_mask).sum(-1).type_as(shrink_lengths)
        shrink_rate = (src_lengths / shrink_lengths).sum()
        return {
            "encoder_out": [shrinked_states],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "encoder_logits": [logits],
            "speech_padding_mask": [speech_padding_mask],
            "src_tokens": [],
            "src_lengths": [],
            "shrink_rate": [shrink_rate]
        }


@register_model_architecture(
    "ws_transformer", "ws_transformer_s"
)
def ws_transformer_s(args):
    # args.encoder_log_penalty = True  # force log penalty
    s2t_transformer_s(args)
