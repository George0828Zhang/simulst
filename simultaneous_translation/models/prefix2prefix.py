#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from fairseq import checkpoint_utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.speech_to_text.s2t_transformer import (
    s2t_transformer_s,
)
from examples.simultaneous_translation.models.transformer_monotonic_attention import (
    TransformerMonotonicDecoder,
)

# user
from simultaneous_translation.models.seq2seq import (
    WeightedShrinkingTransformerModel
)

logger = logging.getLogger(__name__)


@register_model("ws_transformer_monotonic")
class SimulWeightedShrinkingTransformerModel(WeightedShrinkingTransformerModel):
    """
    causal encoder (+ semantic encoder) + monotonic decoder
    """
    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        tgt_dict = task.target_dictionary

        decoder = TransformerMonotonicDecoder(args, tgt_dict, embed_tokens)
        decoder.apply(init_bert_params)
        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
        return decoder


@register_model_architecture(
    "ws_transformer_monotonic", "ws_transformer_monotonic_s"
)
def ws_transformer_monotonic_s(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.share_decoder_input_output_embed = True
    s2t_transformer_s(args)
