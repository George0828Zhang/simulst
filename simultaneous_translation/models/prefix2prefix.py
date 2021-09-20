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

# user
from simultaneous_translation.models.transformer_monotonic_attention import (
    TransformerMonotonicDecoder,
)
from simultaneous_translation.models.seq2seq import (
    ST2TTransformerModel,
    st2t_transformer_s
)

logger = logging.getLogger(__name__)


@register_model("st2t_transformer_monotonic")
class SimulST2TTransformerModel(ST2TTransformerModel):
    """
    causal encoder (+ semantic encoder) + monotonic decoder
    """
    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        tgt_dict = task.target_dictionary

        decoder = TransformerMonotonicDecoder(args, tgt_dict, embed_tokens)

        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
        return decoder


@register_model_architecture(
    "st2t_transformer_monotonic", "st2t_transformer_monotonic_s"
)
def st2t_transformer_monotonic_s(args):
    st2t_transformer_s(args)
