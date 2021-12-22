#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch.nn as nn
from fairseq import checkpoint_utils
from fairseq.models.transformer import (
    Embedding,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerModel,
    s2t_transformer_s,
)

# user
from .speech_encoder import (
    CausalSpeechEncoder,
    s2t_speech_encoder_s
)

logger = logging.getLogger(__name__)


@register_model("s2t_seq2seq")
class S2TSeq2SeqModel(S2TTransformerModel):
    """
    change encoder to speech encoder + ctc projection
    """
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser.
        """
        super(S2TSeq2SeqModel, S2TSeq2SeqModel).add_args(parser)
        parser.add_argument(
            "--lookahead",
            type=int,
            help="number of hidden states speech encoder lags behind speech features for.",
        )
        parser.add_argument(
            "--mtl-layer-id",
            type=int,
            help="the encoder layer to use to perform ctc-asr on. default is the 6-th which is id 5.",
        )
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )

    @classmethod
    def build_encoder(cls, args, task, embed_tokens, ctc_projection):
        sp_encoder = CausalSpeechEncoder(args, task.source_dictionary, ctc_projection)
        if getattr(args, "load_pretrained_encoder_from", None):
            sp_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=sp_encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained speech encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        sp_encoder.mtl_layer_id = getattr(args, "mtl_layer_id", -1)
        return sp_encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        decoder = super().build_decoder(args, task, embed_tokens)

        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
            logger.info(
                f"loaded pretrained decoder from: "
                f"{args.load_pretrained_decoder_from}"
            )
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        s2t_transformer_s(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        encoder_embed_tokens = build_embedding(
            task.source_dictionary, args.encoder_embed_dim
        )
        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        ctc_projection = nn.Linear(
            encoder_embed_tokens.weight.shape[1],
            encoder_embed_tokens.weight.shape[0],
            bias=False,
        )
        nn.init.normal_(
            ctc_projection.weight, mean=0, std=args.encoder_embed_dim ** -0.5
        )
        encoder = cls.build_encoder(args, task, encoder_embed_tokens, ctc_projection)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)

        return cls(encoder, decoder)

    def forward_ctc_projection(self, encoder_out):
        layer_id = self.encoder.mtl_layer_id
        layer_norm = self.encoder.ctc_layer_norm
        speech_states = encoder_out["encoder_states"][layer_id]
        # layernorm
        if layer_norm is not None:
            speech_states = layer_norm(speech_states)
        # ctc projection
        logits = self.encoder.ctc_projection(
            speech_states).transpose(0, 1)

        encoder_out.update({
            "encoder_logits": [logits],
            "speech_padding_mask": encoder_out["encoder_padding_mask"]
        })
        return encoder_out

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        src_txt_tokens=None,  # unused
        src_txt_lengths=None,  # unused
    ):
        """
        """
        encoder_out = self.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=True,
        )
        encoder_out = self.forward_ctc_projection(encoder_out)
        logits, decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        if decoder_out is None:
            decoder_out = {}
        decoder_out.update({
            "encoder_out": encoder_out,
        })
        return logits, decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        """ temp fix for layer index error when loading check"""
        pass


@register_model_architecture(
    "s2t_seq2seq", "s2t_seq2seq_s"
)
def s2t_seq2seq_s(args):
    args.share_decoder_input_output_embed = True

    args.mtl_layer_id = getattr(args, "mtl_layer_id", 7)

    s2t_speech_encoder_s(args)
    s2t_transformer_s(args)
