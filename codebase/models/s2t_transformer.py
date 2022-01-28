#!/usr/bin/env python3

import logging
from pathlib import Path
from fairseq import checkpoint_utils
from fairseq.data.data_utils import lengths_to_padding_mask

logger = logging.getLogger(__name__)

from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerEncoder,
    S2TTransformerModel,
    s2t_transformer_s
)
from fairseq.models.wav2vec.wav2vec2 import (
    make_conv_pos
)


@register_model("s2t_transformer_convpos")
class S2TTransformerConvPosModel(S2TTransformerModel):
    """Replace pos encoding with conv."""
    @staticmethod
    def add_args(parser):
        super(S2TTransformerConvPosModel,
              S2TTransformerConvPosModel).add_args(parser)
        parser.add_argument(
            "--conv-pos",
            type=int,
            metavar="N",
            help="number of filters for convolutional positional embeddings",
        )
        parser.add_argument(
            "--conv-pos-groups",
            type=int,
            metavar="N",
            help="number of groups for convolutional positional embedding",
        )

    @classmethod
    def build_encoder(cls, args):
        encoder = S2TTransformerConvPosEncoder(args)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder


class S2TTransformerConvPosEncoder(S2TTransformerEncoder):
    """S2T Transformer encoder that uses convnet for pos."""

    def __init__(self, args):
        super().__init__(args)

        # self.embed_positions = PositionalEmbedding(
        #     args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        # )
        self.embed_positions = make_conv_pos(
            args.encoder_embed_dim,
            args.conv_pos,
            args.conv_pos_groups,
        )

    def _forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        # positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)

        # T B C -> B C T -> T B C
        positions = self.embed_positions(
            x.permute(1, 2, 0)).permute(2, 0, 1)

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
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask.any()
            else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }


@register_model_architecture("s2t_transformer_convpos", "s2t_transformer_convpos_s")
def s2t_transformer_convpos_s(args):
    # positional embeddings
    args.conv_pos = getattr(args, "conv_pos", 128)
    args.conv_pos_groups = getattr(args, "conv_pos_groups", 16)
    s2t_transformer_s(args)
