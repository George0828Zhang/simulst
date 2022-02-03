#!/usr/bin/env python3
import math
import torch.nn as nn
import logging
from pathlib import Path
from fairseq import checkpoint_utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.modules import (
    SamePad
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerEncoder,
    S2TTransformerModel,
    s2t_transformer_s
)
from .causal_conv import CausalConv1d

logger = logging.getLogger(__name__)


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


# positional convolution
def make_conv_pos(embed_dim, kernel_size, groups, causal=False):
    def init(pos_conv, dropout=0):
        std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * embed_dim))
        nn.init.normal_(pos_conv.weight, mean=0, std=std)
        nn.init.constant_(pos_conv.bias, 0)

        pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
        return pos_conv

    class ConvPosWrapper(nn.Module):
        def __init__(self, conv, *others):
            super().__init__()
            self.conv = conv
            self.others = nn.ModuleList(others)

        def forward(self, x, incremental_state=None):
            x = self.conv(x, incremental_state)
            for m in self.others:
                x = m(x)
            return x

    if causal:
        pos_conv = CausalConv1d(
            embed_dim,
            embed_dim,
            kernel_size=(kernel_size + 1) // 2,  # left only + self
            groups=groups,
        )
        return ConvPosWrapper(
            init(pos_conv), nn.GELU())
    else:
        pos_conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        return nn.Sequential(
            init(pos_conv), SamePad(kernel_size), nn.GELU())


@register_model_architecture("s2t_transformer_convpos", "s2t_transformer_convpos_s")
def s2t_transformer_convpos_s(args):
    # positional embeddings
    args.conv_pos = getattr(args, "conv_pos", 128)
    args.conv_pos_groups = getattr(args, "conv_pos_groups", 16)
    s2t_transformer_s(args)
