#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from collections import OrderedDict
from fairseq import checkpoint_utils
import math
import logging
import re

import torch
import torch.nn as nn

from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqEncoder,
    FairseqDecoder
)
from fairseq.models.transformer import (
    Embedding,
    Linear,
)
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer
)

# user
from simultaneous_translation.models.seq2seq import (
    ST2TTransformerModel,
    st2t_transformer_s
)
from simultaneous_translation.models.nat_utils import (
    generate,
    inject_noise
)
from simultaneous_translation.modules import (
    SinkhornAttention,
)

logger = logging.getLogger(__name__)


class OutProjection(FairseqDecoder):
    def __init__(self, dictionary, out_projection):
        super().__init__(dictionary)
        self.out_projection = out_projection

    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        enc = encoder_out["encoder_out"][0]
        logits = self.out_projection(enc).transpose(1, 0)  # force batch first
        return logits, {}


@register_model("st2t_sinkhorn_encoder")
class ST2TSinkhornEncoderModel(ST2TTransformerModel):
    """
    causal encoder + ASN + output projection
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.one_pass_decoding = True  # must implement generate()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(ST2TSinkhornEncoderModel,
              ST2TSinkhornEncoderModel).add_args(parser)
        parser.add_argument(
            "--non-causal-layers",
            type=int,
            help=(
                'number of layers for non-causal encoder.'
            ),
        )
        parser.add_argument(
            '--sinkhorn-tau',
            type=float,
            help='temperature for gumbel sinkhorn.'
        )
        parser.add_argument(
            "--sinkhorn-iters",
            type=int,
            help=(
                'iters of sinkhorn normalization to perform.'
            ),
        )
        parser.add_argument(
            "--sinkhorn-noise-factor",
            type=float,
            help=(
                'represents how many gumbel randomness in training.'
            ),
        )
        parser.add_argument(
            "--sinkhorn-bucket-size",
            type=int,
            help=(
                'number of elements to group before performing sinkhorn sorting.'
            ),
        )
        parser.add_argument(
            "--sinkhorn-energy",
            type=str,
            choices=["dot", "cos", "l2"],
            help=(
                'type of energy function to use to calculate attention. available: dot, cos, L2'
            ),
        )
        parser.add_argument(
            "--mask-ratio",
            type=float,
            help=(
                'ratio of target tokens to mask when feeding to sorting network.'
            ),
        )
        parser.add_argument(
            "--mask-uniform",
            action="store_true",
            default=False,
            help=(
                'ratio of target tokens to mask when feeding to aligner.'
            ),
        )
        parser.add_argument(
            "--load-pretrained-cascade-from",
            type=str,
            metavar="STR",
            help="model to take cascade (full speech+text) weights from."
            "This should be the trained `st2t_causal_encoder' model.",
        )
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder (out_projection) weights from (for initialization)."
            "This should be the trained `st2t_causal_encoder' model.",
        )

    @classmethod
    def build_encoder(cls, args, task, encoder_embed_tokens, ctc_projection, decoder_embed_tokens):
        encoder = super(ST2TSinkhornEncoderModel, cls).build_encoder(
            args, task, encoder_embed_tokens, ctc_projection)

        cascade_encoder = ASNAugmentedEncoder(
            args, encoder, task.tgt_dict, decoder_embed_tokens)
        if getattr(args, "load_pretrained_cascade_from", None):
            cascade_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=cascade_encoder, checkpoint=args.load_pretrained_cascade_from
            )
            logger.info(
                f"loaded pretrained cascade encoder from: "
                f"{args.load_pretrained_cascade_from}"
            )
        return cascade_encoder

    @classmethod
    def build_decoder(cls, args, task, decoder_embed_tokens):
        out_projection = nn.Linear(
            decoder_embed_tokens.weight.shape[1],
            decoder_embed_tokens.weight.shape[0],
            bias=False,
        )
        nn.init.normal_(
            out_projection.weight, mean=0, std=args.decoder_embed_dim ** -0.5
        )
        decoder = OutProjection(task.target_dictionary, out_projection)
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
        """Build a new model instance.
        Identical to parent, but here encoder also need decoder_embed_tokens.
        """

        st2t_transformer_s(args)

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
        encoder = cls.build_encoder(
            args, task, encoder_embed_tokens, ctc_projection, decoder_embed_tokens)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    def forward_causal(
        self,
        src_tokens,
        src_lengths,
        src_txt_tokens=None,  # unused
        src_txt_lengths=None,  # unused
    ):
        """changed encoder forward to forward_train.
        added prev_output_tokens to encoder for ASN.
        """
        encoder_out = self.encoder.forward(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
        )
        logits, decoder_out = self.decoder(
            prev_output_tokens=None, encoder_out=encoder_out
        )
        decoder_out.update({
            "encoder_out": encoder_out,
            # speech_padding_mask is for the speech encoder output. asr-ctc uses this.
            # encoder_padding_mask is for the cascade output. st-ctc uses this.
            "padding_mask": encoder_out["encoder_padding_mask"][0],
        })
        return logits, decoder_out

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        src_txt_tokens=None,  # unused
        src_txt_lengths=None,  # unused
    ):
        """changed encoder forward to forward_train.
        added prev_output_tokens to encoder for ASN.
        """
        encoder_out = self.encoder.forward_train(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            prev_output_tokens=prev_output_tokens,
        )
        logits, decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        decoder_out.update({
            "encoder_out": encoder_out,
            # speech_padding_mask is for the speech encoder output. asr-ctc uses this.
            # encoder_padding_mask is for the cascade output. st-ctc uses this.
            "padding_mask": encoder_out["encoder_padding_mask"][0],
        })
        return logits, decoder_out

    def generate(self, src_tokens, src_lengths, blank_idx=0, from_encoder=False, **unused):
        logits, extra = self.forward_causal(src_tokens, src_lengths, None)
        return generate(self, src_tokens, src_lengths, net_output=(logits, extra), blank_idx=blank_idx)


class ASNAugmentedEncoder(FairseqEncoder):
    """
    Add following layers to the causal encoder,
    1) several non-causal encoder layers
    2) 1 sinkhorn attention
    """

    def __init__(self, args, causal_encoder, tgt_dict, decoder_embed_tokens):
        super().__init__(None)
        self.causal_encoder = causal_encoder

        # add missing args
        st2t_sinkhorn_encoder_s(args)

        args.decoder_normalize_before = args.encoder_normalize_before
        self.non_causal_layers = nn.ModuleList([
            TransformerDecoderLayer(args) for i in range(args.non_causal_layers)
        ])
        self.train_causal = args.non_causal_layers == 0
        if self.train_causal:
            logger.info("Training causal encoder only! # non causal layers is 0.")
        export = getattr(args, "export", False)
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim, export=export)
        else:
            self.layer_norm = None
        self.sinkhorn_layer = SinkhornAttention(
            args.encoder_embed_dim,
            bucket_size=args.sinkhorn_bucket_size,
            dropout=0,  # args.attention_dropout, already have gumbel noise
            no_query_proj=True,
            no_key_proj=True,
            no_value_proj=True,
            no_out_proj=True,
            sinkhorn_tau=args.sinkhorn_tau,
            sinkhorn_iters=args.sinkhorn_iters,
            sinkhorn_noise_factor=args.sinkhorn_noise_factor,
            energy_fn=args.sinkhorn_energy,
        )
        self.upsample_ratio = args.upsample_ratio
        if self.upsample_ratio > 1:
            self.upsampler = Linear(
                args.encoder_embed_dim, args.encoder_embed_dim * self.upsample_ratio)

        # below are for input target feeding
        if args.mask_ratio == 1.0:
            logger.info("No context provided to ASN.")
        self.mask_ratio = args.mask_ratio
        self.mask_uniform = args.mask_uniform

        self.tgt_dict = tgt_dict
        self.decoder_embed_tokens = decoder_embed_tokens

        self.decoder_embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(
            args.decoder_embed_dim)
        self.decoder_embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,  # this pos is for target tokens
                args.encoder_embed_dim,
                tgt_dict.pad(),
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

    def upsample(self, x, encoder_padding_mask):
        if self.upsample_ratio == 1:
            return x, encoder_padding_mask

        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.repeat_interleave(
                self.upsample_ratio, dim=1)

        T, B, C = x.size()
        # T x B x C
        # -> T x B x C*U
        # -> U * (T x B x C)
        # -> T x U x B x C
        # -> T*U x B x C
        x = torch.stack(
            torch.chunk(
                self.upsampler(x),
                self.upsample_ratio,
                dim=-1
            ),
            dim=1
        ).view(-1, B, C)
        return x, encoder_padding_mask

    def forward(
        self, src_tokens, src_lengths,
        src_txt_tokens=None,  # unused
        src_txt_lengths=None,  # unused
    ):
        causal_out = self.causal_encoder(src_tokens, src_lengths)
        x = causal_out["encoder_out"][0]
        encoder_padding_mask = causal_out["encoder_padding_mask"][0] \
            if len(causal_out["encoder_padding_mask"]) > 0 else None
        x, encoder_padding_mask = self.upsample(x, encoder_padding_mask)
        causal_out.update({
            "encoder_out": [x],  # T x B x C
            # B x T
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask is not None else [],
            "attn": [],
            "log_alpha": [],
        })
        return causal_out

    def forward_train(
        self, src_tokens, src_lengths,
        prev_output_tokens,
        src_txt_tokens=None,  # unused
        src_txt_lengths=None,  # unused
    ):
        """ Added forwards for non-causal and sinkhorn attention """
        if self.train_causal:
            return self.forward(
                src_tokens,
                src_lengths,
                src_txt_tokens,
                src_txt_lengths,
            )

        causal_out = self.causal_encoder(src_tokens, src_lengths)

        # causal outputs
        causal_states = x = causal_out["encoder_out"][0]
        encoder_padding_mask = causal_out["encoder_padding_mask"][0] \
            if len(causal_out["encoder_padding_mask"]) > 0 else None
        encoder_states = causal_out["encoder_states"]

        # target feeding (noise + pos emb)
        if self.mask_ratio == 1.:
            prev_states = prev_padding_mask = None
        else:
            prev_tokens, prev_padding_mask = inject_noise(
                prev_output_tokens,
                self.tgt_dict,
                ratio=self.mask_ratio,
                uniform=self.mask_uniform,
            )

            prev_states = self.decoder_embed_scale * \
                self.decoder_embed_tokens(prev_tokens)
            prev_states += self.decoder_embed_positions(prev_tokens)
            prev_states = prev_states.transpose(0, 1)

        # forward non-causal layers
        for layer in self.non_causal_layers:
            # x = layer(x, encoder_padding_mask)
            x, _, _ = layer(
                x,
                prev_states,
                prev_padding_mask,
                self_attn_padding_mask=encoder_padding_mask,
            )

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # reorder using sinkhorn layers
        # (q,k,v) = (non-causal, causal, causal)
        x, attn, log_alpha = self.sinkhorn_layer(
            x,  # this is non_causal_states
            causal_states,
            causal_states,
            encoder_padding_mask,
        )

        # upsample
        x, encoder_padding_mask = self.upsample(x, encoder_padding_mask)
        causal_states, _ = self.upsample(causal_states, None)

        causal_out.update({
            "encoder_out": [x],
            "encoder_padding_mask": [encoder_padding_mask],
            "encoder_states": encoder_states,
            "attn": [attn],
            "log_alpha": [log_alpha],
            "causal_out": [causal_states],
        })
        return causal_out

    def reorder_encoder_out(self, encoder_out, new_order):
        """ This reorder is fairseq's reordering of batch dimension,
        different from our reorder.
        """
        return self.causal_encoder.reorder_encoder_out(encoder_out, new_order)

    def load_state_dict(self, state_dict, strict=True):
        """
        1. ignores missing non_causal_layers (loading from causal_st)
        """
        ignores = re.compile("^non_causal_layers.")
        cur_state_dict = self.state_dict()
        new_state_dict = state_dict
        for k, v in cur_state_dict.items():
            if ignores.search(k) is not None:
                new_state_dict[k] = v
        return super().load_state_dict(new_state_dict, strict=strict)


@register_model_architecture(
    "st2t_sinkhorn_encoder", "st2t_sinkhorn_encoder_s"
)
def st2t_sinkhorn_encoder_s(args):
    args.non_causal_layers = getattr(args, "non_causal_layers", 3)
    args.upsample_ratio = 1  # speech no need to upsample
    args.sinkhorn_tau = getattr(args, "sinkhorn_tau", 0.13)
    args.sinkhorn_iters = getattr(args, "sinkhorn_iters", 16)
    args.sinkhorn_noise_factor = getattr(args, "sinkhorn_noise_factor", 0.45)
    args.sinkhorn_bucket_size = getattr(args, "sinkhorn_bucket_size", 1)
    args.sinkhorn_energy = getattr(args, "sinkhorn_energy", "dot")
    args.mask_ratio = getattr(args, "mask_ratio", 0.5)

    st2t_transformer_s(args)


@register_model_architecture(
    "st2t_sinkhorn_encoder", "st2t_causal_encoder_s"
)
def st2t_causal_encoder_s(args):
    args.non_causal_layers = 0
    args.sinkhorn_tau = 1
    args.sinkhorn_iters = 1
    args.sinkhorn_noise_factor = 0
    args.sinkhorn_bucket_size = 1
    args.sinkhorn_energy = "dot"
    args.mask_ratio = 1
    args.mask_uniform = False
    args.upsample_ratio = 1

    st2t_transformer_s(args)
