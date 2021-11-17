#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import logging
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models.transformer import (
    Embedding,
    TransformerEncoder
)
from fairseq.models import (
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules.fairseq_dropout import FairseqDropout
# from fairseq.models.speech_to_text.s2t_transformer import (
#     S2TTransformerModel,
#     s2t_transformer_s,
# )
from fairseq.models.speech_to_text.convtransformer import (
    ConvTransformerModel,
    base_architecture as convtransformer_base_architecture,
)

# user
from simultaneous_translation.models.speech_encoder import CausalSpeechEncoder
from simultaneous_translation.modules.monotonic_transformer_layer import (
    CausalTransformerEncoderLayer,
)

logger = logging.getLogger(__name__)


def nan_warn(t: Tensor, name: str):
    # assert not shrinked_states.isnan().any()
    if t.isnan().any():
        logger.warning(f"NaN detected in tensor named: {name}")


@register_model("st2t_transformer")
class ST2TTransformerModel(ConvTransformerModel):
    """
    causal encoder (+ semantic encoder) + normal decoder
    """
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser.
        """
        super(ST2TTransformerModel,
              ST2TTransformerModel).add_args(parser)
        parser.add_argument(
            "--lookahead",
            type=int,
            help="number of hidden states speech encoder lags behind speech features for.",
        )
        parser.add_argument(
            "--weight-norm-conv",
            action="store_true",
            help="apply weight normalization for convolution weights.",
        )
        parser.add_argument(
            "--do-weighted-shrink",
            action="store_true",
            default=False,
            help="shrink the encoder states based on ctc output.",
        )
        parser.add_argument(
            "--fixed-shrink-ratio",
            type=int,
            default=1,
            help="shrink speech encoder output as fixed-length segments.",
        )
        parser.add_argument(
            "--text-encoder-layers",
            type=int,
            help="number of  layers for text (semantic) encoder, after speech encoder.",
        )
        parser.add_argument(
            "--load-pretrained-text-encoder-from",
            type=str,
            metavar="STR",
            help="model to take text encoder weights from (for initialization)",
        )

    @classmethod
    def build_encoder(cls, args, task, embed_tokens, ctc_projection):
        sp_encoder = CausalSpeechEncoder(
            args, task.source_dictionary, ctc_projection)
        if getattr(args, "load_pretrained_encoder_from", None):
            sp_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=sp_encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained speech encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        tx_encoder = None
        if args.text_encoder_layers > 0:
            tx_encoder = CausalTransformerEncoder(
                args, task.source_dictionary, embed_tokens)
            if getattr(args, "load_pretrained_text_encoder_from", None):
                tx_encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=tx_encoder, checkpoint=args.load_pretrained_text_encoder_from
                )
                logger.info(
                    f"loaded pretrained text encoder from: "
                    f"{args.load_pretrained_text_encoder_from}"
                )
        return SpeechTextCascadedEncoder(args, sp_encoder, tx_encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

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
        encoder = cls.build_encoder(args, task, encoder_embed_tokens, ctc_projection)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder)

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
            src_lengths=src_lengths
        )
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


class CausalTransformerEncoder(TransformerEncoder):
    """Transformer encoder that consists of causal attention.
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.layers = nn.ModuleList([
            CausalTransformerEncoderLayer(args) for i in range(args.text_encoder_layers)
        ])

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,  # not used
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        incremental_step: Optional[int] = 1,
        return_all_hiddens: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        1. supports passing speech encoder states as src_tokens
        2. supports passing tokens as src_tokens
        3. supports incremental encoding
        """
        if src_tokens.dim() == 2:
            # case 1: src_tokens are actually tokens
            encoder_padding_mask = src_tokens.eq(self.padding_idx)
        elif src_tokens.dim() == 3:
            # case 2: src_tokens are speech encoder states
            B, S, C = src_tokens.size()
            encoder_padding_mask = padding_mask if padding_mask is not None else src_tokens.new_zeros(
                (B, S)).bool()
        else:
            raise RuntimeError(
                f"invalid argument dimension src_tokens = {src_tokens.dim}.")

        # x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
        # embed positions
        positions = None
        if self.embed_positions is not None:
            # incremental_state for embed_positions is designed for single step.
            # # slow
            # positions = self.embed_positions(
            #     src_tokens,  # incremental_state=incremental_state
            # )
            # fast
            positions = self.embed_positions(
                encoder_padding_mask,
                incremental_state=incremental_state,
                timestep=torch.LongTensor(
                    [encoder_padding_mask.size(1) - incremental_step])
            )
            if incremental_step > 1:
                for i in range(1, incremental_step):
                    timestep = encoder_padding_mask.size(1) - incremental_step + i
                    positions = torch.cat(
                        (
                            positions,
                            self.embed_positions(
                                encoder_padding_mask,
                                incremental_state=incremental_state,
                                timestep=torch.LongTensor([timestep])
                            )
                        ), dim=1
                    )

        if incremental_state is not None:
            src_tokens = src_tokens[:, -incremental_step:]
            if positions is not None:
                positions = positions[:, -incremental_step:]

        if src_tokens.dim() == 2:
            # case 1: src_tokens are actually tokens
            encoder_embedding = self.embed_tokens(src_tokens)
        elif src_tokens.dim() == 3:
            # case 2: src_tokens are speech encoder states
            encoder_embedding = src_tokens
        else:
            raise RuntimeError(
                f"invalid argument dimension src_tokens = {src_tokens.dim}.")

        # embed tokens and positions
        x = self.embed_scale * encoder_embedding

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_padding_mask=encoder_padding_mask if encoder_padding_mask.any() else None,
                incremental_state=incremental_state,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
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


class SpeechTextCascadedEncoder(FairseqEncoder):
    """
    Takes speech encoder and text encoder and make a cascade
    """

    def __init__(self, args, speech_encoder, text_encoder=None):
        super().__init__(None)
        self.speech_encoder = speech_encoder
        self.text_encoder = text_encoder
        self.do_weighted_shrink = args.do_weighted_shrink
        self.fixed_shrink_ratio = args.fixed_shrink_ratio
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

    @property
    def src_dict(self):
        return self.speech_encoder.src_dict

    def reorder_encoder_out(self, encoder_out, new_order):
        return self.speech_encoder.reorder_encoder_out(encoder_out, new_order)

    def forward(
        self,
        src_tokens,
        src_lengths,
        src_txt_tokens=None,  # unused
        src_txt_lengths=None,  # unused
    ):
        # encode speech
        encoder_out = self.speech_encoder(
            src_tokens=src_tokens, src_lengths=src_lengths)
        encoder_out = self.forward_ctc_projection(encoder_out)
        encoder_out = self.shrink_speech(encoder_out)
        # encode text
        if self.text_encoder is not None:
            src_tokens = encoder_out["encoder_out"][0].transpose(0, 1)
            padding_mask = encoder_out["encoder_padding_mask"][0]
            text_out = self.text_encoder(
                src_tokens,
                padding_mask=padding_mask,
            )
            # update text output to encoder_out
            for key in ["encoder_out", "encoder_states"]:
                encoder_out[key] = text_out[key]
        return encoder_out

    def forward_ctc_projection(self, encoder_out):
        speech_states = encoder_out["encoder_out"][0]
        # ctc projection
        logits = self.speech_encoder.ctc_projection(
            speech_states).transpose(0, 1)

        encoder_out.update({
            "encoder_logits": [logits],
            "speech_padding_mask": encoder_out["encoder_padding_mask"]
        })
        return encoder_out

    def shrink_speech(self, encoder_out):
        # speech mask
        speech_padding_mask = encoder_out["encoder_padding_mask"][0] \
            if len(encoder_out["encoder_padding_mask"]) > 0 else None

        # speech hidden states
        speech_states = encoder_out["encoder_out"][0]
        logits = encoder_out["encoder_logits"][0]

        B, S, C = logits.size()
        shrinked_states, shrink_lengths = self.shrinking_op(
            speech_states,
            logits,
            speech_padding_mask
        )

        encoder_padding_mask = lengths_to_padding_mask(shrink_lengths)
        # calculate shrink rate
        src_lengths = shrink_lengths.new_full(
            (B,), S) if speech_padding_mask is None else (~speech_padding_mask).sum(-1).type_as(shrink_lengths)
        shrink_rate = (src_lengths / shrink_lengths).sum()
        return {
            "encoder_out": [shrinked_states],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_out["encoder_states"],  # List[T x B x C]
            "encoder_logits": [logits],
            "speech_padding_mask": [speech_padding_mask],
            "src_tokens": [],
            "src_lengths": [],
            "shrink_rate": [shrink_rate]
        }

    @property
    def shrinking_op(self):
        if self.do_weighted_shrink:
            return self._weighted_shrink
        else:
            return self._fixed_shrink

    def _fixed_shrink(self, speech_states, logits, speech_padding_mask=None):
        S, B, E = speech_states.size()
        if speech_padding_mask is not None:
            assert tuple(speech_padding_mask.shape[:2]) == (B, S)
        else:
            speech_padding_mask = speech_states.new_zeros(
                (B, S), dtype=torch.bool)

        ratio = self.fixed_shrink_ratio
        v = speech_states.transpose(0, 1)
        new_speech_padding_mask = speech_padding_mask
        buckets = math.ceil(S / ratio)

        # pad key value
        new_S = buckets * ratio
        if new_S != S:
            v = torch.cat([
                v,
                v.new_zeros((B, new_S - S, E)),
            ], dim=1)
            new_speech_padding_mask = torch.cat([
                speech_padding_mask,
                speech_padding_mask.new_ones((B, new_S - S)),
            ], dim=1)

        # remove padded states' influence
        v = v * (1 - new_speech_padding_mask.unsqueeze(-1).type_as(v))

        # aggregate buckets (B, new_S, E) -> (B, buckets, E)
        shrinked_states = v.view(
            B, buckets, ratio, E).mean(dim=2).transpose(1, 0)
        # aggregate padding mask by: if a bucket is all pad then it is masked.
        new_speech_padding_mask = new_speech_padding_mask.view(
            B, buckets, ratio).prod(dim=2).bool()
        shrink_lengths = (~new_speech_padding_mask).sum(-1)

        assert shrinked_states.size(1) == B
        # assert not shrinked_states.isnan().any()
        nan_warn(shrinked_states, "fixed_shrink")

        return shrinked_states, shrink_lengths

    def _weighted_shrink(self, speech_states, logits, speech_padding_mask=None):
        blank = self.src_dict.bos()
        pad = self.src_dict.pad()

        B, S, C = logits.size()
        if speech_padding_mask is not None:
            # replace paddings' logits s.t. they predict pads
            one_hot = F.one_hot(torch.LongTensor([pad]), C).type_as(logits)
            logits[speech_padding_mask] = one_hot

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
        shrink_lengths = segment_ids.max(dim=1)[0] + 1
        neg_inf = -torch.finfo(logits.dtype).max
        attn_weights = logits.new_full((B, shrink_lengths.max(), S), neg_inf)
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
        ).type_as(speech_states)
        attn_weights = self.dropout_module(attn_weights)

        # shrink speech states
        # (B, S', S) x (B, S, D) -> (B, S', D) -> (S', B, D)
        shrinked_states = torch.bmm(
            attn_weights,
            speech_states.transpose(0, 1)
        ).transpose(0, 1)

        assert shrinked_states.size(1) == B
        # assert not shrinked_states.isnan().any()
        nan_warn(shrinked_states, "ws_shrink")
        if speech_padding_mask is not None:
            # pad states are shrunk to a segment
            # remove this 'pad segment'
            # first set pad segment to 0
            segment_ids = segment_ids * (1 - speech_padding_mask.long())
            real_shrink_lengths = segment_ids.max(dim=1)[0] + 1
            if real_shrink_lengths.max() < shrink_lengths.max():
                shrink_lengths = real_shrink_lengths
                shrinked_states = shrinked_states[:shrink_lengths.max(), ...]
            del segment_ids

        return shrinked_states, shrink_lengths


@register_model_architecture(
    "st2t_transformer", "st2t_transformer_s"
)
def st2t_transformer_s(args):
    args.encoder_normalize_before = False
    args.decoder_normalize_before = False

    args.lookahead = getattr(args, "lookahead", 1)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.do_weighted_shrink = getattr(args, "do_weighted_shrink", False)
    if args.do_weighted_shrink:
        args.fixed_shrink_ratio = 1
    else:
        getattr(args, "fixed_shrink_ratio", 1)
    args.share_decoder_input_output_embed = True

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    convtransformer_base_architecture(args)
