#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Optional
from collections import OrderedDict
import re
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
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerModel,
    s2t_transformer_s,
)

# user
from simultaneous_translation.models.speech_encoder import CausalSpeechEncoder
from simultaneous_translation.modules import (
    CausalTransformerEncoderLayer,
)

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
        sp_encoder.apply(init_bert_params)
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
            tx_encoder.apply(init_bert_params)
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
        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        ctc_projection = nn.Linear(
            args.encoder_embed_dim,
            len(task.source_dictionary),
            bias=False
        )
        encoder_embed_tokens = build_embedding(
            task.source_dictionary, args.encoder_embed_dim
        )
        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder = cls.build_encoder(args, task, encoder_embed_tokens, ctc_projection)
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
                encoder_padding_mask=encoder_padding_mask,
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

    def load_state_dict(self, state_dict, strict=True):
        """
        1. remove ``causal_encoder'' from the state_dict keys.
        2. ignores upsampler and decoder_embed.
        """
        changes = re.compile("causal_encoder.")
        ignores = ["upsampler", "decoder_embed"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if any([i in k for i in ignores]):
                continue
            new_state_dict[changes.sub("", k)] = v

        return super().load_state_dict(new_state_dict, strict=strict)


class SpeechTextCascadedEncoder(FairseqEncoder):
    """
    Takes speech encoder and text encoder and make a cascade
    """

    def __init__(self, args, speech_encoder, text_encoder=None):
        super().__init__(None)
        self.speech_encoder = speech_encoder
        self.text_encoder = text_encoder
        self.do_weighted_shrink = args.do_weighted_shrink

    @property
    def src_dict(self):
        return self.speech_encoder.src_dict

    def reorder_encoder_out(self, encoder_out, new_order):
        return self.speech_encoder.reorder_encoder_out(encoder_out, new_order)

    def forward(
        self,
        src_tokens,
        src_lengths,
        src_txt_tokens,  # unused
        src_txt_lengths,  # unused
    ):
        # encode speech
        encoder_out = self.speech_encoder(
            src_tokens=src_tokens, src_lengths=src_lengths)
        encoder_out = self.weighted_shrinking(encoder_out)
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
        logits = self.speech_encoder.ctc_projection(
            speech_states).transpose(0, 1)

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
            segment_ids = segment_ids.clone()
            segment_ids[speech_padding_mask] = -1
            shrink_lengths = segment_ids.max(dim=-1)[0] + 1
            shrinked_states = shrinked_states[:shrink_lengths.max(), ...]
            del segment_ids

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
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    s2t_transformer_s(args)
