#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import logging
from typing import Dict, List, Optional, Tuple
# from collections import OrderedDict
# import re
# import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from fairseq import checkpoint_utils
from fairseq.data.data_utils import lengths_to_padding_mask


from fairseq.models import (
    # FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
# from fairseq.modules import LayerNorm
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerEncoder,
    s2t_transformer_s,
)
from fairseq.models.speech_to_text.convtransformer import (
    ConvTransformerEncoder,
    base_architecture as convtransformer_base_architecture,
)

# user
from simultaneous_translation.models.nat_utils import generate
from simultaneous_translation.modules.monotonic_transformer_layer import (
    CausalTransformerEncoderLayer,
)

logger = logging.getLogger(__name__)


def create_encoder_model(klass):
    class SpeechEncoderModel(FairseqEncoderModel):
        """
        causal encoder + output projection
        """
        def __init__(self, encoder):
            super().__init__(encoder)
            self.one_pass_decoding = True  # must implement generate()

        @staticmethod
        def add_args(parser):
            """Add model-specific arguments to the parser."""
            super(SpeechEncoderModel, SpeechEncoderModel).add_args(parser)
            parser.add_argument(
                "--lookahead",
                type=int,
                help="number of hidden states speech encoder lags behind speech features for.",
            )
            parser.add_argument(
                "--load-pretrained-encoder-from",
                type=str,
                metavar="STR",
                help="model to take encoder weights from (for initialization)",
            )

        @classmethod
        def build_encoder(cls, args, src_dict, ctc_projection):
            encoder = klass(args, src_dict, ctc_projection)
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
            src_dict = task.source_dictionary
            ctc_projection = nn.Linear(
                args.encoder_embed_dim,
                len(src_dict),
                bias=False
            )
            nn.init.normal_(
                ctc_projection.weight, mean=0, std=args.encoder_embed_dim ** -0.5
            )
            encoder = cls.build_encoder(
                args, src_dict, ctc_projection)
            return cls(encoder)

        def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
        ):
            # net_output['encoder_out'] is a (B, T, D) tensor
            """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
            logits = net_output[0]

            if torch.is_tensor(logits):
                # syntactic sugar for simple models which don't have a decoder
                # (e.g., the classification tutorial)
                logits_f = logits.float()
                if log_probs:
                    lprobs = F.log_softmax(logits_f, dim=-1)
                else:
                    lprobs = F.softmax(logits_f, dim=-1)
            else:
                raise NotImplementedError

            return lprobs

        def forward(
            self, src_tokens, src_lengths,
            return_all_hiddens: bool = False,
            **unused,
        ):
            encoder_out = self.encoder(
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens
            )
            x = encoder_out["encoder_out"][0]  # speech hidden states
            x = self.encoder.ctc_projection(x)  # ctc projection
            x = x.transpose(1, 0)  # force batch first

            padding_mask = encoder_out["encoder_padding_mask"][0] \
                if len(encoder_out["encoder_padding_mask"]) > 0 else None
            extra = {
                "padding_mask": padding_mask,
                "encoder_out": encoder_out
            }
            return x, extra

        @property
        def output_layer(self):
            """ convenient function for accuracy calculation """
            return self.encoder.ctc_projection

        def generate(self, src_tokens, src_lengths, blank_idx=0, **unused):
            return generate(self, src_tokens, src_lengths, blank_idx=blank_idx)

        def max_decoder_positions(self):
            """Used by sequence generator."""
            return self.encoder.max_positions()

    SpeechEncoderModel.__name__ = klass.__name__
    return SpeechEncoderModel


def causal_encoder(klass):
    """ This decorator will override following functions
    for a speech encoder to provide streaming capability. """
    class CausalSpeechEncoder(klass):
        """Transformer encoder that consists of causal attention.
        """
        def __init__(self, args, src_dict, ctc_projection):
            super().__init__(args)
            self.transformer_layers = nn.ModuleList([
                CausalTransformerEncoderLayer(args, args.lookahead) for i in range(args.encoder_layers)
            ])
            self.src_dict = src_dict
            self.ctc_projection = ctc_projection

        def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
            raise NotImplementedError

        def forward(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,  # not used
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            incremental_step: Optional[int] = 1,
            return_all_hiddens: bool = False,
            padding_mask: Optional[torch.Tensor] = None,
            **unused
        ):
            """ similar to original S2TEncoder, added zeroing by mask and incremental encoding """
            x, input_lengths = self.subsample(src_tokens, src_lengths)
            x = self.embed_scale * x

            encoder_padding_mask = lengths_to_padding_mask(input_lengths)
            has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()
            # account for padding while computing the representation
            if has_pads:
                # T,B,C vs B,T
                assert x.size(1) == encoder_padding_mask.size(0)
                x = x * (1 - encoder_padding_mask.transpose(0, 1).unsqueeze(-1).type_as(x))
            # positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
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
                        timestep = encoder_padding_mask.size(
                            1) - incremental_step + i
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
                positions = positions.transpose(0, 1)

            if incremental_state is not None:
                x = x[-incremental_step:, :]
                if positions is not None:
                    positions = positions[-incremental_step:, :]

            if positions is not None:
                x += positions
            x = self.dropout_module(x)

            encoder_states = []

            for layer in self.transformer_layers:
                x = layer(
                    x,
                    encoder_padding_mask=encoder_padding_mask if has_pads else None,
                    incremental_state=incremental_state,
                )
                if return_all_hiddens:
                    encoder_states.append(x)

            if self.layer_norm is not None:
                x = self.layer_norm(x)

            return {
                "encoder_out": [x],  # T x B x C
                "encoder_padding_mask": [encoder_padding_mask] if has_pads else [],  # B x T
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
                end_id = len(self.transformer_layers)

            for index, layer in enumerate(self.transformer_layers):
                if index < end_id:
                    layer.prune_incremental_state(incremental_state, keep)

        def load_state_dict(self, state_dict, strict=True):
            """
            1. ignores ctc_projection if not available
            """
            ignores = ["ctc_projection.weight", ]
            cur_state_dict = self.state_dict()

            for w in ignores:
                if w not in state_dict:
                    logger.warning("Ignoring CTC projection weights! Make sure this is intended...")
                    state_dict[w] = cur_state_dict[w]

            return super().load_state_dict(state_dict, strict=strict)

    CausalSpeechEncoder.__name__ = klass.__name__
    return CausalSpeechEncoder


@causal_encoder
class S2TCausalSpeechEncoder(S2TTransformerEncoder):
    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        return self.subsample.get_out_seq_lens_tensor(in_seq_lens_tensor)


@causal_encoder
class ConvCausalSpeechEncoder(ConvTransformerEncoder):
    def dropout_module(self, x):
        return F.dropout(x, p=self.dropout, training=self.training)

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        def conv_forw(m, H):
            return (
                (H + 2 * m.padding[0] - m.dilation[0] * (m.kernel_size[0] - 1) - 1) / m.stride[0] + 1
            ).floor()

        out = in_seq_lens_tensor.clone()
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                out = conv_forw(m, out.float())
        return out.long()

    def subsample(self, src_tokens, src_lengths):
        bsz, max_seq_len, _ = src_tokens.size()
        x = (
            src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
            .transpose(1, 2)
            .contiguous()
        )
        x = self.conv(x)
        bsz, _, output_seq_len, _ = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous().view(output_seq_len, bsz, -1)
        x = self.out(x)

        subsampling_factor = int(max_seq_len * 1.0 / output_seq_len + 0.5)
        input_len_0 = (src_lengths.float() / subsampling_factor).ceil().long()
        input_len_1 = x.size(0) * torch.ones([src_lengths.size(0)]).long().to(
            input_len_0.device
        )
        input_lengths = torch.min(input_len_0, input_len_1)

        return x, input_lengths


@register_model("s2t_speech_encoder")
@create_encoder_model
class S2TSpeechEncoderModel(S2TCausalSpeechEncoder):
    pass


@register_model("conv_speech_encoder")
@create_encoder_model
class ConvSpeechEncoderModel(ConvCausalSpeechEncoder):
    pass


@register_model_architecture(
    "s2t_speech_encoder", "s2t_speech_encoder_s"
)
def s2t_speech_encoder_s(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True

    args.lookahead = getattr(args, "lookahead", 1)
    args.encoder_layers = getattr(args, "encoder_layers", 6)  # speech 6, text 6
    s2t_transformer_s(args)


@register_model_architecture(
    "conv_speech_encoder", "conv_speech_encoder_s"
)
def conv_speech_encoder_s(args):
    args.encoder_normalize_before = False
    args.decoder_normalize_before = False

    args.lookahead = getattr(args, "lookahead", 1)
    args.encoder_layers = getattr(args, "encoder_layers", 6)  # speech 6, text 6
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    # s2t_transformer_s(args)
    convtransformer_base_architecture(args)
