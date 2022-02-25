#!/usr/bin/env python3

import logging
import math
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict
from fairseq import checkpoint_utils
from fairseq.models import (
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    FairseqDropout
)
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerEncoder,
    S2TTransformerModel,
    s2t_transformer_s
)
from codebase.models.s2t_transformer import make_conv_pos
from codebase.models.causal_conv import CausalConv1dSubsampler
from codebase.models.torchaudio_models import Emformer

logger = logging.getLogger(__name__)


#################################################
# block-causal transformer encoder #
#################################################
@with_incremental_state
class S2TEmformerEncoder(FairseqEncoder):
    def __init__(self, args):
        super().__init__(args)

        self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.subsample = CausalConv1dSubsampler(
            args.input_feat_per_channel * args.input_channels,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )

        self.embed_positions = make_conv_pos(
            args.encoder_embed_dim,
            args.conv_pos,
            args.conv_pos_groups,
            causal=True
        )

        stride = self.conv_layer_stride()
        self.stride = stride
        self.left_context = args.segment_left_context // stride
        self.right_context = args.segment_right_context // stride
        self.segment_length = args.segment_length // stride

        self.emformer_blocks = Emformer(
            input_dim=args.encoder_embed_dim,
            num_heads=args.encoder_attention_heads,
            ffn_dim=args.encoder_ffn_embed_dim,
            num_layers=args.encoder_layers,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            activation='gelu',  # args.activation_fn,
            left_context_length=self.left_context,
            right_context_length=self.right_context,
            segment_length=self.segment_length,
            max_memory_size=args.max_memory_size,
            tanh_on_mem=args.tanh_on_mem,
            negative_inf=-1e4 if args.fp16 else -1e8,
            weight_init_scale_strategy='depthwise',
            normalize_before=args.encoder_normalize_before
        )

    def conv_layer_stride(self):
        stride = 1
        for c in self.subsample.conv_layers:
            stride *= c.stride[0]
        return stride

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, src_tokens, src_lengths):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(src_tokens, src_lengths)
        else:
            x = self._forward(src_tokens, src_lengths)
        return x

    def _forward(self, src_tokens, src_lengths):
        """
        Emformer
        input (torch.Tensor)
            utterance frames right-padded with right context frames, with shape (B, T, D).
        lengths (torch.Tensor)
            with shape (B,) and i-th element representing number of valid frames for i-th batch element in input.
        -> Tuple[torch.Tensor, torch.Tensor]
            output frames, with shape (B, T - ``right_context_length`, D)`.
            output lengths, with shape (B,) and representing number of valid frames in output frames.
        """
        # Step 1. extract features
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        # encoder_embedding = x
        x = self.embed_scale * x

        # Step 2. add padding and positions
        # T B C -> B C T
        x = x.permute(1, 2, 0)
        # right-padding
        x = F.pad(x, (0, self.right_context))
        # add position
        x += self.embed_positions(x)
        # B C T -> B T C
        x = x.transpose(2, 1)
        encoder_embedding = x.transpose(0, 1).clone()
        x = self.dropout_module(x)

        # Step 3. emformer forward
        assert x.size(1) == input_lengths.max().item() + self.right_context
        x, out_lengths, encoder_states = self.emformer_blocks(x, input_lengths)
        # B T C -> T B C
        x = x.transpose(0, 1)
        encoder_padding_mask = lengths_to_padding_mask(out_lengths)

        assert (out_lengths == input_lengths).all()
        return {
            "encoder_out": [x],
            "encoder_padding_mask": [encoder_padding_mask],
            "encoder_embedding": [encoder_embedding],
            "encoder_states": encoder_states,
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        return S2TTransformerEncoder.reorder_encoder_out(self, encoder_out, new_order)

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "emformer_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "emformer_state", buffer)

    def infer(self, src_tokens, src_lengths, incremental_state, finish=False):
        assert src_tokens.size(0) == 1, "batched streaming not supported yet"
        import math
        t = int(math.ceil((src_lengths.item() - 96) / 64))
        print(f'({t}) 0: {src_tokens.size(1)}')

        x, input_lengths = self.subsample(
            src_tokens, src_lengths, incremental_state, finish=finish)
        # encoder_embedding = x.clone()  # ok
        x = self.embed_scale * x
        print(f'({t}) 1: {x.size(0)} {input_lengths.item()}')

        # T B C -> B C T
        x = x.permute(1, 2, 0)
        # right-padding
        if finish:
            print("=======================finish=======================")
            x = F.pad(x, (0, self.right_context))
        print(f'({t}) 2: {x.size(2)}')
        # add position
        x = x + self.embed_positions(x, incremental_state)  # += causes bug in inference
        # B C T -> B T C
        x = x.transpose(2, 1)
        print(f'({t}) 3: {x.size(1)}')

        encoder_embedding = x.transpose(0, 1).clone()  # ok

        block_rc_len = input_lengths
        saved_state = self._get_input_buffer(incremental_state)
        # we need to carry the last segment's rc to this segment
        if "carry" in saved_state:
            carry = saved_state["carry"]
            x = torch.cat((carry, x), dim=1)
            block_rc_len = input_lengths + carry.size(1)
        print(f'({t}) 4: {x.size(1)}')

        # compute carry segment
        # next_carry_len = block_rc_len - self.segment_length
        carry = x[:, self.segment_length:, :]

        # current input size is length + rc
        carry_len = torch.zeros_like(block_rc_len)
        if block_rc_len.item() > self.segment_length:
            carry_len = block_rc_len - self.segment_length
            x = x[:, :self.segment_length + self.right_context, :]
            block_rc_len[0] = x.size(1)

        # retrieve prev states
        states = None
        if "prev_state" in saved_state:
            states = saved_state["prev_state"]
            assert states is not None

        print(f'({t}) 5: {x.size(1)}')
        # emformer forward
        x, out_lengths, encoder_states = self.emformer_blocks.infer(x, block_rc_len, states)
        print(f'({t}) 6: {x.size(1)} {out_lengths.item()}')

        # cache the carry & state for next segment
        saved_state["carry"] = carry
        saved_state["prev_state"] = encoder_states
        self._set_input_buffer(incremental_state, saved_state)

        # extra forward for last segment
        if finish and carry_len.item() > 0:
            assert carry.size(1) == carry_len.item() + self.right_context, (
                f"{carry.size(1)} == {carry_len.item()} + {self.right_context}")
            assert carry.numel() > 0
            carry_len = carry_len + self.right_context
            print(f'({t}) rc: {carry.size(1)} {carry_len.item()}')
            rc, rc_lengths, rc_states = self.emformer_blocks.infer(carry, carry_len, encoder_states)
            print(f'({t}) rc: {rc.size(1)} {rc_lengths.item()}')
            x = torch.cat((x, rc), dim=1)
            out_lengths = out_lengths + rc_lengths
            print(f'({t}) 7: {x.size(1)} {out_lengths.item()}')
            # carry_size = block_rc_len.fill_(self.right_context)
            # print(f'({t}) rc1: {carry.size(1)} {carry_size.item()}')
            # carry = F.pad(carry, (0, 0, 0, 2 * self.right_context - carry.size(1)))
            # print(f'({t}) rc2: {carry.size(1)}')
            # rc, rc_lengths, rc_states = self.emformer_blocks.infer(carry, carry_size, encoder_states)
            # x = torch.cat((x, rc), dim=1)
            # out_lengths = out_lengths + rc_lengths
            # print(f'({t}) 7: {x.size(1)} {out_lengths.item()}')

        # B T C -> T B C
        x = x.transpose(0, 1)
        encoder_padding_mask = lengths_to_padding_mask(out_lengths)

        return {
            "encoder_out": [x],
            "encoder_padding_mask": [encoder_padding_mask],
            "encoder_embedding": [encoder_embedding],
            "encoder_states": encoder_states,
            "src_tokens": [],
            "src_lengths": [],
        }


@register_model("s2t_emformer")
class S2TEmformerModel(S2TTransformerModel):
    @staticmethod
    def add_args(parser):
        super(S2TEmformerModel, S2TEmformerModel).add_args(parser)
        # conv pos
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
        # emformer
        parser.add_argument(
            "--segment-length",
            type=int,
            metavar="N",
            help="length of each segment (not including left context / right context)",
        )
        parser.add_argument(
            "--segment-left-context",
            type=int,
            help="length of left context in a segment",
        )
        parser.add_argument(
            "--segment-right-context",
            type=int,
            help="length of right context in a segment",
        )
        parser.add_argument(
            "--max-memory-size",
            type=int,
            help="Right context for the segment.",
        )
        parser.add_argument(
            "--tanh-on-mem",
            action="store_true",
            help="whether to use tanh for memory bank vectors.",
        )

    @classmethod
    def build_encoder(cls, args):
        encoder = S2TEmformerEncoder(args)

        if getattr(args, "load_pretrained_encoder_from", None) is not None:
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        return encoder


@register_model_architecture(
    "s2t_emformer", "s2t_emformer_s"
)
def s2t_emformer_s(args):
    # positional embeddings
    args.conv_pos = getattr(args, "conv_pos", 128)
    args.conv_pos_groups = getattr(args, "conv_pos_groups", 16)
    # emformer
    args.segment_length = getattr(args, "segment_length", 64)
    args.segment_left_context = getattr(args, "segment_left_context", 128)
    args.segment_right_context = getattr(args, "segment_right_context", 32)
    args.max_memory_size = getattr(args, "max_memory_size", 5)  # 3 ~ 5 is good
    args.tanh_on_mem = getattr(args, "tanh_on_mem", True)  # if False, hard clipping to +-10 is used.
    s2t_transformer_s(args)
