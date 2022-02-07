#!/usr/bin/env python3

import logging
import math
import torch.nn.functional as F
from fairseq import checkpoint_utils
from fairseq.models import (
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    FairseqDropout
)
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerEncoder,
    S2TTransformerModel,
    s2t_transformer_s
)
from .s2t_transformer import make_conv_pos
from .causal_conv import CausalConv1dSubsampler
from .torchaudio_models import Emformer

logger = logging.getLogger(__name__)


#################################################
# block-causal transformer encoder #
#################################################
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

    # def forward(self, src_tokens, src_lengths):
    #     self._forward(src_tokens, src_lengths)

    #     from torch.profiler import profile, ProfilerActivity
    #     # record_shapes=True, with_stack=True,
    #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, profile_memory=True) as prof:
    #         net_out = self._forward(src_tokens, src_lengths)

    #     print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_memory_usage', row_limit=10))
    #     import pdb
    #     pdb.set_trace()

    def forward(self, src_tokens, src_lengths):
        """
        Emformer
        input (torch.Tensor)
            utterance frames right-padded with right context frames, with shape (B, T, D).
        lengths (torch.Tensor)
            with shape (B,) and i-th element representing number of valid frames for i-th batch element in input.
        -> Tuple[torch.Tensor, torch.Tensor]
            output frames, with shape (B, T - ``right_context_length`, D)`.
            output lengths, with shape (B,) and i-th element representing number of valid frames for i-th batch element in output frames.
        """
        # Step 1. extract features
        x, input_lengths = self.subsample(src_tokens, src_lengths)
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
        x = self.dropout_module(x)

        # Step 3. emformer forward
        assert x.size(1) == input_lengths.max().item() + self.right_context
        x, out_lengths = self.emformer_blocks(x, input_lengths)
        # B T C -> T B C
        x = x.transpose(0, 1)
        encoder_padding_mask = lengths_to_padding_mask(out_lengths)

        assert (out_lengths == input_lengths).all()
        return {
            "encoder_out": [x],
            "encoder_padding_mask": [encoder_padding_mask],
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        return S2TTransformerEncoder.reorder_encoder_out(self, encoder_out, new_order)


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
