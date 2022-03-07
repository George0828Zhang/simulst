# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional
import logging
from torch import Tensor
from fairseq import checkpoint_utils
from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairseq.models.transformer import (
    TransformerDecoder
)
from fairseq.modules import TransformerDecoderLayer
from codebase.models.s2t_emformer import (
    S2TEmformerModel,
    s2t_emformer_s
)
from codebase.modules import (
    build_monotonic_attention
)

logger = logging.getLogger(__name__)


class MMADecoderLayer(TransformerDecoderLayer):
    def build_encoder_attention(self, embed_dim, args):
        assert hasattr(args, "simul_attn_type"), "Use --simul-attn-type to specify attention type."
        return build_monotonic_attention(args)

    def prune_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ):
        input_buffer = self.self_attn._get_input_buffer(incremental_state)
        for key in ["prev_key", "prev_value"]:
            input_buffer_key = input_buffer[key]
            assert input_buffer_key is not None
            if input_buffer_key.size(2) > 1:
                input_buffer[key] = input_buffer_key[:, :, :-1, :]
            else:
                typed_empty_dict: Dict[str, Optional[Tensor]] = {}
                input_buffer = typed_empty_dict
                break
        assert incremental_state is not None
        self.self_attn._set_input_buffer(incremental_state, input_buffer)


class MMADecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, output_projection=None):
        super().__init__(args, dictionary, embed_tokens, False, output_projection)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return MMADecoderLayer(args, no_encoder_attn)

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def pre_attention(
        self,
        prev_output_tokens,
        encoder_out_dict: Dict[str, List[Tensor]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        positions = (
            self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_out = encoder_out_dict["encoder_out"][0]

        if "encoder_padding_mask" in encoder_out_dict:
            encoder_padding_mask = (
                encoder_out_dict["encoder_padding_mask"][0]
                if encoder_out_dict["encoder_padding_mask"]
                and len(encoder_out_dict["encoder_padding_mask"]) > 0
                else None
            )
        else:
            encoder_padding_mask = None

        return x, encoder_out, encoder_padding_mask

    def post_attention(self, x):
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x

    def clear_cache(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        end_id: Optional[int] = None,
    ):
        """
        Clean cache in the monotonic layers.
        The cache is generated because of a forward pass of decoder has run but no prediction,
        so that the self attention key value in decoder is written in the incremental state.
        end_id is the last idx of the layers
        """
        if end_id is None:
            end_id = len(self.layers)

        for index, layer in enumerate(self.layers):
            if index < end_id:
                layer.prune_incremental_state(incremental_state)

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,  # unused
        alignment_layer: Optional[int] = None,  # unused
        alignment_heads: Optional[int] = None,  # unsed
    ):
        # incremental_state = None
        assert encoder_out is not None
        (x, encoder_outs, encoder_padding_mask) = self.pre_attention(
            prev_output_tokens, encoder_out, incremental_state
        )
        attn = None
        inner_states = [x]
        attn_list: List[Optional[Dict[str, Tensor]]] = []

        # p_choose = torch.tensor([1.0])

        for i, layer in enumerate(self.layers):

            x, attn, _ = layer(
                x=x,
                encoder_out=encoder_outs,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                self_attn_mask=self.buffered_future_mask(x)
                if incremental_state is None
                else None,
            )

            inner_states.append(x)
            attn_list.append(attn)  # a dict with alpha, beta and p_choose

            if incremental_state is not None:
                if_online = incremental_state.get("online", False)
                if if_online:
                    # Online indicates that the encoder states are still changing
                    # Any head decide to read than read
                    head_read = layer.encoder_attn._get_monotonic_buffer(incremental_state)["head_read"]
                    assert head_read is not None
                    if head_read.any():
                        # We need to prune the last self_attn saved_state
                        # if model decide not to read
                        # otherwise there will be duplicated saved_state
                        self.clear_cache(incremental_state, i + 1)

                        return x, {
                            "action": 0,
                            "attn": [None],
                            "attn_list": attn_list,
                            "encoder_out": encoder_out,
                            "inner_states": inner_states
                        }

        x = self.post_attention(x)

        return x, {
            "action": 1,
            "attn": [None],
            "attn_list": attn_list,
            "encoder_out": encoder_out,
            "encoder_padding_mask": encoder_padding_mask
        }


@register_model("mma_model")
class MMAModel(S2TEmformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(MMAModel,
              MMAModel).add_args(parser)
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder (output_projection) weights from (for initialization)."
        )

    @classmethod
    def build_decoder(cls, args, task, decoder_embed_tokens):
        decoder = MMADecoder(args, task.target_dictionary, decoder_embed_tokens)
        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
            logger.info(
                f"loaded pretrained decoder from: "
                f"{args.load_pretrained_decoder_from}"
            )
        return decoder


@register_model_architecture(
    "mma_model", "mma_model_s"
)
def ssnt_model_s(args):
    args.noise_var = getattr(args, "noise_var", 2.0)
    args.noise_mean = getattr(args, "noise_mean", 0.0)
    args.energy_bias_init = getattr(args, "energy_bias_init", -2.0)
    args.attention_eps = getattr(args, "attention_eps", 1e-10)
    s2t_emformer_s(args)
