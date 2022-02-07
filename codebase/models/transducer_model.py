import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, List
from torch import Tensor
from fairseq import checkpoint_utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerDecoder,
    Linear
)
from codebase.models.s2t_emformer import (
    S2TEmformerModel,
    s2t_emformer_s
)
# from codebase.models.rnn_decoder import RNNDecoder

logger = logging.getLogger(__name__)


class SimpleJoiner(nn.Module):
    def __init__(self, args, output_projection=None):
        super().__init__()

        self.acoustic_weight = args.acoustic_weight

        assert args.encoder_embed_dim == args.decoder_embed_dim

        self.fuse_act_fn = nn.GELU()

        # emission prediction
        if args.predict_emission:
            emit_fc1 = Linear(args.encoder_embed_dim, 1, bias=True)
            emit_act_fn = nn.Tanh()
            emit_fc2 = nn.utils.weight_norm(
                nn.Linear(1, 1, bias=True), name="weight", dim=-1)
            nn.init.constant_(emit_fc2.bias, -5.)
            self.emit_projection = nn.Sequential(
                emit_fc1,
                emit_act_fn,
                emit_fc2
            )
        else:
            self.emit_projection = None

        # word prediction
        self.output_projection = output_projection

    def forward(self, src_feats, tgt_feats, target_masks=None):
        S, B, C = src_feats.size()
        assert tgt_feats.shape[0] == B
        T = tgt_feats.size(1)

        # tgt B T C -> B, T, S, C
        tgt_feats = tgt_feats.unsqueeze(2)

        # src S B C -> B, T, S, C
        src_feats = src_feats.transpose(0, 1).unsqueeze(1)

        if self.training:
            tgt_feats.register_hook(lambda g: g / S)
            src_feats.register_hook(lambda g: g / T)

        # combine
        fused_feats = src_feats * self.acoustic_weight + tgt_feats

        if target_masks is not None:
            # B, T, S, C -> T_flat, S, C
            fused_feats = fused_feats[target_masks]

        fused_feats = self.fuse_act_fn(fused_feats)

        # log_alpha
        log_emit = None
        if self.emit_projection is not None:
            log_emit = self.emit_projection(fused_feats).squeeze(-1)
            if self.training:
                log_emit += torch.randn_like(log_emit)
        logits = self.output_projection(fused_feats)
        return logits, log_emit


class TransducerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, output_projection=None):
        super().__init__(args, dictionary, embed_tokens, True, output_projection)
        self.downsample = max(args.downsample, 1)
        self.joiner = SimpleJoiner(args, self.output_projection)
        self.padding_idx = dictionary.pad()
        self.memory_efficient = args.memory_efficient

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        # target_masks: Optional[Tensor] = None,
        # memory_efficient: Optional[bool] = False,
        **unused
    ):
        src_feats = encoder_out["encoder_out"][0]  # T B C
        padding_mask = encoder_out["encoder_padding_mask"][0]  # B, T
        src_feats = src_feats[::self.downsample]
        if padding_mask is not None:
            padding_mask = padding_mask[:, ::self.downsample]

        tgt_feats, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=None,
            incremental_state=incremental_state,
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            input_buffer = self._get_input_buffer(incremental_state)
            prev_emit = input_buffer.get("prev_emit", None)

        target_masks = prev_output_tokens.ne(
            self.padding_idx) if self.memory_efficient else None
        logits, log_emit = self.joiner(
            src_feats,
            tgt_feats,
            target_masks=target_masks,
            # memory_efficient=memory_efficient
        )

        if incremental_state is not None:
            S, B, C = src_feats.size()
            V = logits.size(-1)
            # normal: (B, 1, S) since T == 1
            # mem_ef: (B, S) since T_flat == B
            if not self.memory_efficient:
                log_emit = log_emit.squeeze(1)
                logits = logits.squeeze(1)
            # force emit at source eos
            if padding_mask is not None:
                source_eos = (~padding_mask).long().sum(-1) - 1
            else:
                source_eos = log_emit.new_full((B,), S - 1).long()
            log_emit.scatter_(
                1,
                source_eos.view(B, 1),
                1e4
            )
            if prev_emit is not None:
                # mask past
                mask = (
                    torch
                    .arange(S)
                    .to(log_emit.device)
                    .view(1, S)
                    .expand(B, -1)
                ) < prev_emit.view(B, 1).expand(-1, S)
                log_emit[mask] = -1e4

            # .sigmoid()
            # .ge(0.5)
            new_emit = (
                log_emit
                .ge(0.0)
                .cumsum(1)
                .eq(1)
                .long()
                .argmax(1)
            )

            # B, S, V -> B, 1, V
            logits = logits.gather(
                1,
                new_emit.view(B, 1, 1).expand(-1, -1, V)
            )

            input_buffer["prev_emit"] = new_emit
            self._set_input_buffer(incremental_state, input_buffer)

        extra["log_emit"] = log_emit
        extra["padding_mask"] = padding_mask
        return logits, extra

    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                if input_buffer[k] is not None:
                    input_buffer[k] = input_buffer[k].index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "joiner_state")
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
        return self.set_incremental_state(incremental_state, "joiner_state", buffer)


@register_model("transducer_model")
class TransducerModel(S2TEmformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(TransducerModel,
              TransducerModel).add_args(parser)
        parser.add_argument(
            "--downsample",
            type=int,
        )
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder (output_projection) weights from (for initialization)."
        )
        parser.add_argument(
            "--acoustic-weight",
            type=float,
            help="increase weight of acoustic model over lm"
        )
        parser.add_argument(
            "--predict-emission",
            action="store_true",
            help="if ssnt, need to predict emission"
        )

    @classmethod
    def build_decoder(cls, args, task, decoder_embed_tokens):
        decoder = TransducerDecoder(args, task.target_dictionary, decoder_embed_tokens)
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
    "transducer_model", "ssnt_model_s"
)
def transducer_model_s(args):
    args.downsample = getattr(args, "downsample", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.acoustic_weight = getattr(args, "acoustic_weight", 4)
    args.predict_emission = True
    s2t_emformer_s(args)
