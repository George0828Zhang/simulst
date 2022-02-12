import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
from torch import Tensor
from fairseq import checkpoint_utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerDecoder,
)
from codebase.models.s2t_emformer import (
    S2TEmformerModel,
    s2t_emformer_s
)

logger = logging.getLogger(__name__)


class InplaceTanh(nn.Module):
    def forward(self, x):
        return x.tanh_()


class SimpleJoiner(nn.Module):
    def __init__(self, args, output_projection=None):
        super().__init__()

        hidden_dim = args.decoder_embed_dim

        # transcriber projection
        self.source_projection = nn.Linear(
            args.encoder_embed_dim,
            hidden_dim,
            bias=True,
        )
        self.target_projection = nn.Linear(
            args.decoder_embed_dim,
            hidden_dim,
            bias=False,
        )

        # in-place & inexpensive is preferred.
        self.join_act = InplaceTanh()
        # word prediction
        self.output_projection = output_projection

        nn.init.xavier_uniform_(
            self.source_projection.weight,
            gain=(args.encoder_layers + 1) ** -0.5
        )
        nn.init.xavier_uniform_(
            self.target_projection.weight,
            gain=(args.decoder_layers + 1) ** -0.5
        )

    def forward(self, src_feats, tgt_feats):
        assert src_feats.size(1) == tgt_feats.size(0)

        # src S B C -> B, S, T, C
        src_feats = self.source_projection(src_feats)
        src_feats = src_feats.transpose(0, 1).unsqueeze(2)

        # tgt B T C -> B, S, T, C
        tgt_feats = self.target_projection(tgt_feats)
        tgt_feats = tgt_feats.unsqueeze(1)

        # join
        join_feats = src_feats + tgt_feats
        join_feats = self.join_act(join_feats)
        logits = self.output_projection(join_feats)

        return logits


class AvgPool1dTBCPad(nn.AvgPool1d):
    def forward(self, x, padding_mask=None):
        T, B, C = x.size()
        k = self.kernel_size[0]
        if padding_mask is not None:
            lengths = (~padding_mask).sum(1)
            x[padding_mask.transpose(1, 0)] = 0
            padding_mask = padding_mask[:, ::k]
        x = super().forward(
            x.permute(1, 2, 0)).permute(2, 0, 1)
        if padding_mask is not None:
            with torch.no_grad():
                r = torch.remainder(lengths - 1, k) + 1
                # for each in B, multiply k / r at newlen - 1
                index = (lengths - r).div(k).long()
                index = index.view(1, B, 1).expand(-1, -1, C)
                scale = (k / r).masked_fill(lengths == T, 1)
                scale = scale.type_as(x).view(1, B, 1).expand(-1, -1, C)
                x.scatter_(0, index, scale, reduce="multiply")
        return x, padding_mask


class TransducerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, output_projection=None):
        super().__init__(args, dictionary, embed_tokens, True, output_projection)
        self.downsample_op = AvgPool1dTBCPad(
            kernel_size=args.downsample,
            stride=args.downsample,
            ceil_mode=True,
        ) if args.downsample > 1 else None

        self.joiner = SimpleJoiner(args, self.output_projection)
        self.padding_idx = dictionary.pad()

        @torch.no_grad()
        def scale(m):
            m.weight.mul_((3 * 2 * args.decoder_layers) ** -0.25)

        scale(self.embed_tokens)
        for layer in self.layers:
            scale(layer.self_attn.v_proj)
            scale(layer.self_attn.out_proj)
            scale(layer.fc1)
            scale(layer.fc2)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        **unused
    ):
        src_feats = encoder_out["encoder_out"][0]  # T B C
        padding_mask = encoder_out["encoder_padding_mask"][0]  # B, T
        if self.downsample_op is not None:
            src_feats, padding_mask = self.downsample_op(src_feats, padding_mask)

        # size
        S, B, C = src_feats.size()
        V = len(self.dictionary)

        bos = self.dictionary.bos()
        eos = self.dictionary.eos()
        pad = self.dictionary.pad()
        # prepend bos
        prev_output_tokens[:, 0] = bos
        if incremental_state is not None:
            input_buffer = self._get_input_buffer(incremental_state)
            prev_emit = input_buffer.get("prev_emit", None)
        else:
            # since prev_output_tokens is just targets with eos moved to front,
            # we replace the eos at front, and add back the eos at tail
            prev_output_tokens = torch.cat(
                [prev_output_tokens, prev_output_tokens.new_full((B, 1), pad)],
                dim=1
            )
            target_length = prev_output_tokens.ne(pad).sum(1, keepdim=True)
            prev_output_tokens.scatter_(1, target_length, eos)

        tgt_feats, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=None,
            incremental_state=incremental_state
        )
        logits = self.joiner(src_feats, tgt_feats)

        if incremental_state is not None:
            # (B, S, 1, V) since T == 1
            logits = logits.squeeze(2)
            assert tuple(logits.size()) == (B, S, V), f"{logits.size()} != {(B, S, V)}"

            # force emit at source eos (no blanks)
            if padding_mask is not None:
                source_eos = (~padding_mask).long().sum(-1) - 1
            else:
                source_eos = prev_output_tokens.new_full((B,), S - 1).long()
            # B, S
            logits[..., bos].scatter_(
                1,
                source_eos.view(B, 1),
                -1e4
            )
            if prev_emit is not None:
                # mask past (make them all blank)
                mask = (
                    torch
                    .arange(S, device=logits.device)
                    .view(1, S)
                ) < prev_emit.view(B, 1)
                logits[mask] = F.one_hot(torch.LongTensor([bos]), V).type_as(logits)

            # B, S
            preds = logits.argmax(-1)
            # find first non-blank prediction
            new_emit = (
                preds
                .ne(bos)
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

        extra["padding_mask"] = padding_mask
        return logits, extra

    def rollback(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        steps: Optional[int] = 0,
    ):
        """
        Clear incremental states in the transformer layers.
        The cache is generated because of a forward pass of decode but no prediction.
        """
        def prune_incremental_states(layer, prune):
            if hasattr(layer, "self_attn"):
                input_buffer = layer.self_attn._get_input_buffer(incremental_state)
                for key in ["prev_key", "prev_value"]:
                    input_buffer_key = input_buffer[key]
                    assert input_buffer_key is not None
                    if input_buffer_key.size(2) > prune:
                        input_buffer[key] = input_buffer_key[:, :, :-prune, :]
                    else:
                        typed_empty_dict: Dict[str, Optional[Tensor]] = {}
                        input_buffer = typed_empty_dict
                        break
                assert incremental_state is not None
                layer.self_attn._set_input_buffer(incremental_state, input_buffer)

        for layer in self.layers:
            prune_incremental_states(layer, incremental_state, steps)

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
    "transducer_model", "transducer_model_s"
)
def transducer_model_s(args):
    args.downsample = getattr(args, "downsample", 8)
    # args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    s2t_emformer_s(args)
