import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, List
from torch import Tensor
from pathlib import Path
from fairseq import checkpoint_utils

from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairseq.models.transformer import (
    TransformerDecoder,
    Linear
)
from fairseq.incremental_decoding_utils import with_incremental_state

from codebase.models.torch_cif import cif_function
from codebase.models.s2t_emformer import (
    S2TEmformerEncoder,
    S2TEmformerModel,
    s2t_emformer_s
)

logger = logging.getLogger(__name__)


@register_model("cif_transformer")
class CIFTransformerModel(S2TEmformerModel):
    @staticmethod
    def add_args(parser):
        super(CIFTransformerModel,
              CIFTransformerModel).add_args(parser)
        parser.add_argument(
            "--nar-decoder",
            action="store_true",
            help="train non-autoregressive decoder."
        )

    @classmethod
    def build_encoder(cls, args):
        encoder = CIFEncoder(args)
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

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        return CIFDecoder(args, task.tgt_dict, embed_tokens)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(
            src_tokens,
            src_lengths,
            prev_output_tokens.ne(self.decoder.padding_idx).sum(1),
        )
        logits, extra = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        extra["alpha_sum"] = encoder_out["alpha_sum"]
        return logits, extra


@with_incremental_state
class CIFLayer(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_dim,
        beta,
        max_len,
    ):
        super().__init__()

        # emission prediction
        emit_fc1 = Linear(in_features, hidden_dim, bias=True)
        emit_act_fn = nn.Tanh()
        emit_fc2 = nn.utils.weight_norm(
            Linear(hidden_dim, 1, bias=True), name="weight")
        nn.init.constant_(emit_fc2.weight_g, hidden_dim ** -0.5)
        nn.init.constant_(emit_fc2.bias, -1.)
        self.emit_projection = nn.Sequential(
            emit_fc1,
            emit_act_fn,
            emit_fc2
        )
        self.beta = beta
        self.max_len = max_len

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor] = None,
        target_lengths: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        r"""
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if incremental_state is not None:
            raise NotImplementedError("This functionality is for streaming. You be implemented soon.")

        x = x.transpose(1, 0)

        # calculate integration weights
        alpha = self.emit_projection(x).sigmoid_().squeeze(-1)

        x, feat_lengths, alpha_sum = cif_function(
            x,
            alpha,
            beta=self.beta,
            padding_mask=encoder_padding_mask,
            target_lengths=target_lengths,
            max_output_length=self.max_len
        )

        # project back and (B, T, C-1) -> (T, B, C-1)
        x = x.transpose(0, 1)

        return x, feat_lengths, alpha_sum


class CIFEncoder(S2TEmformerEncoder):
    def __init__(self, args):
        super().__init__(args)
        self.cif_layer = CIFLayer(
            in_features=args.encoder_embed_dim,
            hidden_dim=args.encoder_embed_dim,
            beta=1.0,
            max_len=args.max_target_positions - 1
        )

    def forward(self, src_tokens, src_lengths, target_lengths=None, incremental_state=None):
        x = super().forward(src_tokens, src_lengths)

        cif_out, cif_lengths, alpha_sum = self.cif_layer(
            x["encoder_out"][0],
            x["encoder_padding_mask"][0] if len(x["encoder_padding_mask"]) > 0 else None,
            target_lengths=target_lengths,
            incremental_state=incremental_state,
        )
        x["cif_out"] = [cif_out]
        x["cif_lengths"] = [cif_lengths]
        x["alpha_sum"] = [alpha_sum]
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = super().reorder_encoder_out(encoder_out, new_order)
        new_encoder_out["cif_out"] = (
            []
            if len(encoder_out["cif_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["cif_out"]]
        )
        new_encoder_out["cif_lengths"] = (
            []
            if len(encoder_out["cif_lengths"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["cif_lengths"]]
        )
        return new_encoder_out

    def load_state_dict(self, state_dict, strict=True):
        """
        1. ignores cif projection if not available
        """
        ignores = [
            "cif_layer.emit_projection.0.weight",
            "cif_layer.emit_projection.0.bias",
            "cif_layer.emit_projection.2.bias",
            "cif_layer.emit_projection.2.weight_g",
            "cif_layer.emit_projection.2.weight_v"
        ]
        cur_state_dict = self.state_dict()

        for w in ignores:
            if w not in state_dict:
                logger.warning("Ignoring CIF projection weights! Make sure this is intended...")
                state_dict[w] = cur_state_dict[w]

        return super().load_state_dict(state_dict, strict=strict)


class CIFDecoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        output_projection=None,
    ):
        super().__init__(
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=True,
            output_projection=output_projection
        )
        self.is_nar = args.nar_decoder

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        cif: Optional[Tensor] = None
        cif_lengths: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        eos_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
            cif = encoder_out["cif_out"][0]
            assert (
                cif.size()[1] == bs
            ), f"Expected cif.shape == (t, {bs}, c) got {cif.shape}"
            cif = cif.transpose(1, 0)
            cif_lengths = encoder_out["cif_lengths"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            _T = prev_output_tokens.size(1)
            eos_mask = cif_lengths < _T
            cif_index = cif_lengths.clip(max=_T) - 1
            cif = cif.gather(
                1,
                cif_index.view(bs, 1, 1).expand(-1, -1, cif.size(-1))
            )
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = cif
        if not self.is_nar:
            x += self.embed_tokens(prev_output_tokens)

        x *= self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment and not self.is_nar:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if eos_mask is not None:
            # x: (B, 1, C)
            # eos_mask: (B, )
            x[eos_mask, -1, self.dictionary.eos()] = 1e4

        return x, {"attn": [attn], "inner_states": inner_states}


@register_model_architecture("cif_transformer", "cif_transformer_s")
def cif_transformer(args):
    args.nar_decoder = getattr(args, "nar_decoder", False)
    s2t_emformer_s(args)
