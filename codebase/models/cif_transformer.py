import re
import math
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
from fairseq.modules import (
    LayerNorm,
    TransformerDecoderLayer
    # FairseqDropout
)
from fairseq.incremental_decoding_utils import with_incremental_state

from codebase.models.torch_cif import cif_function
from codebase.models.s2t_emformer import (
    S2TEmformerEncoder,
    S2TEmformerModel,
    s2t_emformer_s
)
from codebase.modules.causal_conv import CausalConvTBC

logger = logging.getLogger(__name__)


@register_model("cif_transformer")
class CIFTransformerModel(S2TEmformerModel):
    @staticmethod
    def add_args(parser):
        super(CIFTransformerModel,
              CIFTransformerModel).add_args(parser)
        parser.add_argument(
            "--cif-beta",
            type=float,
            help="Cif firing threshold."
        )
        parser.add_argument(
            "--cif-sg-alpha",
            action="store_true",
            help="stop gradient for alpha prediction."
        )
        parser.add_argument(
            "--cif-conv-kernel",
            type=int,
            help="Conv1d kernel for alpha prediction."
        )
        parser.add_argument(
            "--cif-highway",
            action="store_true",
            help="add highway connection from cif to output layer."
        )

    @classmethod
    def build_encoder(cls, args, task):
        encoder = CIFEncoder(args, task.target_dictionary)
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
        extra.update(encoder_out)
        return logits, extra

    def load_state_dict(self, state_dict, strict=True, model_cfg=None):
        """ legacy models ctc_layer was on decoder. """
        for w in state_dict.keys():
            if re.search(r"decoder.ctc_layer\..*", w) is not None:
                new_w = w.replace("decoder", "encoder")
                state_dict[new_w] = state_dict[w]
                del state_dict[w]

        return super().load_state_dict(state_dict, strict=strict)


@with_incremental_state
class CIFLayer(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_dim,
        kernel_size,
        dropout,
        sg_alpha,
        beta,
    ):
        super().__init__()

        self.alpha_proj = nn.Sequential(
            CausalConvTBC(in_features, hidden_dim, kernel_size=kernel_size),
            LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout), inplace=True),
            Linear(hidden_dim, 1, bias=True)
        )
        self.sg_alpha = sg_alpha
        self.beta = beta
        self.tail_thres = beta / 2

    def extra_repr(self):
        s = "sg_alpha={}, beta={}, tail_thres={}".format(
            self.sg_alpha,
            self.beta,
            self.tail_thres,
        )
        return s

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor] = None,
        target_lengths: Optional[Tensor] = None,
    ):
        r"""
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # calculate integration weights
        if self.sg_alpha:
            alpha = self.alpha_proj(x.detach())
        else:
            alpha = self.alpha_proj(x)

        x = x.transpose(1, 0)
        alpha = alpha.transpose(1, 0).sigmoid().squeeze(-1)

        # apply masking first
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.unsqueeze(2), 0)
            alpha = alpha.masked_fill(encoder_padding_mask, 0)

        cif_out = cif_function(
            x,
            alpha,
            beta=self.beta,
            tail_thres=self.tail_thres,
            # padding_mask=encoder_padding_mask,
            target_lengths=target_lengths,
        )

        # (B, T, C-1) -> (T, B, C-1)
        cif_feats = cif_out["cif_out"][0].transpose(0, 1)
        cif_out.update({
            "cif_out": [cif_feats],
            "alpha": [alpha]
        })
        return cif_out

    def infer(
        self,
        x,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        finish=False,
    ):
        """ expects incremental input (x is a new chunk)
        x: (chunk_len, batch, embed_dim)
        """
        chunk_len, bsz, C = x.size()
        if bsz > 1:
            raise NotImplementedError("batched infer not supported for now.")

        # calculate integration weights
        alpha = x
        for m in self.alpha_proj:
            if isinstance(m, CausalConvTBC):
                alpha = m(alpha, incremental_state)
            else:
                alpha = m(alpha)

        x = x.transpose(1, 0)
        alpha = alpha.transpose(1, 0).sigmoid().squeeze(-1)
        # alpha_fw = alpha.clone()

        cached_state = self.get_incremental_state(incremental_state, "cif_state")
        if cached_state is None:
            cached_state: Dict[str, Optional[Tensor]] = {}
        if (
            "prev_weight" in cached_state
            and cached_state["prev_weight"].numel() > 0
        ):
            # leftover features with weight
            # we can treat it as a single source feature
            prev_weight = cached_state["prev_weight"]   # (B, 1)
            prev_feat = cached_state["prev_feat"]       # (B, 1, C)
            alpha = torch.cat((prev_weight, alpha), dim=1)
            x = torch.cat((prev_feat, x), dim=1)

        cif_out = cif_function(
            x,
            alpha,
            beta=self.beta,
            tail_thres=self.tail_thres if finish else 0,
        )

        cif_feats = cif_out["cif_out"][0]  # (B, t, C)
        cif_len = cif_out["cif_lengths"][0]  # (B,)
        tail_weight = cif_out["tail_weights"][0]  # (B,)
        # we now assume B = 1
        if not finish:
            prev_feat = cif_feats[:, cif_len.item() - 1:, :]  # (B, 1, C)
            prev_weight = tail_weight.view(bsz, 1)   # (B, 1)
            # feat was normalized to beta in cif_function(), unscale to 1 for next segment
            prev_feat = prev_feat / self.beta
        else:
            # this will trigger error if infer is invoked after finish.
            prev_feat = None
            prev_weight = None

        cached_state["prev_feat"] = prev_feat
        cached_state["prev_weight"] = prev_weight
        self.set_incremental_state(incremental_state, "cif_state", cached_state)

        cif_len = cif_len if finish else (cif_len - 1)
        cif_feats = cif_feats.narrow(
            1, 0, cif_len.item()).transpose(0, 1)  # (B, t-1, C)
        cif_out.update({
            "cif_out": [cif_feats],
            "cif_lengths": [cif_len],
            "alpha": [alpha]
        })
        return cif_out


class CIFEncoder(S2TEmformerEncoder):
    def __init__(self, args, dictionary=None):
        super().__init__(args, dictionary)
        self.cif_layer = CIFLayer(
            in_features=args.encoder_embed_dim,
            hidden_dim=args.encoder_embed_dim,
            kernel_size=args.cif_conv_kernel,
            dropout=args.activation_dropout,
            sg_alpha=args.cif_sg_alpha,
            beta=args.cif_beta,
        )

    def forward(self, src_tokens, src_lengths, target_lengths=None):
        encoder_out = super().forward(src_tokens, src_lengths)
        src_feats = encoder_out["encoder_out"][0]  # T B C
        padding_mask = encoder_out["encoder_padding_mask"][0]  # B, T

        cif_out = self.cif_layer(
            src_feats,
            padding_mask,
            target_lengths=target_lengths,
        )
        encoder_out.update(cif_out)
        return encoder_out

    def infer(self, src_tokens, src_lengths, incremental_state, finish=False):
        encoder_out = super().infer(
            src_tokens,
            src_lengths,
            incremental_state=incremental_state,
            finish=finish
        )
        src_feats = encoder_out["encoder_out"][0]  # T B C
        padding_mask = encoder_out["encoder_padding_mask"][0]  # B, T

        cif_out = self.cif_layer.infer(
            src_feats,
            incremental_state=incremental_state,
            encoder_padding_mask=padding_mask,
            finish=finish,
        )
        encoder_out.update(cif_out)
        return encoder_out

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = super().reorder_encoder_out(encoder_out, new_order)
        new_encoder_out["cif_out"] = (
            []
            if len(encoder_out["cif_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["cif_out"]]
        )
        for key in ("cif_lengths", "alpha", "delays"):
            new_encoder_out[key] = (
                []
                if len(encoder_out[key]) == 0
                else [x.index_select(0, new_order) for x in encoder_out[key]]
            )
        return new_encoder_out

    def load_state_dict(self, state_dict, strict=True):
        """
        1. ignores cif projection if not available
        """
        cur_state_dict = self.state_dict()

        for w in cur_state_dict.keys():
            if re.search(r"cif_layer\..*", w) is not None and w not in state_dict:
                logger.warning("Ignoring CIF projection weights! Make sure this is intended...")
                state_dict[w] = cur_state_dict[w]
            if re.search(r"ctc_layer\..*", w) is not None and w not in state_dict:
                logger.warning("Ignoring CTC projection weights! Make sure this is intended...")
                state_dict[w] = cur_state_dict[w]

        return super().load_state_dict(state_dict, strict=strict)


class FakeCrossAttn(nn.Module):
    def __init__(self, embed_dim, kdim, bias=True):
        super().__init__()
        self.kdim = kdim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.activation_fn = nn.GELU()
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))

        # init
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query, key, value, **kwargs):
        assert query.size() == key.size()
        q = self.q_proj(query)
        k = self.k_proj(key)
        out = self.activation_fn(q + k)
        return self.out_proj(out), None  # followed by dropout


class CIFDecoderLayer(TransformerDecoderLayer):
    def build_encoder_attention(self, embed_dim, args):
        return FakeCrossAttn(
            embed_dim,
            kdim=args.encoder_embed_dim
        )


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
            # no_encoder_attn=True,
            output_projection=output_projection
        )
        self.highway = args.cif_highway

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return CIFDecoderLayer(args, no_encoder_attn)

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

        cif: Optional[Tensor] = None
        cif_lengths: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["cif_out"]) > 0:
            cif = encoder_out["cif_out"][0]
            assert (
                cif.size()[1] == bs
            ), f"Expected cif.shape == (t, {bs}, c) got {cif.shape}"
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
            cif_index = cif_lengths.clip(max=_T) - 1
            cif = cif.gather(
                0,
                cif_index.view(1, bs, 1).expand(-1, -1, cif.size(-1))
            )
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_tokens(prev_output_tokens) * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x, inplace=True)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            x, layer_attn, _ = layer(
                x,
                cif,  # enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # highway connection
        if self.highway:
            x += cif

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        overshoot_weight: float = 1.0,
        **unused
    ):
        x, extra = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            x = self.output_layer(x)

        if incremental_state is not None:
            cif_lengths = encoder_out["cif_lengths"][0]
            overshoot = (prev_output_tokens.size(1) - cif_lengths).clip(min=0)
            # x: (B, 1, C)
            # overshoot: (B, )
            eos = self.dictionary.eos()
            x[:, -1, eos] += overshoot.type_as(x) * overshoot_weight

        return x, extra


@register_model_architecture("cif_transformer", "cif_transformer_s")
def cif_transformer(args):
    args.cif_beta = getattr(args, "cif_beta", 1.0)  # set to smaller value to allow longer predictions
    args.cif_sg_alpha = getattr(args, "cif_sg_alpha", False)
    args.cif_conv_kernel = getattr(args, "cif_conv_kernel", 3)
    args.cif_highway = getattr(args, "cif_highway", False)
    args.ctc_layer = True
    s2t_emformer_s(args)
