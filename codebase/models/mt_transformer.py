#!/usr/bin/env python3

from fairseq.models import register_model_architecture
from fairseq.models.transformer import base_architecture


@register_model_architecture("transformer", "transformer_small")
def transformer_small(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.left_pad_source = False
    base_architecture(args)
