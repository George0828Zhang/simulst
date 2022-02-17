import torch
import torch.nn as nn
from fairseq.models.transformer import (
    TransformerDecoder,
    Linear
)
from fairseq.modules import (
    LayerNorm,
    FairseqDropout
)


@torch.no_grad()
def scale_init(decoder: TransformerDecoder):
    assert decoder.args.decoder_layers > 0
    num_attn = 2 if decoder.layers[0].encoder_attn is None else 3
    init_const = (3 * num_attn * decoder.args.decoder_layers) ** -0.25

    def scale(m):
        m.weight.mul_(init_const)

    scale(decoder.embed_tokens)
    for layer in decoder.layers:
        scale(layer.self_attn.v_proj)
        scale(layer.self_attn.out_proj)
        scale(layer.fc1)
        scale(layer.fc2)


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


class InplaceTanh(nn.Module):
    def forward(self, x):
        return x.tanh_()


class EnergyProjection(nn.Module):
    def __init__(self, in_features, hidden_dim, dropout=0, mlp=True, layernorm=False, out_bias_init=-1.0, discretize=False):
        super().__init__()
        if mlp:
            self.fc1 = Linear(in_features, hidden_dim, bias=True)
            self.activation_fn = InplaceTanh()
        else:
            self.fc1 = None
            self.activation_fn = None

        if layernorm:
            self.layer_norm = LayerNorm(hidden_dim)
        else:
            self.layer_norm = None

        self.dropout_module = FairseqDropout(
            float(dropout), module_name=self.__class__.__name__
        )

        self.fc2 = nn.utils.weight_norm(
            Linear(hidden_dim, 1, bias=True), name="weight")
        nn.init.constant_(self.fc2.weight_g, hidden_dim ** -0.5)
        nn.init.constant_(self.fc2.bias, out_bias_init)
        self.discretize = discretize

    def forward(self, x, squeeze=True):
        if self.fc1 is not None:
            x = self.activation_fn(self.fc1(x))
        x = self.dropout_module(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = self.fc2(x)
        if self.training and self.discretize:
            x += torch.randn_like(x)
        return x.squeeze(-1) if squeeze else x
