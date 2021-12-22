import logging
import torch.nn as nn

from fairseq.models import (
    register_model,
    register_model_architecture
)

# user
from simultaneous_translation.models.seq2seq import (
    S2TSeq2SeqModel,
    s2t_seq2seq_s
)
from simultaneous_translation.models.nat_utils import (
    generate,
)
from simultaneous_translation.models.sinkhorn_encoder import (
    OutProjection
)

logger = logging.getLogger(__name__)


@register_model("s2t_nat")
class S2TNATModel(S2TSeq2SeqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.one_pass_decoding = True  # must implement generate()

    @classmethod
    def build_encoder(cls, args, task, embed_tokens, ctc_projection):
        return super().build_encoder(args, task, embed_tokens, ctc_projection, non_causal_text=True)

    @classmethod
    def build_decoder(cls, args, task, decoder_embed_tokens):
        out_projection = nn.Linear(
            decoder_embed_tokens.weight.shape[1],
            decoder_embed_tokens.weight.shape[0],
            bias=False,
        )
        nn.init.normal_(
            out_projection.weight, mean=0, std=args.decoder_embed_dim ** -0.5
        )
        decoder = OutProjection(task.target_dictionary, out_projection)
        return decoder

    def generate(self, src_tokens, src_lengths, blank_idx=0, **unused):
        return generate(self, src_tokens, src_lengths, blank_idx=blank_idx, blank_penalty=2.15)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        src_txt_tokens=None,  # unused
        src_txt_lengths=None,  # unused
    ):
        logits, decoder_out = super().forward(
            src_tokens,
            src_lengths,
            prev_output_tokens,
            src_txt_tokens=src_txt_tokens,
            src_txt_lengths=src_txt_lengths
        )
        encoder_out = decoder_out["encoder_out"]
        decoder_out.update({
            "padding_mask": encoder_out["encoder_padding_mask"][0],
        })
        return logits, decoder_out


@register_model_architecture(
    "s2t_nat", "s2t_nat_s"
)
def s2t_nat_s(args):
    s2t_seq2seq_s(args)
