# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from typing import Optional
from fairseq import metrics
from fairseq.criterions import (
    register_criterion,
)
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterionConfig,
    LabelSmoothedCrossEntropyCriterion
)
from simuleval.metrics.latency import DifferentiableAverageLagging
from codebase.criterion.best_alignment import best_alignment
import logging

logger = logging.getLogger(__name__)


@dataclass
class CIFCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    zero_infinity: Optional[bool] = field(
        default=True,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    ctc_factor: Optional[float] = field(
        default=0.3,
        metadata={"help": "factor for ctc loss."},
    )
    quant_factor: Optional[float] = field(
        default=1.0,
        metadata={"help": "factor for quantity loss."},
    )
    quant_type: Optional[str] = field(
        default="align",
        metadata={"help": "type of quantity loss. sum or align"},
    )
    quant_clip: Optional[float] = field(
        default=10.0,
        metadata={"help": "for each example in batch, clip the max value for quant loss."},
    )
    latency_factor: Optional[float] = field(
        default=0.0,
        metadata={"help": "factor for latency loss."},
    )
    ms_per_frame_shift: Optional[float] = field(
        default=10,
        metadata={"help": "The milliseconds per frame shift used to compute the delay."},
    )


def clipped_l2_loss(x, y, reduce=True, clip=None):
    y = y.type_as(x)
    if clip is not None:
        clip = clip ** 0.5
        with torch.no_grad():
            clipped_y = y.clip(min=x - clip, max=x + clip)
    else:
        clipped_y = y
    l_quant = F.mse_loss(x, clipped_y, reduction='none')
    return l_quant.sum() if reduce else l_quant


@register_criterion(
    "cif_loss", dataclass=CIFCriterionConfig
)
class CIFCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, cfg, task):
        super().__init__(
            task,
            cfg.sentence_avg,
            cfg.label_smoothing,
            ignore_prefix_size=cfg.ignore_prefix_size,
            report_accuracy=cfg.report_accuracy
        )

        self.blank_idx = task.target_dictionary.index(task.blank_symbol) if hasattr(task, 'blank_symbol') else 0
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.zero_infinity = cfg.zero_infinity
        self.ctc_factor = cfg.ctc_factor
        self.quant_factor = cfg.quant_factor
        self.quant_clip = cfg.quant_clip
        self.latency_factor = cfg.latency_factor
        self.ms_per_frame_shift = cfg.ms_per_frame_shift
        self.quant_type = cfg.quant_type

    def prepare_tensors(self, model, net_output, sample):
        # ctc loss
        target = sample["target"]
        logits = net_output[1]["ctc_logits"]  # list, may be empty
        alpha = net_output[1]["alpha"][0].float()
        delays = net_output[1]["delays"][0].float()
        encoder_padding_mask = net_output[1]["encoder_padding_mask"][0]

        assert len(logits) > 0 or self.ctc_factor == 0
        if len(logits) > 0:
            lprobs = model.get_normalized_probs(
                (logits[0], None), log_probs=True
            )
            bsz = target.size(0)
            # lprobs is expected to be batch first.
            if lprobs.size(0) != bsz:
                raise RuntimeError(
                    f'batch size error: lprobs shape={lprobs.size()}, bsz={bsz}')
            max_src = lprobs.size(1)
            # reshape lprobs to (L,B,X) for torch.ctc
            lprobs = lprobs.transpose(1, 0).contiguous()
        else:
            lprobs = torch.empty(0).type_as(alpha)

        # get encoder subsampling mask & lengths
        if encoder_padding_mask is not None:
            encoder_lengths = (~encoder_padding_mask).long().sum(-1)
        else:
            encoder_lengths = alpha.new_ones(
                (bsz, max_src), dtype=torch.long).sum(-1)

        # get target mask
        target_padding_mask = (target == self.pad_idx)

        return {
            "ctc_lprobs": lprobs,
            "alpha": alpha,
            "delays": delays,
            "encoder_padding_mask": encoder_padding_mask,
            "encoder_lengths": encoder_lengths,
            "target_padding_mask": target_padding_mask
        }

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        # ce loss
        loss, nll_loss = self.compute_loss(
            model, net_output, sample, reduce=reduce)

        tensors = self.prepare_tensors(model, net_output, sample)

        # ctc loss
        ctc_loss = self.compute_ctc_loss(tensors, sample)

        # quant loss
        l_quant, quant_acc = self.compute_quantity_loss(
            tensors, sample, model.encoder.cif_layer.beta)

        # latency loss
        l_latency, latency = self.compute_latency_loss(tensors, sample)

        # combine
        loss = loss + l_quant * self.quant_factor + l_latency * self.latency_factor + self.ctc_factor * ctc_loss

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,

            "ctc_loss": ctc_loss.data,
            "quantity": l_quant.data,
            "q_acc": quant_acc.data,
            "latency": latency.data,
        }
        return loss, sample_size, logging_output

    def compute_ctc_loss(self, tensors, sample):
        if self.ctc_factor == 0:
            return torch.zeros(1).type_as(tensors["alpha"])
        lprobs = tensors["ctc_lprobs"]
        encoder_lengths = tensors["encoder_lengths"]
        target_padding_mask = tensors["target_padding_mask"]

        target = sample["target"]
        target_lengths = sample["target_lengths"]
        targets_flat = target.masked_select(~target_padding_mask)

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = F.ctc_loss(
                lprobs,
                targets_flat,
                encoder_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        return ctc_loss

    def compute_latency_loss(self, tensors, sample):
        input_lengths = sample["net_input"]["src_lengths"]
        target_lengths = sample["target_lengths"]
        delays = tensors["delays"]
        encoder_lengths = tensors["encoder_lengths"]
        target_padding_mask = tensors["target_padding_mask"]

        expected_latency = DifferentiableAverageLagging(
            delays,
            encoder_lengths,
            target_lengths,
            target_padding_mask=target_padding_mask
        )
        latency_loss = expected_latency.clip(min=0).sum()
        # renormalize delays to ms
        expected_latency = expected_latency * (input_lengths / encoder_lengths * self.ms_per_frame_shift)
        return latency_loss, expected_latency.sum()

    def compute_quantity_loss(self, tensors, sample, beta=1.0):
        # alpha_sum = tensors["alpha_sum"]
        alpha = tensors["alpha"]
        ctc_lprobs = tensors["ctc_lprobs"]
        encoder_lengths = tensors["encoder_lengths"]
        encoder_padding_mask = tensors["encoder_padding_mask"]
        target = sample["target"]
        target_lengths = sample["target_lengths"]

        if self.quant_type == "sum":
            quant_targets = target_lengths.unsqueeze(1)
            boundary = torch.ones_like(quant_targets)
            quant_outputs = alpha.sum(1, keepdim=True) / beta
        elif self.quant_type == "align":
            with torch.no_grad():
                bsz, tgt_len = target.size()
                src_len = ctc_lprobs.size(0)
                assert tuple(alpha.shape) == (bsz, src_len), f"size mismatch {alpha.shape} != {(bsz, src_len)}"

                # we treat blanks (s % 2 == 0) as next segment.
                states = best_alignment(
                    ctc_lprobs,
                    target,
                    encoder_lengths,
                    target_lengths,
                    blank=self.blank_idx,
                )
                # (B, S) tensor indicating each source position's target segment
                # we treat all but the last blanks (s % 2 == 0) as next segment
                seg_ids = states.div(2, rounding_mode='floor')
                seg_ids_next = seg_ids.roll(-1, dims=1)

                # (B, S) boundary is not blank and diff from next
                boundary = (seg_ids != seg_ids_next) & (states % 2 != 0)

                if encoder_padding_mask is not None:
                    boundary[encoder_padding_mask] = 0

                assert tuple(boundary.shape) == (bsz, src_len), f"size mismatch {boundary.shape} != {(bsz, src_len)}"
                # NOTE: the below assert will break if impossible to align (e.g. src < tgt or many dups in tgt)
                # we will ignore the unaligned part for now.
                # assert (boundary.sum(1) == target_lengths).all(), (
                #     f"""boundary corrupt: {boundary.cpu()}, expected to sum to
                #     {target_lengths.cpu()} at dim 1, got {boundary.sum(1).cpu()}""")
                quant_targets = boundary.cumsum(1)
            # assume alpha is already masked appropriately.
            quant_outputs = alpha.cumsum(1) / beta
        else:
            raise NotImplementedError

        l_quant = clipped_l2_loss(
            quant_outputs[boundary],
            quant_targets[boundary],
            reduce=False,
            clip=self.quant_clip
        )
        # since we use cumsum, we should mean across time.
        norm = boundary / boundary.sum(1, keepdim=True)
        l_quant = (l_quant * norm[boundary]).sum()

        # quant acc
        quant_acc = (
            ((quant_outputs[:, -1] - target_lengths).abs() / target_lengths) <= 0.1
        ).long().sum()

        return l_quant, quant_acc

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        super().reduce_metrics(logging_outputs)

        def sum_logs(key):
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        quantity = sum_logs("quantity")
        q_acc = sum_logs("q_acc")
        nsentences = sum_logs("nsentences")
        latency = sum_logs("latency")

        metrics.log_scalar(
            "quantity", quantity / nsentences, nsentences, round=3
        )
        metrics.log_scalar(
            "q_acc", q_acc / nsentences, nsentences, round=3
        )

        metrics.log_scalar(
            "latency", latency / nsentences, nsentences, round=3
        )

        ctc_loss = sum_logs("ctc_loss")
        sample_size = sum_logs("sample_size")
        metrics.log_scalar(
            "ctc_loss", ctc_loss / sample_size, sample_size, round=3
        )
