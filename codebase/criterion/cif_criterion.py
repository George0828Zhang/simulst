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

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        # ce loss
        loss, nll_loss = self.compute_loss(
            model, net_output, sample, reduce=reduce)
        # ctc loss
        ctc_loss = self.compute_ctc_loss(
            model, net_output, sample)

        nsentences = sample["target"].size(0)
        sample_size = (
            nsentences if self.sentence_avg else sample["ntokens"]
        )
        # quant loss
        alpha_sum = net_output[1]["alpha_sum"][0].float()
        target_lengths = sample["target_lengths"].type_as(alpha_sum)
        l_quant = clipped_l2_loss(
            alpha_sum, target_lengths, clip=self.quant_clip)
        # quant acc
        quant_acc = (
            ((alpha_sum - target_lengths).abs() / target_lengths) <= 0.1
        ).long().sum()
        # latency loss
        l_latency, latency_factor = self.compute_latency_loss(
            model, net_output, sample)
        # combine
        loss = loss + l_quant * self.quant_factor + l_latency * latency_factor + self.ctc_factor * ctc_loss

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": nsentences,
            "sample_size": sample_size,

            "ctc_loss": ctc_loss.data,
            "quantity": l_quant.data,
            "q_acc": quant_acc.data,
            "dal": l_latency.data,
            "dal_factor": latency_factor
        }
        return loss, sample_size, logging_output

    def compute_ctc_loss(self, model, net_output, sample):
        """
        lprobs is expected to be batch first. (from model forward output, or net_output)
        """
        target = sample["target"]
        logits = net_output[1]["ctc_logits"][0]
        lprobs = model.get_normalized_probs(
            (logits, None), log_probs=True
        )
        bsz = target.size(0)
        # reshape lprobs to (L,B,X) for torch.ctc
        if lprobs.size(0) != bsz:
            raise RuntimeError(
                f'batch size error: lprobs shape={lprobs.size()}, bsz={bsz}')
        max_src = lprobs.size(1)
        lprobs = lprobs.transpose(1, 0).contiguous()

        # get subsampling padding mask & lengths
        if net_output[1]["padding_mask"][0] is not None:
            non_padding_mask = ~net_output[1]["padding_mask"][0]
            input_lengths = non_padding_mask.long().sum(-1)
        else:
            input_lengths = lprobs.new_ones(
                (bsz, max_src), dtype=torch.long).sum(-1)

        pad_mask = (target != self.pad_idx) & (
            target != self.eos_idx
        )
        targets_flat = target.masked_select(pad_mask)
        target_lengths = pad_mask.long().sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        return ctc_loss

    def compute_latency_loss(self, model, net_output, sample):
        delays = net_output[1]["delays"][0].float()
        input_lengths = sample["net_input"]["src_lengths"].type_as(delays)
        target_lengths = sample["target_lengths"].type_as(delays)
        target_padding_mask = sample["target"].eq(self.pad_idx)

        # get subsampling padding mask & lengths
        if net_output[1]["padding_mask"][0] is not None:
            non_padding_mask = ~net_output[1]["padding_mask"][0]
            encoder_lengths = non_padding_mask.type_as(delays).sum(-1)
        else:
            raise RuntimeError("expected padding mask in net_output[1] to determine the encoder out lengths.")

        DAL = DifferentiableAverageLagging(
            delays,
            encoder_lengths,
            target_lengths,
            target_padding_mask=target_padding_mask
        )
        # renormalize delays to ms
        DAL *= (input_lengths / encoder_lengths * self.ms_per_frame_shift)

        latency_factor = self.latency_factor
        # if hasattr(model, "latency_control"):
        #     latency_factor = model.latency_control(DAL.mean())
        return DAL.sum(), latency_factor

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
        dal = sum_logs("dal")

        metrics.log_scalar(
            "quantity", quantity / nsentences, nsentences, round=3
        )
        metrics.log_scalar(
            "q_acc", q_acc / nsentences, nsentences, round=3
        )

        metrics.log_scalar(
            "dal", dal / nsentences, nsentences, round=3
        )
        dal_factor = sum_logs("dal_factor")
        metrics.log_scalar(
            "w_dal", dal_factor, 1, round=3
        )

        ctc_loss = sum_logs("ctc_loss")
        sample_size = sum_logs("sample_size")
        metrics.log_scalar(
            "ctc_loss", ctc_loss / sample_size, sample_size, round=3
        )
