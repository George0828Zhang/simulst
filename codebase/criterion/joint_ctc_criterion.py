# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from typing import Optional
from fairseq import metrics
from fairseq.logging.meters import safe_round
from fairseq.criterions import (
    register_criterion,
)
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterionConfig,
    LabelSmoothedCrossEntropyCriterion
)
import logging

logger = logging.getLogger(__name__)


def calc_recall_precision(predict, target, blank_idx=0, pad_idx=1, eps=1e-8):
    N, S = predict.size()
    N, T = target.size()

    uniq, inverse = torch.unique(
        torch.cat((predict, target), dim=1),
        return_inverse=True,
    )
    src = target.new_ones(1)

    def collect(tokens):
        return tokens.new_zeros(
            (N, uniq.size(-1))
        ).scatter_add_(1, tokens, src.expand_as(tokens))

    pred_words = collect(inverse[:, :S])
    target_words = collect(inverse[:, S:])

    # 1. target does not have blank. 2. predict does not have pad
    # therefore, we only need to adjust the denominator

    match = torch.min(target_words, pred_words).sum(-1)
    recall = match / (target.ne(pad_idx).sum(-1) + eps)
    precision = match / (predict.ne(blank_idx).sum(-1) + eps)
    return recall.sum(), precision.sum()


@dataclass
class JointCTCCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    zero_infinity: Optional[bool] = field(
        default=True,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    ctc_factor: Optional[float] = field(
        default=1.0,
        metadata={"help": "factor for ctc loss."},
    )


@register_criterion(
    "joint_ctc_criterion", dataclass=JointCTCCriterionConfig
)
class JointCTCCriterion(LabelSmoothedCrossEntropyCriterion):
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

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(
            model, net_output, sample, reduce=reduce)

        ctc_logits = net_output[1]["ctc_logits"][0]
        ctc_loss, _ = self.compute_ctc_loss(
            model, (ctc_logits, *net_output[1:]), sample["target"], reduce=reduce)
        loss += ctc_loss * self.ctc_factor

        if self.report_accuracy:
            with torch.no_grad():
                y_pred = ctc_logits.argmax(-1)
                recall, precision = calc_recall_precision(
                    y_pred,
                    sample["target"],
                    blank_idx=self.blank_idx,
                    pad_idx=self.pad_idx
                )
                blank_rate = y_pred.eq(self.blank_idx).float().mean(-1).sum()
        else:
            recall = 0
            precision = 0
            blank_rate = 0

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ctc_loss": ctc_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,

            "recall": recall,
            "precision": precision,
            "blank_rate": blank_rate,
        }
        return loss, sample_size, logging_output

    def compute_ctc_loss(self, model, net_output, target, reduce=True):
        """
        lprobs is expected to be batch first. (from model forward output, or net_output)
        """
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        )
        bsz = target.size(0)
        # reshape lprobs to (L,B,X) for torch.ctc
        if lprobs.size(0) != bsz:
            raise RuntimeError(
                f'batch size error: lprobs shape={lprobs.size()}, bsz={bsz}')
        max_src = lprobs.size(1)
        lprobs = lprobs.transpose(1, 0).contiguous()

        # get subsampling padding mask & lengths
        encoder_mask = net_output[1]["encoder_padding_mask"][0]
        if encoder_mask is not None:
            input_lengths = (~encoder_mask).long().sum(-1)
        else:
            input_lengths = lprobs.new_ones(
                (bsz, max_src), dtype=torch.long).sum(-1)

        pad_mask = (target != self.pad_idx) & (
            target != self.eos_idx
        )
        targets_flat = target.masked_select(pad_mask)
        target_lengths = pad_mask.long().sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            nll_loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        # label smoothing
        smooth_loss = -lprobs.sum(dim=-1).transpose(1, 0)  # (L,B) -> (B,L)
        if encoder_mask is not None:
            smooth_loss.masked_fill_(encoder_mask, 0.0)
        eps_i = self.eps / lprobs.size(-1)
        loss = (1.0 - self.eps) * nll_loss + eps_i * smooth_loss.sum()

        return loss, nll_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        super().reduce_metrics(logging_outputs)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        nsentences = sum_logs("nsentences")
        recall = sum_logs("recall")
        precision = sum_logs("precision")
        blank_rate = sum_logs("blank_rate")
        ctc_loss = sum_logs("ctc_loss")
        sample_size = sum_logs("sample_size")

        metrics.log_scalar(
            "blank_rate", blank_rate / nsentences, nsentences, round=3
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss / sample_size, sample_size, round=3
        )

        metrics.log_scalar("_recall", recall / nsentences, nsentences)
        metrics.log_scalar("_precision", precision / nsentences, nsentences)

        def calc_f1(meters):
            """ this is approx. """
            r = meters["_recall"].avg
            p = meters["_precision"].avg
            if r + p > 0:
                return safe_round(2 * p * r / (p + r), 3)
            else:
                return 0

        metrics.log_derived(
            "f1",
            calc_f1,
        )
