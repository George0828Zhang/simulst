# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from typing import Optional
from fairseq import metrics
# from fairseq.logging.meters import safe_round
from fairseq.criterions import (
    register_criterion,
)
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterionConfig,
    LabelSmoothedCrossEntropyCriterion
)
import logging
from codebase.criterion.monotonic import ssnt_loss, ssnt_loss_mem

logger = logging.getLogger(__name__)


@dataclass
class SSNTCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    zero_infinity: Optional[bool] = field(
        default=True,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    memory_efficient: Optional[bool] = field(
        default=False,
        metadata={"help": "concatenate the 'targets' dimension to improve efficiency"},
    )


@register_criterion(
    "ssnt_criterion", dataclass=SSNTCriterionConfig
)
class SSNTCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, cfg, task):
        super().__init__(
            task,
            cfg.sentence_avg,
            cfg.label_smoothing,
            ignore_prefix_size=cfg.ignore_prefix_size,
            report_accuracy=cfg.report_accuracy
        )

        # self.blank_idx = task.target_dictionary.index(task.blank_symbol) if hasattr(task, 'blank_symbol') else 0
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        # self.zero_infinity = cfg.zero_infinity
        self.memory_efficient = cfg.memory_efficient

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_ssnt_loss(
            model, net_output, sample)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        # sample_size = 1
        # if not self.sentence_avg:
        #     factor = sample["ntokens"] / sample["target"].size(0)
        #     loss /= factor

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,

            "npads": sample["target"].numel() - sample["ntokens"]
        }
        return loss, sample_size, logging_output

    def compute_ssnt_loss(self, model, net_output, sample, reduce=True):
        """
        B x T x S
        """
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        )
        targets = sample["target"]

        log_emit = F.logsigmoid(
            net_output[1]["log_emit"]
        )

        bsz = targets.size(0)
        if self.memory_efficient:
            target_masks = targets.ne(self.pad_idx)
            target_lengths = target_masks.long().sum(-1)
            targets = targets[target_masks]
            _bsz = target_lengths.sum().item()
            if lprobs.size(0) != _bsz:
                raise RuntimeError(
                    f'batch size error: lprobs shape={lprobs.size()}, bsz={_bsz}')
            if log_emit.size(0) != _bsz:
                raise RuntimeError(
                    f'batch size error: log_emit shape={log_emit.size()}, bsz={_bsz}')
        else:
            target_lengths = sample["target_lengths"]
            if lprobs.size(0) != bsz:
                raise RuntimeError(
                    f'batch size error: lprobs shape={lprobs.size()}, bsz={bsz}')
            if log_emit.size(0) != bsz:
                raise RuntimeError(
                    f'batch size error: log_emit shape={log_emit.size()}, bsz={bsz}')

        max_src = lprobs.size(-2)
        lprobs = lprobs.contiguous()
        log_emit = log_emit.contiguous()

        # get subsampling padding mask & lengths
        if net_output[1]["padding_mask"] is not None:
            non_padding_mask = ~net_output[1]["padding_mask"]
            input_lengths = non_padding_mask.long().sum(-1)
        else:
            input_lengths = lprobs.new_ones(
                (bsz, max_src), dtype=torch.long).sum(-1)

        if self.memory_efficient:
            loss = ssnt_loss_mem(
                lprobs,
                targets,
                log_emit,
                input_lengths,
                target_lengths,
                reduction="sum"
            )
        else:
            loss = ssnt_loss(
                lprobs,
                targets,
                log_emit,
                input_lengths,
                target_lengths,
                reduction="sum"
            )

        return loss, loss.detach()

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

        ntokens = sum_logs("ntokens")
        npads = sum_logs("npads")
        ntotal = ntokens + npads
        metrics.log_scalar(
            "pad_rate", npads / ntotal, ntotal, round=3
        )

        # nsentences = sum_logs("nsentences")
        # recall = sum_logs("recall")
        # precision = sum_logs("precision")
        # blank_rate = sum_logs("blank_rate")

        # metrics.log_scalar(
        #     "blank_rate", blank_rate / nsentences, nsentences, round=3
        # )

        # metrics.log_scalar("_recall", recall / nsentences, nsentences)
        # metrics.log_scalar("_precision", precision / nsentences, nsentences)

        # def calc_f1(meters):
        #     """ this is approx. """
        #     r = meters["_recall"].avg
        #     p = meters["_precision"].avg
        #     if r + p > 0:
        #         return safe_round(2 * p * r / (p + r), 3)
        #     else:
        #         return 0

        # metrics.log_derived(
        #     "f1",
        #     calc_f1,
        # )
