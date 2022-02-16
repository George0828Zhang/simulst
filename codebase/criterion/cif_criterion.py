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
import logging

logger = logging.getLogger(__name__)


@dataclass
class CIFCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    quant_factor: Optional[float] = field(
        default=1.0,
        metadata={"help": "factor for quantity loss."},
    )
    quant_clip: Optional[float] = field(
        default=10.0,
        metadata={"help": "for each example in batch, clip the max value for quant loss."},
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

        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.quant_factor = cfg.quant_factor
        self.quant_clip = cfg.quant_clip

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(
            model, net_output, sample, reduce=reduce)

        nsentences = sample["target"].size(0)
        sample_size = (
            nsentences if self.sentence_avg else sample["ntokens"]
        )
        alpha_sum = net_output[1]["alpha_sum"][0].float()
        target_lengths = sample["target_lengths"].type_as(alpha_sum)
        l_quant = clipped_l2_loss(
            alpha_sum, target_lengths, clip=self.quant_clip)
        loss = loss + l_quant * self.quant_factor
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": nsentences,
            "sample_size": sample_size,

            "quantity": l_quant.data,
        }
        return loss, sample_size, logging_output

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
        nsentences = sum_logs("nsentences")

        metrics.log_scalar(
            "quantity", quantity / nsentences, nsentences, round=3
        )
