# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
import torch
from typing import Optional
from fairseq import metrics
from fairseq.logging.meters import safe_round
from fairseq.criterions import (
    register_criterion,
)
import logging

from simultaneous_translation.criterion.label_smoothed_ctc_criterion import (
    calc_recall_precision,
    LabelSmoothedCTCCriterionConfig,
    LabelSmoothedCTCCriterion
)

logger = logging.getLogger(__name__)


@dataclass
class LabelSmoothedMTLCriterionConfig(LabelSmoothedCTCCriterionConfig):
    asr_factor: Optional[float] = field(
        default=0.5,
        metadata={
            "help":
            "the multitask learning criterion is described by"
            "asr_factor*ctc_asr_loss + (1-asr_factor)*cross_entropy"
        },
    )


@register_criterion(
    "label_smoothed_mtl", dataclass=LabelSmoothedMTLCriterionConfig
)
class LabelSmoothedMTLCriterion(LabelSmoothedCTCCriterion):
    def __init__(self, cfg, task):
        super().__init__(cfg, task)
        self.asr_factor = cfg.asr_factor
        self.blank_idx = task.source_dictionary.bos()
        assert self.pad_idx == task.source_dictionary.pad()
        assert self.eos_idx == task.source_dictionary.eos()

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])

        # ctc for asr
        encoder_out = net_output[1]["encoder_out"]
        padding_mask = encoder_out["speech_padding_mask"][0] \
            if len(encoder_out["speech_padding_mask"]) > 0 else None
        asr_output = (
            encoder_out["encoder_logits"][0],
            {"padding_mask": padding_mask},
        )
        asr_target = sample['net_input']['src_txt_tokens']
        asr_loss, _ = self.compute_ctc_loss(
            model, asr_output, asr_target, reduce=reduce)
        shrink_rate = encoder_out.get(
            "shrink_rate", [sample["target"].size(0)])[0]

        # cross_entropy for translation
        ce_loss, nll_loss = self.compute_loss(
            model, net_output, sample, reduce=reduce)

        # combine
        loss = (1 - self.asr_factor) * ce_loss + self.asr_factor * asr_loss

        if self.report_accuracy:
            with torch.no_grad():
                # asr
                asr_pred = asr_output[0].argmax(-1)
                asr_recall, asr_precision = calc_recall_precision(
                    asr_pred,
                    asr_target,
                    blank_idx=self.blank_idx,
                    pad_idx=self.pad_idx
                )
                blank_rate = asr_pred.eq(self.blank_idx).float().mean(-1).sum()
                # st
                st_pred = net_output[0].argmax(-1)
                recall, precision = calc_recall_precision(
                    st_pred,
                    sample["target"],
                    blank_idx=self.blank_idx,
                    pad_idx=self.pad_idx
                )
        else:
            asr_recall = asr_precision = 0
            recall = precision = 0
            blank_rate = 0

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ce_loss": ce_loss.data,
            "asr_loss": asr_loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,

            "asr_recall": asr_recall,
            "asr_precision": asr_precision,
            "recall": recall,
            "precision": precision,
            "blank_rate": blank_rate,
            "shrink_rate": shrink_rate,
        }
        return loss, sample_size, logging_output

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

        sample_size = sum_logs("sample_size")
        nsentences = sum_logs("nsentences")
        asr_loss = sum_logs("asr_loss")
        ce_loss = sum_logs("ce_loss")
        shrink_rate = sum_logs("shrink_rate")

        metrics.log_scalar(
            "asr_loss", asr_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ce_loss", ce_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "shrink_rate", shrink_rate / nsentences, nsentences, round=3
        )

        recall = sum_logs("asr_recall")
        precision = sum_logs("asr_precision")

        metrics.log_scalar("_asr_recall", recall / nsentences, nsentences)
        metrics.log_scalar("_asr_precision", precision / nsentences, nsentences)

        def calc_f1(meters):
            """ this is approx. """
            r = meters["_asr_recall"].avg
            p = meters["_asr_precision"].avg
            if r + p > 0:
                return safe_round(2 * p * r / (p + r), 3)
            else:
                return 0

        metrics.log_derived(
            "asr_f1",
            calc_f1,
        )
