# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import torch
from typing import Optional
from fairseq.criterions import (
    register_criterion,
)
from fairseq.criterions.label_smoothed_cross_entropy import (
    label_smoothed_nll_loss,
    LabelSmoothedCrossEntropyCriterionConfig,
    LabelSmoothedCrossEntropyCriterion
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class RNNTCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    memory_efficient: Optional[bool] = field(
        default=False,
        metadata={"help": "To use the ``gather'' option in warp-rnnt."},
    )
    fastemit_lambda: Optional[float] = field(
        default=0.01,
        metadata={
            "help":
            "scales the non-blank prediction gradient by 1 + lambda to encourage faster emission"
        },
    )
    offline_lambda: Optional[float] = field(
        default=1.0,
        metadata={
            "help":
            "optimize the offline path at the same time."
        },
    )


@register_criterion(
    "rnnt_criterion", dataclass=RNNTCriterionConfig
)
class RNNTCriterion(LabelSmoothedCrossEntropyCriterion):
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
        self.fastemit_lambda = cfg.fastemit_lambda
        self.offline_lambda = cfg.offline_lambda
        self.memory_efficient = cfg.memory_efficient

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_rnnt_loss(
            model, net_output, sample)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_rnnt_loss(self, model, net_output, sample, reduce=True):
        try:
            from warp_rnnt import rnnt_loss
        except ImportError:
            raise ImportError("Please install warp_rnnt: pip install warp-rnnt")
        """
        B x S x T x V
        """
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        )
        targets = sample["target"]
        target_lengths = sample["target_lengths"]

        bsz = targets.size(0)
        if lprobs.size(0) != bsz:
            raise RuntimeError(
                f'batch size error: lprobs shape={lprobs.size()}, bsz={bsz}')

        bsz, max_src, U, V = lprobs.size()
        lprobs = lprobs.contiguous()

        # get subsampling padding mask & lengths
        if net_output[1]["padding_mask"] is not None:
            non_padding_mask = ~net_output[1]["padding_mask"]
            input_lengths = non_padding_mask.long().sum(-1)
        else:
            input_lengths = targets.new_ones(
                (bsz, max_src), dtype=torch.int32).sum(-1)

        loss = rnnt_loss(
            lprobs,
            targets.int(),
            input_lengths.int(),
            target_lengths.int(),
            average_frames=False,
            reduction="sum",
            blank=self.blank_idx,
            gather=self.memory_efficient,
            fastemit_lambda=self.fastemit_lambda
        )

        # offline loss computation
        # optimizes the path: last source aligned to each target
        if self.offline_lambda > 0:
            # p(yi | target_i aligned to source eos)
            # pick source_eos at each index
            lprobs_label = lprobs.gather(
                dim=1,
                index=(input_lengths - 1).view(
                    bsz, 1, 1, 1).expand(bsz, -1, U - 1, V)  # U -1 means discard target eos
            )
            # lprobs is now (B, 1, U-1, V)
            # targets is (B, U-1)
            off_loss, nll_loss = label_smoothed_nll_loss(
                lprobs_label.view(-1, V),
                targets.view(-1),
                self.eps,
                ignore_index=self.pad_idx,
                reduce=True,
            )
            loss += self.offline_lambda * off_loss
        else:
            nll_loss = loss

        return loss, nll_loss.detach()
