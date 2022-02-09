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
    LabelSmoothedCrossEntropyCriterionConfig,
    LabelSmoothedCrossEntropyCriterion
)
import logging
from codebase.criterion.ssnt_loss import ssnt_loss, ssnt_loss_mem

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
    fastemit_lambda: Optional[float] = field(
        default=0.01,
        metadata={
            "help":
            "scales the non-blank prediction gradient by 1 + lambda to encourage faster emission"
        },
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
        self.fastemit_lambda = cfg.fastemit_lambda

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_ssnt_loss(
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

        emit_logits = net_output[1]["emit_logits"]

        bsz = targets.size(0)
        if self.memory_efficient:
            target_masks = targets.ne(self.pad_idx)
            target_lengths = target_masks.long().sum(-1)
            targets = targets[target_masks]
            _bsz = target_lengths.sum().item()
            if lprobs.size(0) != _bsz:
                raise RuntimeError(
                    f'batch size error: lprobs shape={lprobs.size()}, bsz={_bsz}')
            if emit_logits.size(0) != _bsz:
                raise RuntimeError(
                    f'batch size error: emit_logits shape={emit_logits.size()}, bsz={_bsz}')
        else:
            target_lengths = sample["target_lengths"]
            if lprobs.size(0) != bsz:
                raise RuntimeError(
                    f'batch size error: lprobs shape={lprobs.size()}, bsz={bsz}')
            if emit_logits.size(0) != bsz:
                raise RuntimeError(
                    f'batch size error: emit_logits shape={emit_logits.size()}, bsz={bsz}')

        max_src = lprobs.size(-2)
        lprobs = lprobs.contiguous()
        emit_logits = emit_logits.contiguous()

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
                input_lengths,
                target_lengths,
                emit_logits=emit_logits,
                reduction="sum",
                fastemit_lambda=self.fastemit_lambda
            )
        else:
            loss = ssnt_loss(
                lprobs,
                targets,
                input_lengths,
                target_lengths,
                emit_logits=emit_logits,
                reduction="sum",
                fastemit_lambda=self.fastemit_lambda
            )

        return loss, loss.detach()
