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
from codebase.criterion.ssnt_loss import ssnt_loss, ssnt_loss_mem

logger = logging.getLogger(__name__)


@dataclass
class SSNTCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    memory_efficient: Optional[bool] = field(
        default=False,
        metadata={"help": "concatenate the 'targets' dimension to improve efficiency"},
    )
    fastemit_lambda: Optional[float] = field(
        default=0.0,
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
        self.fastemit_lambda = cfg.fastemit_lambda
        self.offline_lambda = cfg.offline_lambda
        self.memory_efficient = cfg.memory_efficient

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
        }
        return loss, sample_size, logging_output

    def compute_ssnt_loss(self, model, net_output, sample, reduce=True):
        """
        B x T x S x V
        """
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        )
        targets = sample["target"]
        target_lengths = sample["target_lengths"]
        target_masks = targets.ne(self.pad_idx)

        emit_logits = net_output[1]["emit_logits"]

        bsz = targets.size(0)
        if self.memory_efficient:
            targets = targets[target_masks]
            T_flat = target_lengths.sum().item()
            if lprobs.size(0) != T_flat:
                raise RuntimeError(
                    f'batch size error: lprobs shape={lprobs.size()}, bsz={T_flat}')
            if emit_logits.size(0) != T_flat:
                raise RuntimeError(
                    f'batch size error: emit_logits shape={emit_logits.size()}, bsz={T_flat}')
        else:
            if lprobs.size(0) != bsz:
                raise RuntimeError(
                    f'batch size error: lprobs shape={lprobs.size()}, bsz={bsz}')
            if emit_logits.size(0) != bsz:
                raise RuntimeError(
                    f'batch size error: emit_logits shape={emit_logits.size()}, bsz={bsz}')

        max_src, V = lprobs.shape[-2:]
        T = target_lengths.max()
        lprobs = lprobs.contiguous()
        emit_logits = emit_logits.contiguous()

        # get subsampling padding mask & lengths
        if net_output[1]["padding_mask"] is not None:
            non_padding_mask = ~net_output[1]["padding_mask"]
            input_lengths = non_padding_mask.long().sum(-1)
        else:
            input_lengths = targets.new_ones(
                (bsz, max_src), dtype=torch.long).sum(-1)

        if self.memory_efficient:
            loss, lattice, lprobs_emit = ssnt_loss_mem(
                lprobs,
                targets,
                input_lengths,
                target_lengths,
                emit_logits=emit_logits,
                reduction="sum",
                fastemit_lambda=self.fastemit_lambda
            )
        else:
            loss, lattice, lprobs_emit = ssnt_loss(
                lprobs,
                targets,
                input_lengths,
                target_lengths,
                emit_logits=emit_logits,
                reduction="sum",
                fastemit_lambda=self.fastemit_lambda
            )

        # offline loss computation
        # optimizes the path: last source aligned to each target
        if self.offline_lambda > 0:
            # p(yi | target_i aligned to source eos)
            # pick source_eos at each index
            if self.memory_efficient:
                # T_flat,S,V -> T_flat,1,V
                index = torch.repeat_interleave(
                    input_lengths, target_lengths, dim=0) - 1
                lprobs_label = lprobs.gather(
                    dim=1,
                    index=index.view(
                        T_flat, 1, 1).expand(T_flat, -1, V)
                )
                # T_flat,S -> T_flat,1
                lprobs_emit = lprobs_emit.gather(
                    dim=1,
                    index=index.view(
                        T_flat, 1)
                )
                off_loss_emit = -lprobs_emit.sum()
            else:
                # B,T,S,V -> B,T,1,V
                index = input_lengths - 1
                lprobs_label = lprobs.gather(
                    dim=2,
                    index=index.view(
                        bsz, 1, 1, 1).expand(bsz, T, -1, V)
                )
                # B,T,S -> B,T,1
                lprobs_emit = lprobs_emit.gather(
                    dim=2,
                    index=index.view(
                        bsz, 1, 1).expand(bsz, T, -1)
                )
                off_loss_emit = -lprobs_emit[target_masks].sum()

            off_loss, nll_loss = label_smoothed_nll_loss(
                lprobs_label.view(-1, V),
                targets.view(-1),
                self.eps,
                ignore_index=self.pad_idx,
                reduce=True,
            )
            off_loss += off_loss_emit
            nll_loss += off_loss_emit
            loss += self.offline_lambda * off_loss
        else:
            nll_loss = loss

        return loss, nll_loss
