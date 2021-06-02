# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
# import numpy as np
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from typing import Optional
from fairseq import utils, metrics
from fairseq.criterions import (
    FairseqCriterion,
    register_criterion,
)
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterionConfig,
    LabelSmoothedCrossEntropyCriterion
)
import logging

logger = logging.getLogger(__name__)

def calc_recall_precision(predict, target, pad_idx=1, eps=1e-8):
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

    match = torch.min(target_words, pred_words).sum(-1)
    recall = match / (target.ne(pad_idx).sum(-1) + eps)
    precision = match / (predict.ne(pad_idx).sum(-1) + eps)
    return recall.sum(), precision.sum()

@dataclass
class LabelSmoothedCTCCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    decoder_use_ctc: bool = field(
        default=False,
        metadata={"help": "use ctcloss for decoder loss."},
    )
    zero_infinity: Optional[bool] = field(
        default=True,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    report_sinkhorn_dist: bool = field(
        default=False,
        metadata={"help": "print sinkhorn distance value."},
    )

@register_criterion(
    "label_smoothed_ctc", dataclass=LabelSmoothedCTCCriterionConfig
)
class LabelSmoothedCTCCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, cfg, task):
        super().__init__(
            task,
            cfg.sentence_avg,
            cfg.label_smoothing,
            ignore_prefix_size=cfg.ignore_prefix_size,
            report_accuracy=cfg.report_accuracy
        )
        self.decoder_use_ctc = cfg.decoder_use_ctc
        if self.decoder_use_ctc:
            logger.info("Using ctc loss for decoder!")

        self.blank_idx = task.target_dictionary.index(task.blank_symbol) if hasattr(task, 'blank_symbol') else 0
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.zero_infinity = cfg.zero_infinity
        self.report_sinkhorn_dist = cfg.report_sinkhorn_dist

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        if self.decoder_use_ctc:
            loss, nll_loss = self.compute_ctc_loss(model, net_output, sample["target"], reduce=reduce)
        else:
            # original label smoothed xentropy loss by fairseq
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        if self.report_sinkhorn_dist:
            # # compute sinkhorn distance
            # attn = net_output[1]["attn"][0].float()
            # cost = -net_output[1]["log_alpha"][0].float()
            # pad_mask = net_output[1]["padding_mask"]
            # if pad_mask is not None:
            #     cost = cost.masked_fill(
            #         pad_mask.unsqueeze(1), 0
            #     ).masked_fill(
            #         pad_mask.unsqueeze(2), 0
            #     )
            # B, S, denom = attn.size()
            # dist = (cost * attn).mean() * B * S
            # loss = loss * 0.7 + dist * 0.3

            with torch.no_grad():
                attn = net_output[1]["attn"][0].float()
                cost = -net_output[1]["log_alpha"][0].float()
                pad_mask = net_output[1]["padding_mask"]

                B, S, denom = attn.size()
                dist = (cost * attn).mean() * B * S

                # compute inversion rate
                # expected value of position in source that aligns to each target
                alignment = (utils.new_arange(attn) * attn).sum(-1)  # (N, L1)
                inv_rate = alignment[:, :-1] - alignment[:, 1:]
                inv_rate = (inv_rate / denom).clamp(min=0).float().sum()
                # inv_rate = alignment[:, 1:] < alignment[:, :-1]
                # inv_rate = inv_rate.float().sum() / denom

                try:
                    entropy = Categorical(probs=attn.float()).entropy().sum() / denom
                except ValueError:
                    logger.warning("entropy calculation failed because of invalid input!")
                    entropy = 0

        else:
            dist = inv_rate = entropy = 0

        if self.report_accuracy:
            encoder_out = net_output[1]["encoder_out"]
            encoder_states = encoder_out["causal_out"][0] \
                if "causal_out" in encoder_out else encoder_out["encoder_out"][0]
            with torch.no_grad():
                x = encoder_states
                logits = model.output_layer(x.permute(1, 0, 2))
                y_pred = logits.argmax(-1)
                recall, precision = calc_recall_precision(y_pred, sample["target"])
                blank_rate = y_pred.eq(self.blank_idx).float().mean()
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
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "sinkhorn_dist": dist,
            "inv_rate": inv_rate,
            "matching_entropy": entropy,

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
        if net_output[1]["padding_mask"] is not None:
            non_padding_mask = ~net_output[1]["padding_mask"]
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
        if net_output[1]["padding_mask"] is not None:
            smooth_loss.masked_fill_(
                net_output[1]["padding_mask"],
                0.0
            )
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

        inv_rate = sum_logs("inv_rate")
        sinkhorn_dist_sum = sum_logs("sinkhorn_dist")
        sample_size = sum_logs("sample_size")
        nsentences = sum_logs("nsentences")
        matching_entropy = sum_logs("matching_entropy")
        recall = sum_logs("recall")
        precision = sum_logs("precision")
        blank_rate = sum_logs("blank_rate")

        metrics.log_scalar(
            "inversion_rate", inv_rate / nsentences, nsentences, round=3
        )
        metrics.log_scalar(
            "sinkhorn_dist", sinkhorn_dist_sum / nsentences, nsentences, round=3
        )
        metrics.log_scalar(
            "matching_entropy", matching_entropy / nsentences, nsentences, round=3
        )

        metrics.log_scalar(
            "recall", recall / nsentences, nsentences, round=3
        )
        metrics.log_scalar(
            "precision", precision / nsentences, nsentences, round=3
        )
        metrics.log_scalar(
            "blank_rate", blank_rate, 1, round=3
        )
