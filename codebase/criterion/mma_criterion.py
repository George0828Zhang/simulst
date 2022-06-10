# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)

try:
    from simuleval.metrics.latency import (
        AverageLagging,
        AverageProportion,
        DifferentiableAverageLagging,
    )

    LATENCY_METRICS = {
        "average_lagging": AverageLagging,
        "average_proportion": AverageProportion,
        "differentiable_average_lagging": DifferentiableAverageLagging,
    }
except ImportError:
    LATENCY_METRICS = None


@dataclass
class MMACriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    latency_avg_weight: float = field(
        default=0.0,
        metadata={"help": "weight fot average latency loss."},
    )
    latency_var_weight: float = field(
        default=0.0,
        metadata={"help": "weight fot variance latency loss."},
    )
    latency_avg_type: str = field(
        default="differentiable_average_lagging",
        metadata={"help": "latency type for average loss"},
    )
    latency_var_type: str = field(
        default="variance_delay",
        metadata={"help": "latency typ for variance loss"},
    )
    latency_gather_method: str = field(
        default="weighted_average",
        metadata={"help": "method to gather latency loss for all heads"},
    )
    latency_update_after: int = field(
        default=0,
        metadata={"help": "Add latency loss after certain steps"},
    )
    ms_per_frame_shift: float = field(
        default=10,
        metadata={"help": "The milliseconds per frame shift used to compute the delay."},
    )


@register_criterion(
    "mma_criterion", dataclass=MMACriterionConfig,
)
class MMACriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size,
        report_accuracy,
        latency_avg_weight,
        latency_var_weight,
        latency_avg_type,
        latency_var_type,
        latency_gather_method,
        latency_update_after,
        ms_per_frame_shift
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        assert LATENCY_METRICS is not None, "Please make sure SimulEval is installed."

        self.latency_avg_weight = latency_avg_weight
        self.latency_var_weight = latency_var_weight
        self.latency_avg_type = latency_avg_type
        self.latency_var_type = latency_var_type
        self.latency_gather_method = latency_gather_method
        self.latency_update_after = latency_update_after
        self.ms_per_frame_shift = ms_per_frame_shift

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        # 1. Compute cross entropy loss
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        # 2. Compute cross latency loss
        latency_loss, expected_latency, expected_delays_var = self.compute_latency_loss(
            model, sample, net_output
        )

        if self.latency_update_after > 0:
            num_updates = getattr(model.decoder, "num_updates", None)
            assert (
                num_updates is not None
            ), "model.decoder doesn't have attribute 'num_updates'"
            if num_updates <= self.latency_update_after:
                latency_loss = 0

        loss += latency_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "latency": expected_latency.data,
            "delays_var": expected_delays_var.data,
            "latency_loss": latency_loss.data,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_latency_loss(self, model, sample, net_output):
        # 1. Obtain the expected alignment
        alpha_list = [item["alpha"] for item in net_output[1]["attn_list"]]
        num_layers = len(alpha_list)
        bsz, num_heads, tgt_len, src_len = alpha_list[0].size()

        # bsz * num_layers * num_heads, tgt_len, src_len
        alpha_all = torch.cat(alpha_list, dim=1).view(-1, tgt_len, src_len)

        # 2 compute expected delays
        # bsz * num_heads * num_layers, tgt_len, src_len for MMA
        steps = (
            torch.arange(1, 1 + src_len)
            .unsqueeze(0)
            .unsqueeze(1)
            .expand_as(alpha_all)
            .type_as(alpha_all)
        )

        expected_delays = torch.sum(steps * alpha_all, dim=-1)

        target_padding_mask = sample["target"] == self.padding_idx
        target_lengths = (~target_padding_mask).sum(1)

        input_lengths = sample["net_input"]["src_lengths"]
        encoder_padding_mask = net_output[-1]["encoder_padding_mask"]
        if isinstance(encoder_padding_mask, list):
            encoder_padding_mask = encoder_padding_mask[0]
        assert not encoder_padding_mask[:, 0].any(), "Only right padding is supported."
        encoder_lengths = (~encoder_padding_mask).sum(-1)

        def expand(t):
            return torch.repeat_interleave(t, num_layers * num_heads, 0)

        expected_latency = LATENCY_METRICS[self.latency_avg_type](
            expected_delays,
            expand(encoder_lengths),
            expand(target_lengths),
            target_padding_mask=expand(target_padding_mask)
        )

        # 2.1 average expected latency of heads
        # bsz, num_layers * num_heads
        expected_latency = expected_latency.view(bsz, -1)
        if self.latency_gather_method == "average":
            # bsz * tgt_len
            expected_latency = expected_delays.mean(dim=1)
        elif self.latency_gather_method == "weighted_average":
            weights = torch.nn.functional.softmax(expected_latency, dim=1)
            expected_latency = torch.sum(expected_latency * weights, dim=1)
        elif self.latency_gather_method == "max":
            expected_latency = expected_latency.max(dim=1)[0]
        else:
            raise NotImplementedError

        avg_loss = self.latency_avg_weight * expected_latency.clip(min=0).sum()

        # 2.2 variance of expected delays
        expected_delays_var = (
            expected_delays.view(bsz, -1, tgt_len).var(dim=1).mean(dim=1)
        )
        expected_delays_var = expected_delays_var.sum()
        var_loss = self.latency_var_weight * expected_delays_var

        # 3. Final loss
        latency_loss = avg_loss + var_loss

        # renormalize delays to ms
        expected_latency = expected_latency * (input_lengths / encoder_lengths * self.ms_per_frame_shift)
        return latency_loss, expected_latency.sum(), expected_delays_var

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        latency = sum(log.get("latency", 0) for log in logging_outputs)
        delays_var = sum(log.get("delays_var", 0) for log in logging_outputs)
        latency_loss = sum(log.get("latency_loss", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        metrics.log_scalar("latency", latency.float() / nsentences, nsentences, round=3)
        metrics.log_scalar("delays_var", delays_var / nsentences, nsentences, round=3)
        metrics.log_scalar(
            "latency_loss", latency_loss / nsentences, nsentences, round=3
        )
