# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
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
from fairseq.dataclass import ChoiceEnum
from fairseq.data import data_utils
from fairseq.tasks import FairseqTask
import logging
from omegaconf import II

from entmax import entmax15

logger = logging.getLogger(__name__)

def cos_dist(x, y):
    """Computes batched the cosine distance between each pair of the two collections of row vectors.
    shape: (B, L, E)
    (B, 1, L, E) (B, L, 1, E) -> (B, L, L)
    """
    cos = F.cosine_similarity(
        x.unsqueeze(1),
        y.unsqueeze(2),
        dim=-1,
    )
    return 1. - cos

def l2_dist(x, y):
    """Computes batched the 2-norm distance between each pair of the two collections of row vectors.
    """
    cost = torch.cdist(x, y, p=2)
    return cost

def dot_dist(x, y):
    """Computes scaled dot product between each pair of the two collections of row vectors.
    """
    attn = torch.bmm(x, y.permute(0, 2, 1)) * (x.size(-1) ** -0.5)
    return -attn

def sample_gumbel(proto, eps=1e-8):
    u = torch.empty_like(proto).uniform_(0, 1)
    return -torch.log(-torch.log(u + eps) + eps)

def log_sinkhorn_norm(log_alpha: torch.Tensor, n_iter: int = 20) -> (torch.Tensor,):
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
    return log_alpha.exp()

def gumbel_sinkhorn(log_alpha: torch.Tensor, tau: float = 0.7, n_iter: int = 20, noise: bool = True) -> (torch.Tensor,):
    if noise:
        gumbel_noise = sample_gumbel(log_alpha)
        log_alpha = (log_alpha + gumbel_noise) / tau
    sampled_perm_mat = log_sinkhorn_norm(log_alpha, n_iter)
    return sampled_perm_mat

@dataclass
class LabelSmoothedCrossEntropySinkhornCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    aux_factor: Optional[float] = field(
        default=1.0,
        metadata={"help": "loss scaling factor for auxiliary loss."},
    )
    aux_type: Optional[ChoiceEnum(["l2", "dot", "cos"])] = field(  # noqa: F821
        default="cos", metadata={"help": "types of loss to use l2, dot, cos"}
    )
    stop_grad_embeddings: Optional[bool] = field(
        default=False,
        metadata={"help": "stop gradient of auxiliary loss flowing through decoder hidden states / embeddings."},
    )
    sinkhorn_temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "temperature for gumbel sinkorn. the higher, the sparser the output permutation."},
    )
    sinkhorn_iters: Optional[int] = field(
        default=20,
        metadata={"help": "number of iterations for sinkhorn normalization."},
    )

@register_criterion(
    "label_smoothed_cross_entropy_sinkhorn", dataclass=LabelSmoothedCrossEntropySinkhornCriterionConfig
)
class LabelSmoothedCrossEntropySinkhornCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, cfg, task):
        super().__init__(
            task,
            cfg.sentence_avg,
            cfg.label_smoothing,
        )
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()

        self.aux_factor = cfg.aux_factor
        self.stop_grad_embeddings = cfg.stop_grad_embeddings
        self.sinkhorn_temperature = cfg.sinkhorn_temperature
        self.sinkhorn_iters = cfg.sinkhorn_iters

        if cfg.aux_type == "cos":
            self.dist_fn = cos_dist
        elif cfg.aux_type == "l2":
            self.dist_fn = l2_dist
        elif cfg.aux_type == "dot":
            self.dist_fn = dot_dist
        else:
            raise NotImplementedError(f"auxiliary loss {cfg.aux_type} not found.")

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        ls_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        sinkhorn_dist, inv_rate, entropy = self.compute_sinkhorn_distance(
            model, net_output, sample)

        loss = (ls_loss / sample_size) + sinkhorn_dist * self.aux_factor
        sample_size = 1

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ls_loss": ls_loss.data,    # w/ label smooth
            "sinkhorn_dist": sinkhorn_dist.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,

            "inv_rate": inv_rate,
            "matching_entropy": entropy,
        }
        return loss, sample_size, logging_output

    def compute_sinkhorn_distance(self, model, net_output, sample):
        """ """
        # logits, extra = net_output
        decoder_states = net_output[1]["decoder_states"]
        target = model.get_targets(sample, net_output)

        padding_mask = target.eq(self.pad_idx)  # | target.eq(self.eos_idx)

        N, L, E = decoder_states.size()
        target_emb = model.forward_embeddings(target)
        if self.stop_grad_embeddings:
            target_emb.detach_()

        cost = self.dist_fn(decoder_states, target_emb)
        if padding_mask.any():
            # mask out non-pad -> pad attentions.
            log_alpha = (-cost).float().masked_fill(
                padding_mask.unsqueeze(1) & (~padding_mask).unsqueeze(2),
                float("-inf")
            ).type_as(cost)
        else:
            log_alpha = -cost

        attn = gumbel_sinkhorn(
            log_alpha,
            tau=self.sinkhorn_temperature,
            n_iter=self.sinkhorn_iters,
            noise=True
        )

        loss = cost * attn
        if padding_mask.any():
            loss.masked_fill_(
                padding_mask.unsqueeze(2), 0
            )
        loss = loss.mean() * L

        with torch.no_grad():
            # compute inversion rate
            # expected value of position in source that aligns to each target
            alignment = (utils.new_arange(attn) * attn).sum(-1)  # (N, T)
            inv_rate = alignment[:, 1:] < alignment[:, :-1]
            inv_rate = inv_rate.float().mean(-1).sum().item()

            attn = attn[attn > 0]
            entropy = -(attn * attn.log()).sum().data / (L * N)
        return loss, inv_rate, entropy

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

        metrics.log_scalar(
            "inversion_rate", inv_rate / nsentences, nsentences, round=3
        )
        metrics.log_scalar(
            "sinkhorn_dist", sinkhorn_dist_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "matching_entropy", matching_entropy / sample_size, sample_size, round=3
        )
