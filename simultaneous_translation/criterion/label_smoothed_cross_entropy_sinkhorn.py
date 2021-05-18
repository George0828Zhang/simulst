# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
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

def sample_gumbel(proto, eps=1e-6):
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
    cost_fn: Optional[ChoiceEnum(["l2", "dot", "cos"])] = field(  # noqa: F821
        default="cos", metadata={"help": "types of loss to use l2, dot, cos"}
    )
    stop_grad_embeddings: Optional[bool] = field(
        default=False,
        metadata={"help": "stop gradient of auxiliary loss flowing through decoder hidden states / embeddings."},
    )
    sinkhorn_temperature: Optional[float] = field(
        default=0.75,
        metadata={"help": "temperature for gumbel sinkorn. the higher, the sparser the output permutation."},
    )
    sinkhorn_iters: Optional[int] = field(
        default=8,
        metadata={"help": "number of iterations for sinkhorn normalization."},
    )
    sinkhorn_bucket_size: Optional[int] = field(
        default=7,
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
        self.sinkhorn_bucket_size = cfg.sinkhorn_bucket_size

        if cfg.cost_fn == "cos":
            self.cost_fn = cos_dist
        elif cfg.cost_fn == "l2":
            self.cost_fn = l2_dist
        elif cfg.cost_fn == "dot":
            self.cost_fn = dot_dist
        else:
            raise NotImplementedError(f"auxiliary loss {cfg.cost_fn} not found.")

        self.log_penalty_mask = torch.empty(0)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        1. Forward Encoder to get causal speech states
        2. Compute sinkhorn attnetion with targets to obtain sinkhorn loss and reorder matrix.
        3. reorder targets based on reorder matrix
        4. compute wait-k teacher forcing on decoder.
        """
        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        encoder_out = model.encoder(src_tokens=src_tokens, src_lengths=src_lengths)

        sink_dist, attn, extras = self.compute_sinkhorn_distance(model, encoder_out, sample)

        # replace and return as well
        prev_tokens, tgt_tokens = self.reorder_tokens(attn, sample)

        net_output = model.decoder(
            prev_output_tokens=prev_tokens,
            encoder_out=encoder_out,
        )
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        loss = (1 - self.aux_factor) * loss + self.aux_factor * sink_dist

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
        logging_output.update(extras)
        return loss, sample_size, logging_output

    def reorder_tokens(self, attn, sample):
        """ shapes:
                attn (N, T, S)
                prev_output_tokens (N, T)
                target_tokens (N, T)

            This function assumes torch.sort is stable sorting... wouldnt work otherwise.
        """
        prev_tokens, target = sample["net_input"]["prev_output_tokens"], sample["target"]

        def reorder_fn(tokens, eos_begin=False):
            sort_key = attn.argmax(-1)  # (N, T)
            sort_key.masked_fill_(tokens.eq(self.pad_idx), 1e6)  # will sort to last
            sort_key.masked_fill_(tokens.eq(self.eos_idx), -1 if eos_begin else 1e6)  # will sort to first or last

            sort_ord = sort_key.argsort(dim=1)
            return torch.gather(tokens, dim=1, index=sort_ord)

        # reduce memory footprint.
        sample["net_input"]["prev_output_tokens"] = reorder_fn(prev_tokens, eos_begin=True)
        sample["target"] = reorder_fn(target)

        return sample["net_input"]["prev_output_tokens"], sample["target"]

    def buffered_log_penalty(self, tensor):
        dim = tensor.size(1)  # batch first
        # self.log_penalty_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self.log_penalty_mask.size(0) == 0
            or (not self.log_penalty_mask.device == tensor.device)
            or self.log_penalty_mask.size(0) < dim
        ):
            penalty = torch.arange(dim, dtype=torch.float)
            penalty = torch.abs(
                penalty.unsqueeze(1) - penalty
            ).clamp(min=1)
            self.log_penalty_mask = penalty.log()
        self.log_penalty_mask = self.log_penalty_mask.to(tensor)
        return self.log_penalty_mask[:dim, :dim]

    def compute_sinkhorn_distance(self, model, encoder_out, sample):
        """ """
        # encoder states is key (value)
        k = encoder_out["encoder_out"][0].transpose(1, 0)

        # target embed is query
        target = sample["target"]
        q = model.forward_embeddings(target)  # need gradients!

        # masks
        key_padding_mask = encoder_out["encoder_padding_mask"][0] \
            if len(encoder_out["encoder_padding_mask"]) > 0 else None
        query_padding_mask = target.eq(self.pad_idx) | target.eq(self.eos_idx)

        # S = S // bucket_size; T = S
        k, key_padding_mask, _ = self.pad_to_multiple(
            k, key_padding_mask)
        q, k, qk_padding_mask, n_query_padding = self.aggregate_buckets(
            q, k, query_padding_mask, key_padding_mask)

        N, S, E = k.size()

        # compute costs (N, T, E), (N, L2, E) -> (N, T, S)
        cost = self.cost_fn(q.float(), k.float())

        # masked positions have inf cost
        # cost (N, T, S)
        # query_padding_mask (N, T)
        # key_padding_mask (N, S)
        log_alpha = -cost - self.buffered_log_penalty(cost).unsqueeze(0)
        # log_alpha = log_alpha.float()  # does not work with fp16. cast to fp32
        if qk_padding_mask is not None:
            final_mask = qk_padding_mask.unsqueeze(1) & (~qk_padding_mask).unsqueeze(2)
            # mask out normal -> pad attentions
            log_alpha.masked_fill_(
                final_mask,
                float("-inf"),
            )
            # mask out pad -> normal attentions
            log_alpha.masked_fill_(
                final_mask.transpose(2, 1),
                float("-inf"),
            )

        # import pdb; pdb.set_trace()
        # compute sinkhorn
        attn = gumbel_sinkhorn(
            log_alpha,
            tau=self.sinkhorn_temperature,
            n_iter=self.sinkhorn_iters,
            noise=True
        )

        # compute sinkhorn distance
        dist = (cost * attn).sum()  # should be divided by n_tokens

        with torch.no_grad():
            # compute inversion rate
            # expected value of position in source that aligns to each target
            alignment = (utils.new_arange(attn) * attn).sum(-1)  # (N, L1)
            inv_rate = alignment[:, 1:] < alignment[:, :-1]
            inv_rate = inv_rate.float().sum() / S

            try:
                p_attn = F.normalize(F.relu(attn) + 1e-8, p=1.0, dim=-1)
                entropy = Categorical(probs=p_attn).entropy().sum() / S
            except ValueError:
                entropy = 0

        attn = attn[:, :-n_query_padding, :] if n_query_padding > 0 else attn
        return dist, attn.data, {
            "sinkhorn_dist": dist.data,
            "inv_rate": inv_rate.data,
            "matching_entropy": entropy,
            # "n_dummy": n_dummy
        }

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

    def pad_to_multiple(self, k, key_padding_mask):
        """Input shape
            states: (B, S, E),
            key_padding_mask: (B, S),
        """
        B, S, E = k.size()

        new_k = k
        new_key_padding_mask = key_padding_mask

        kv_buckets = math.ceil(S / self.sinkhorn_bucket_size)

        # pad key value
        new_S = kv_buckets * self.sinkhorn_bucket_size
        if new_S != S:
            new_k = torch.cat([
                k,
                k.new_zeros((B, new_S - S, E)),
            ], dim=1)
            if key_padding_mask is None:
                key_padding_mask = k.new_zeros((B, S), dtype=torch.bool)

            new_key_padding_mask = torch.cat([
                key_padding_mask,
                key_padding_mask.new_ones((B, new_S - S)),
            ], dim=1)

        return (
            new_k,
            new_key_padding_mask,
            new_S - S
        )

    def aggregate_buckets(self, q, k, query_padding_mask, key_padding_mask):
        """Input shape
            q: (B, T, E),
            k: (B, S, E),
            query_padding_mask: (B, T),
            key_padding_mask: (B, S),
        """
        B, T, E = q.size()
        B, S, E = k.size()
        kv_buckets = S // self.sinkhorn_bucket_size

        # aggregate key by meaning (summing in paper?) each buckets
        new_k = k.view(B, kv_buckets, self.sinkhorn_bucket_size, E).mean(dim=2)

        # aggregate padding mask by: if a bucket is all pad then it is masked.
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(
                B, kv_buckets, self.sinkhorn_bucket_size).prod(dim=2).type_as(key_padding_mask)        

        # add dummy points to query
        new_q = q
        new_query_padding_mask = query_padding_mask
        new_key_padding_mask = key_padding_mask
        n_padding = kv_buckets - T
        if n_padding > 0:
            new_q = torch.cat([
                q,
                q.new_zeros((B, n_padding, E)),
            ], dim=1)
            if query_padding_mask is None:
                query_padding_mask = q.new_zeros((B, T), dtype=torch.bool)

            new_query_padding_mask = torch.cat([
                query_padding_mask,
                query_padding_mask.new_ones((B, n_padding)),
            ], dim=1)

        elif n_padding < 0:
            new_k = torch.cat([
                new_k,
                new_k.new_zeros((B, -n_padding, E)),
            ], dim=1)
            if key_padding_mask is None:
                key_padding_mask = k.new_zeros((B, kv_buckets), dtype=torch.bool)

            new_key_padding_mask = torch.cat([
                key_padding_mask,
                key_padding_mask.new_ones((B, -n_padding)),
            ], dim=1)

        # zero out dummy (and padding) points so that they are 0-cost
        if new_query_padding_mask is not None:
            new_q.masked_fill_(
                new_query_padding_mask.unsqueeze(-1), 0.
            )
        if new_key_padding_mask is not None:
            new_k.masked_fill_(
                new_key_padding_mask.unsqueeze(-1), 0.
            )

        # determine much many pads are considered as dummy
        # convert some paddings into dummies by increasing non pad tokens
        B, S, E = new_q.size()
        query_len = new_q.new_full((B,), S) if new_query_padding_mask is None else (~new_query_padding_mask).sum(-1)
        key_len = new_k.new_full((B,), S) if new_key_padding_mask is None else (~new_key_padding_mask).sum(-1)

        final_len = torch.max(query_len, key_len)
        if final_len.eq(S).all():
            final_padding_mask = None
        else:
            final_padding_mask = lengths_to_padding_mask(final_len, S)

        return (
            new_q,
            new_k,
            final_padding_mask,
            n_padding
        )

    # def undo_aggregate_buckets(self, v, tail_v):
    #     """Input shape
    #         v: (B, new_S, E),
    #     """
    #     B, kv_buckets, bucket_size_E = v.size()
    #     E = bucket_size_E // self.sinkhorn_bucket_size
    #     new_v = v.view(B, kv_buckets * self.sinkhorn_bucket_size, E)
    #     return new_v[:, :-tail_v, :] if tail_v > 0 else new_v

def lengths_to_padding_mask(lengths, max_length=None):
    """Convert lengths of shape (B, ) to padding mask."""
    batch_size = lengths.shape[0]
    if max_length is None:
        max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(  # [0, ..., T-1]
        max_length, device=lengths.device, dtype=lengths.dtype
    ).expand(batch_size, max_length) >= lengths.unsqueeze(1)

    return padding_mask
