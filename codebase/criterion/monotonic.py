import torch
from typing import Optional
from torch import Tensor


def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


def log1mexp(x):
    # Computes log(1-exp(x)) assuming x is all negative
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    _x = x.float()
    out = torch.where(_x > -0.693, torch.log(-torch.expm1(_x)), torch.log1p(-torch.exp(_x)))
    return out.type_as(x)


def log_exclusive_cumprod(tensor, dim: int):
    """
    Implementing exclusive cumprod in log space (assume tensor is in log space)
    exclusive cumprod(x) = [1, x1, x1x2, x1x2x3, ..., prod_{i=1}^{n-1} x_i]
    """
    tensor = tensor.roll(1, dims=dim)  # right shift 1
    tensor.select(dim, 0).fill_(0)
    tensor = tensor.cumsum(dim)
    return tensor


def prob_check(tensor, eps=1e-10, neg_inf=-1e8, logp=True):
    assert not torch.isnan(tensor).any(), (
        "Nan in a probability tensor."
    )
    # Add the eps here to prevent errors introduced by precision
    if logp:
        assert tensor.le(0).all() and tensor.ge(neg_inf).all(), (
            "Incorrect values in a log-probability tensor"
            ", -inf <= tensor <= 0"
        )
    else:
        assert tensor.le(1.0 + eps).all() and tensor.ge(0.0 - eps).all(), (
            "Incorrect values in a probability tensor"
            ", 0.0 <= tensor <= 1.0"
        )


def expected_alignment_from_log_p_choose(
    log_p_choose: Tensor,
    padding_mask: Optional[Tensor] = None,
    neg_inf: float = -1e4
):
    """
    Calculating expected alignment for from stepwise probability
    Assuming p_choose is given in log space (output from torch.log_sigmoid)
    Reference:
    Online and Linear-Time Attention by Enforcing Monotonic Alignments
    https://arxiv.org/pdf/1704.00784.pdf
    q_ij = (1 − p_{ij−1})q_{ij−1} + a+{i−1j}
    a_ij = p_ij q_ij
    Parallel solution:
    ai = p_i * cumprod(1 − pi) * cumsum(a_i / cumprod(1 − pi))
    ============================================================
    Expected input size
    p_choose: bsz, tgt_len, src_len
    """
    prob_check(log_p_choose, neg_inf=neg_inf, logp=True)

    # p_choose: bsz, tgt_len, src_len
    bsz, tgt_len, src_len = log_p_choose.size()
    dtype = log_p_choose.dtype
    log_p_choose = log_p_choose.float()

    if padding_mask is not None:
        log_p_choose = log_p_choose.masked_fill(padding_mask.unsqueeze(1), neg_inf)

    def clamp_logp(x, min=neg_inf, max=0):
        return x.clamp(min=min, max=max)

    # cumprod_1mp : bsz, tgt_len, src_len
    log_cumprod_1mp = log_exclusive_cumprod(
        log1mexp(log_p_choose), dim=2)

    log_alpha = log_p_choose.new_zeros([bsz, 1 + tgt_len, src_len])
    log_alpha[:, 0, 1:] = neg_inf

    for i in range(tgt_len):
        # p_choose: bsz , tgt_len, src_len
        # cumprod_1mp_clamp : bsz, tgt_len, src_len
        # previous_alpha[i]: bsz, src_len
        # alpha_i: bsz, src_len
        log_alpha_i = clamp_logp(
            log_p_choose[:, i]
            + log_cumprod_1mp[:, i]
            + torch.logcumsumexp(
                log_alpha[:, i] - log_cumprod_1mp[:, i], dim=1
            )
        )
        log_alpha[:, i + 1] = log_alpha_i

    # alpha: bsz, tgt_len, src_len
    log_alpha = log_alpha[:, 1:, :]

    # Mix precision to prevent overflow for fp16
    log_alpha = log_alpha.type(dtype)

    prob_check(log_alpha, neg_inf=neg_inf, logp=True)

    return log_alpha


def ssnt_loss(
    log_probs: Tensor,
    targets: Tensor,
    log_p_choose: Tensor,
    source_lengths: Tensor,
    target_lengths: Tensor,
    neg_inf: float = -1e4,
    reduction="mean",
):
    """
    Very similar to monotonic attention, except taking translation probability
    into account: p(y_j | h_i, s_j)
    log_probs: bsz, tgt_len, src_len, vocab
    targets: bsz, tgt_len
    p_choose: bsz, tgt_len, src_len
    padding_mask: bsz, src_len
    """
    prob_check(log_probs, neg_inf=neg_inf, logp=True)
    prob_check(log_p_choose, neg_inf=neg_inf, logp=True)

    # p_choose: bsz, tgt_len, src_len
    bsz, tgt_len, src_len = log_p_choose.size()
    dtype = log_p_choose.dtype
    log_p_choose = log_p_choose.float()

    source_padding_mask = lengths_to_padding_mask(source_lengths)
    if source_padding_mask.any():
        log_p_choose = log_p_choose.masked_fill(source_padding_mask.unsqueeze(1), neg_inf)

    def clamp_logp(x, min=neg_inf, max=0):
        return x.clamp(min=min, max=max)

    # cumprod_1mp : bsz, tgt_len, src_len
    log_cumprod_1mp = log_exclusive_cumprod(
        log1mexp(log_p_choose), dim=2)

    log_alpha = log_p_choose.new_zeros([bsz, 1 + tgt_len, src_len])
    log_alpha[:, 0, 1:] = neg_inf

    for i in range(tgt_len):
        # log_probs:    bsz, tgt_len, src_len, vocab
        # p_choose:     bsz, tgt_len, src_len
        # cumprod_1mp:  bsz, tgt_len, src_len
        # alpha[i]:     bsz, src_len

        # get p(y_i | h_*, s_i) -> bsz, src_len
        # log_probs[:,i]:   bsz, src_len, vocab
        # targets[:,i]:     bsz,
        logp_trans = log_probs[:, i].gather(
            dim=-1,
            index=targets[:, i].view(bsz, 1, 1).expand(-1, src_len, -1)
        ).squeeze(-1)
        log_alpha_i = clamp_logp(
            logp_trans
            + log_p_choose[:, i]
            + log_cumprod_1mp[:, i]
            + torch.logcumsumexp(
                log_alpha[:, i] - log_cumprod_1mp[:, i], dim=1
            )
        )
        log_alpha[:, i + 1] = log_alpha_i

    # alpha: bsz, 1 + tgt_len, src_len
    # seq-loss: alpha(J, I)
    # pick source endpoints
    log_alpha = log_alpha.gather(
        dim=2,
        index=(source_lengths - 1).view(bsz, 1, 1).expand(-1, 1 + tgt_len, -1)
    )
    # pick target endpoints
    log_alpha = log_alpha.gather(
        dim=1,
        index=target_lengths.view(bsz, 1, 1)
    ).view(bsz)

    prob_check(log_alpha, neg_inf=neg_inf, logp=True)

    if reduction == "sum":
        log_alpha = log_alpha.sum()
    elif reduction == "mean":
        log_alpha = log_alpha.mean()

    # Mix precision to prevent overflow for fp16
    log_alpha = log_alpha.type(dtype)

    return -log_alpha
