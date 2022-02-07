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
    return exclusive_cumsum(tensor, dim)


def exclusive_cumsum(tensor, dim: int):
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
    reduction="none",
):
    """The SNNT loss is very similar to monotonic attention,
    except taking word prediction probability p(y_j | h_i, s_j)
    into account.

    Args:
        log_probs (Tensor): Word prediction log-probs, should be output of log_softmax.
            tensor with shape (N, T, S, V)
            where N is the minibatch size, T is the maximum number of
            output labels, S is the maximum number of input frames and V is
            the vocabulary of labels.
        targets (Tensor): Tensor with shape (N, T) representing the
            reference target labels for all samples in the minibatch.
        log_p_choose (Tensor): emission log-probs, should be output of F.logsigmoid.
            tensor with shape (N, T, S)
            where N is the minibatch size, T is the maximum number of
            output labels, S is the maximum number of input frames.
        source_lengths (Tensor): Tensor with shape (N,) representing the
            number of frames for each sample in the minibatch.
        target_lengths (Tensor): Tensor with shape (N,) representing the
            length of the transcription for each sample in the minibatch.
        neg_inf (float, optional): The constant representing -inf used for masking.
            Default: -1e4
        reduction (string, optional): Specifies reduction. suppoerts mean / sum.
            Default: None.
    """
    prob_check(log_probs, neg_inf=neg_inf, logp=True)
    prob_check(log_p_choose, neg_inf=neg_inf, logp=True)

    # p_choose: bsz, tgt_len, src_len
    bsz, tgt_len, src_len = log_p_choose.size()
    # dtype = log_p_choose.dtype
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

    return -log_alpha


def ssnt_loss_mem(
    log_probs: Tensor,
    targets: Tensor,
    log_p_choose: Tensor,
    source_lengths: Tensor,
    target_lengths: Tensor,
    neg_inf: float = -1e4,
    reduction="mean",
):
    """The memory efficient implementation concatenates along the targets
    dimension to reduce wasted computation on padding positions.

    Assuming the summation of all targets in the batch is T_flat, then
    the original B x T x ... tensor is reduced to T_flat x ...

    The input tensors can be obtained by using target mask:
    Example:
        >>> target_mask = targets.ne(pad)   # (B, T)
        >>> targets = targets[target_mask]  # (T_flat,)
        >>> log_probs = log_probs[target_mask]  # (T_flat, S, V)

    Args:
        log_probs (Tensor): Word prediction log-probs, should be output of log_softmax.
            tensor with shape (T_flat, S, V)
            where T_flat is the summation of all target lengths,
            S is the maximum number of input frames and V is
            the vocabulary of labels.
        targets (Tensor): Tensor with shape (T_flat,) representing the
            reference target labels for all samples in the minibatch.
        log_p_choose (Tensor): emission log-probs, should be output of F.logsigmoid.
            tensor with shape (T_flat, S)
            where T_flat is the summation of all target lengths,
            S is the maximum number of input frames.
        source_lengths (Tensor): Tensor with shape (N,) representing the
            number of frames for each sample in the minibatch.
        target_lengths (Tensor): Tensor with shape (N,) representing the
            length of the transcription for each sample in the minibatch.
        neg_inf (float, optional): The constant representing -inf used for masking.
            Default: -1e4
        reduction (string, optional): Specifies reduction. suppoerts mean / sum.
            Default: None.
    """
    prob_check(log_probs, neg_inf=neg_inf, logp=True)
    prob_check(log_p_choose, neg_inf=neg_inf, logp=True)

    bsz = source_lengths.size(0)
    tgt_len_flat, src_len = log_p_choose.size()
    log_p_choose = log_p_choose.float()

    source_lengths_rep = torch.repeat_interleave(source_lengths, target_lengths, dim=0)
    source_padding_mask = lengths_to_padding_mask(source_lengths_rep)
    if source_padding_mask.any():
        assert source_padding_mask.size() == log_p_choose.size()
        log_p_choose = log_p_choose.masked_fill(source_padding_mask, neg_inf)

    def clamp_logp(x, min=neg_inf, max=0):
        return x.clamp(min=min, max=max)

    # cumprod_1mp : tgt_len_flat, src_len
    log_cumprod_1mp = log_exclusive_cumprod(
        log1mexp(log_p_choose), dim=-1)

    log_alpha = log_p_choose.new_zeros([tgt_len_flat + bsz, src_len])
    offsets = exclusive_cumsum(target_lengths, dim=0)
    offsets_out = exclusive_cumsum(target_lengths + 1, dim=0)
    log_alpha[offsets_out, 1:] = neg_inf

    for i in range(target_lengths.max()):
        # log_probs:    tgt_len_flat, src_len, vocab
        # p_choose:     tgt_len_flat, src_len
        # cumprod_1mp:  tgt_len_flat, src_len

        # operate on fake bsz (aka indices.size(0) below)
        # get p(y_i | h_*, s_i) -> bsz, src_len
        # log_probs[indices]:   bsz, src_len, vocab
        # targets[indices]:     bsz,

        indices = (offsets + i)[i < target_lengths]
        indices_out = (offsets_out + i)[i < target_lengths]
        fake_bsz = indices.numel()

        logp_trans = (
            log_probs[indices]
            .gather(-1, index=targets[indices].view(fake_bsz, 1, 1).expand(-1, src_len, -1))
        ).squeeze(-1)
        log_alpha_i = clamp_logp(
            logp_trans
            + log_p_choose[indices]
            + log_cumprod_1mp[indices]
            + torch.logcumsumexp(
                log_alpha[indices_out] - log_cumprod_1mp[indices], dim=1
            )
        )
        log_alpha[indices_out + 1] = log_alpha_i

    # alpha: tgt_len_flat + bsz, src_len
    # seq-loss: alpha(J, I)
    # pick target endpoints (bsz, src_len)
    log_alpha = log_alpha[offsets_out + target_lengths]
    # pick source endpoints
    log_alpha = log_alpha.gather(
        dim=-1,
        index=(source_lengths - 1).view(bsz, 1)
    ).view(bsz)

    prob_check(log_alpha, neg_inf=neg_inf, logp=True)

    if reduction == "sum":
        log_alpha = log_alpha.sum()
    elif reduction == "mean":
        log_alpha = log_alpha.mean()

    return -log_alpha
