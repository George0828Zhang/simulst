import torch
from typing import Dict, List
from torch import Tensor
from fairseq.utils import new_arange


def inject_noise(prev_output_tokens, dictionary, ratio=-1, uniform=False):
    """ mask out tokens. uniform: uniform length masking """
    pad = dictionary.pad()
    bos = dictionary.bos()
    eos = dictionary.eos()
    unk = dictionary.unk()

    # move eos to the back
    N, T = prev_output_tokens.shape[:2]
    target_tokens = torch.cat(
        (
            prev_output_tokens[:, 1:],
            prev_output_tokens.new_full((N, 1), pad)
        ), dim=1
    )
    target_length = target_tokens.ne(pad).sum(1, keepdim=True)
    target_tokens.scatter_(1, target_length, eos)

    if not uniform:
        assert 0 <= ratio <= 1, "mask ratio invalid."
        if ratio == 0:
            return target_tokens, target_tokens.eq(pad)

    target_masks = (
        target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
    )
    target_score = target_tokens.clone().float().uniform_()
    target_score.masked_fill_(~target_masks, 2.0)
    target_length = target_masks.sum(1).float()
    if uniform:
        target_length = target_length * target_length.clone().uniform_()
    else:
        target_length = target_length * target_length.clone().fill_(ratio)
    target_length = target_length + 1  # make sure to mask at least one token.

    _, target_rank = target_score.sort(1)
    target_cutoff = new_arange(target_rank) < target_length[:, None].long()
    target_tokens.masked_fill_(
        target_cutoff.scatter(1, target_rank, target_cutoff), unk
    )

    return target_tokens, target_tokens.eq(pad)


def generate(model, src_tokens, src_lengths, net_output=None, blank_idx=0, collapse=True, **unused):
    """
    lprobs is expected to be batch first. (from model forward output, or net_output)
    """

    if net_output is None:
        net_output = model.forward(src_tokens, src_lengths, None)
    lprobs = model.get_normalized_probs(
        net_output, log_probs=True
    )

    # eos_penalty = 1
    # if eos_penalty > 0.0:
    #     lprobs[:, :, blank_idx] -= eos_penalty

    # get subsampling padding mask & lengths
    if net_output[1]["padding_mask"] is not None:
        non_padding_mask = ~net_output[1]["padding_mask"]
        input_lengths = non_padding_mask.long().sum(-1)
    else:
        sum_dim = 1
        input_lengths = lprobs.new_ones(
            lprobs.shape[:2], dtype=torch.long).sum(sum_dim)

    bsz = lprobs.size(0)

    # list of completed sentences
    finalized = torch.jit.annotate(
        List[List[Dict[str, Tensor]]],
        [torch.jit.annotate(List[Dict[str, Tensor]], [])
            for i in range(bsz)],
    )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

    # TODO faster argmax before for loop?
    for sent, lp, inp_l in zip(
        range(bsz),
        lprobs,
        input_lengths,
    ):
        lp = lp[:inp_l]

        toks = lp.argmax(dim=-1)
        score = torch.index_select(
            lp.view(inp_l, -1), -1, toks.view(-1)).sum()
        if collapse:
            toks = toks.unique_consecutive()
        if toks.eq(blank_idx).all():
            toks = toks[:1]
        else:
            toks = toks[toks != blank_idx]

        p_score = torch.zeros_like(toks).float()

        finalized[sent].append(
            {
                "tokens": toks,
                "score": score,
                "attention": None,  # src_len x tgt_len
                "alignment": torch.empty(0),
                "positional_scores": p_score,
            }
        )
    return finalized
