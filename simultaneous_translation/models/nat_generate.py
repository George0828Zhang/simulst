import torch
from typing import Dict, List, Optional, Tuple
from torch import Tensor
import torch.nn.functional as F

def generate(model, src_tokens, src_lengths, net_output=None, blank_idx=0, collapse=True, **unused):
    """
    lprobs is expected to be batch first. (from model forward output, or net_output)
    """

    if net_output is None:
        net_output = model.forward(src_tokens, src_lengths, None)
    lprobs = model.get_normalized_probs(
        net_output, log_probs=True
    )  # lprobs = F.log_softmax(net_output[0], dim=-1)
    
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
        lp = lp[:inp_l]  # (inp_l, vocab) #.unsqueeze(0)

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
