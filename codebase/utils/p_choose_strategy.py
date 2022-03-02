from typing import Optional, Dict
from torch import Tensor
import torch


def waitk_p_choose(
    tgt_len: int,
    src_len: int,
    bsz: int,
    waitk_lagging: int,
    key_padding_mask: Optional[Tensor] = None,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None
):

    max_src_len = src_len
    max_tgt_len = tgt_len

    if max_src_len < waitk_lagging:
        if incremental_state is not None:
            max_tgt_len = 1
        return torch.zeros(
            bsz, max_tgt_len, max_src_len
        )

    # Assuming the p_choose looks like this for wait k=3
    # src_len = 6, max_tgt_len = 5
    #   [0, 0, 1, 0, 0, 0, 0]
    #   [0, 0, 0, 1, 0, 0, 0]
    #   [0, 0, 0, 0, 1, 0, 0]
    #   [0, 0, 0, 0, 0, 1, 0]
    #   [0, 0, 0, 0, 0, 0, 1]

    p_choose = (
        torch.ones(max_tgt_len, max_src_len)
        .triu(waitk_lagging - 1)
        .tril(waitk_lagging - 1)
        .unsqueeze(0)
        .repeat_interleave(bsz, 0)
    )
    if key_padding_mask is not None:
        p_choose = p_choose.to(key_padding_mask)
        p_choose = p_choose.masked_fill(key_padding_mask.unsqueeze(1), 0)

    if incremental_state is not None:
        p_choose = p_choose[:, -1:]

    return p_choose


def learnable_p_choose(
    energy,
    noise_mean: float = 0.0,
    noise_std: float = 1.0,
    training: bool = True
):
    """
    Calculating step wise prob for reading and writing
    0 to read, 1 to write
    energy: bsz, tgt_len, src_len
    """

    noise = 0
    if training:
        # add noise here to encourage discretness
        noise = torch.randn_like(energy) * noise_std + noise_mean

    p_choose = torch.sigmoid(energy + noise)

    # p_choose: bsz * self.num_heads, tgt_len, src_len
    return p_choose
