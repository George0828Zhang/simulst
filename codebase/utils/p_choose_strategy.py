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
    # Assuming the p_choose looks like this for wait k=3
    # src_len = 6, max_tgt_len = 5
    #   [0, 0, 1, 0, 0, 0, 0]
    #   [0, 0, 0, 1, 0, 0, 0]
    #   [0, 0, 0, 0, 1, 0, 0]
    #   [0, 0, 0, 0, 0, 1, 0]
    #   [0, 0, 0, 0, 0, 0, 1]

    if key_padding_mask is not None:
        key_eos = (~key_padding_mask).long().sum(-1) - 1  # a (B,) tensor indicating eos
    else:
        key_eos = torch.full((bsz,), src_len - 1)

    # for each (B, T), find a index following waitk and <= eos
    monotonic_step = (
        torch.arange(tgt_len, device=key_eos.device)
        .add(waitk_lagging - 1)
        .unsqueeze(0)
        .expand(bsz, -1)
        .clone()
    )
    monotonic_step = monotonic_step.clip(
        max=key_eos.unsqueeze(1).expand(-1, tgt_len)
    )

    p_choose = (
        torch.arange(src_len, device=key_eos.device)
        .unsqueeze(0)
        .unsqueeze(1)
        .expand(bsz, tgt_len, -1)
    ) == monotonic_step.unsqueeze(2)

    # assert p_choose.sum(-1).eq(1).all(), f"{p_choose.sum(-1)}"

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
