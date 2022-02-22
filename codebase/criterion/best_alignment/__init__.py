import torch
import torch.utils.cpp_extension
from pathlib import Path
import logging

module_path = Path(__file__).parent
build_path = module_path / "build"
build_path.mkdir(exist_ok=True)
extension = torch.utils.cpp_extension.load(
    "best_alignment_fn",
    sources=[
        module_path / "best_alignment.cpp",
        module_path / "best_alignment.cu",
    ],
    build_directory=build_path.as_posix()
)

logger = logging.getLogger(__name__)


def best_alignment(
    log_prob: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    as_labels: bool = False
):
    """Get best alignment (maximum probability sequence of ctc states)
        conditioned on log probabilities and target sequences.

    Args:
        log_prob (Tensor): (S, N, V) Log emission probabilities. Expected
            to be after log_softmax.
        targets (Tensor): (N, T) Target labels to force align.
        input_lengths (Tensor, optional): (N,) Length of each sources.
        target_lengths (Tensor, optional): (N,) Length of each targets.
        blank (int, optional): The blank index for ctc.
            Default: 0
        as_labels (bool, optional): Translate the state sequence to labels.
            Default: False

    Returns:
        output (Tensor): (N, S) The force alignment sequences.
            If as_labels=False, returns the state sequence, i.e. the values
                are in [0, 2T+1).
            If as_labels=True, returns the aligned label sequence, i.e. the
                values are in [0, V).
    """
    _, log_alpha, alignment = extension.best_alignment(
        log_prob, targets, input_lengths, target_lengths, blank, True
    )
    log_alpha = log_alpha.transpose(1, 2)  # (N, 2*T+1, S)
    alignment = alignment.transpose(1, 2)  # (N, 2*T+1, S)
    bsz, max_states, src_len = log_alpha.size()
    state_lengths = target_lengths * 2 + 1

    # mask out all but the last 2 positions at dim 1 (last 2 ctc final states)
    mask = torch.arange(max_states, device=log_alpha.device).view(1, max_states)
    # find last non -inf state at dim 1 for each src eos
    last_idx = (
        log_alpha
        .gather(  # gather at src eos
            2,
            (input_lengths - 1).view(bsz, 1, 1).expand(-1, max_states, -1))
        .squeeze(2)  # (N, 2*T+1)
        .eq(float("-inf"))
        .cumsum(1)
        .eq(1)
        .long()
        .argmax(1)  # first -inf index
        .add(-1)
        .remainder(state_lengths)  # % 2*T+1
    )
    # if (last_idx < state_lengths - 2).any():
    #     inds = (last_idx < (state_lengths - 2)).nonzero().squeeze(-1).cpu().numpy()
    #     logger.warning(
    #         f"examples ids={inds} cannot be fully aligned, probably"
    #         " due to label duplications in target. returning partial alignment instead.")
    last_idx = torch.min(last_idx, state_lengths - 2)
    mask = (mask < last_idx.view(bsz, 1)) | (
        mask >= state_lengths.view(bsz, 1))
    log_alpha = log_alpha.masked_fill(mask.unsqueeze(2), float("-inf"))

    # (N, 1, S)
    # all but the last position at dim 1 will be overridden below
    path_decode = log_alpha.argmax(1, keepdim=True)

    for t in range(src_len - 1, 0, -1):  # s - 1 ~ 1
        # only operate on 'msk' examples
        msk = t < input_lengths
        prev = path_decode[msk, :, t]   # b, 1
        path = alignment[msk, :, t]     # b, 2*T+1

        path_decode[msk, :, t - 1] = path.gather(1, prev)  # b, 1

    # (N, S)
    states = path_decode.squeeze(1)
    if as_labels:
        labels = torch.where(
            states % 2 == 1,
            targets.gather(1, states.div(2, rounding_mode='floor')),
            blank
        )
        return labels
    else:
        return states
