import torch
import torch.utils.cpp_extension
from pathlib import Path

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


@torch.no_grad()
def best_alignment(
    log_prob: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
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

    Returns:
        output (Tensor): (N, S) The force alignment state sequences. The
            aligned target labels can be recovered using gather:

            >>> labels = torch.where(
            >>>     states % 2 == 1,
            >>>     targets.gather(1, states // 2),
            >>>     blank
            >>> )
    """
    _, log_alpha, alignment = extension.best_alignment(
        log_prob, targets, input_lengths, target_lengths, blank, True
    )
    log_alpha = log_alpha.transpose(1, 2)  # (N, 2*T+1, S)
    alignment = alignment.transpose(1, 2)  # (N, 2*T+1, S)
    bsz, max_states, src_len = log_alpha.size()
    state_lengths = target_lengths * 2 + 1

    mask = torch.arange(max_states, device=log_alpha.device).view(1, max_states)
    mask = (mask < (state_lengths - 2).view(bsz, 1)) | (
        mask >= state_lengths.view(bsz, 1))
    log_alpha = log_alpha.masked_fill(mask.unsqueeze(2), float("-inf"))

    # (N, 1, S)
    # only the values at source - 1 are kept, others overridden below
    path_decode = log_alpha.argmax(1, keepdim=True)

    for t in range(src_len - 1, 0, -1):  # s - 1 ~ 1
        # only operate on 'msk' examples
        msk = t < input_lengths
        prev = path_decode[msk, :, t]   # b, 1
        path = alignment[msk, :, t]     # b, 2*T+1

        path_decode[msk, :, t - 1] = path.gather(1, prev)  # b, 1

    return path_decode.squeeze(1)
