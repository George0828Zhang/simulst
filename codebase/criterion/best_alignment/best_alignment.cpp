#include <torch/extension.h>

#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
best_alignment_op(const torch::Tensor &log_probs, const torch::Tensor &targets,
                  at::IntArrayRef input_lengths, at::IntArrayRef target_lengths,
                  int64_t BLANK, bool zero_infinity);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
best_alignment(const torch::Tensor &log_probs, const torch::Tensor &targets,
               const torch::Tensor &input_lengths,
               const torch::Tensor &target_lengths, int64_t BLANK,
               bool zero_infinity) {
  torch::Tensor ilc =
      input_lengths.to(at::Device(at::kCPU), at::kLong).contiguous();
  torch::Tensor tlc =
      target_lengths.to(at::Device(at::kCPU), at::kLong).contiguous();
  at::IntArrayRef il(ilc.data_ptr<int64_t>(), ilc.numel());
  at::IntArrayRef tl(tlc.data_ptr<int64_t>(), tlc.numel());

  auto res =
      best_alignment_op(log_probs, targets.to(log_probs.device(), at::kLong),
                        il, tl, BLANK, zero_infinity);

  return res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("best_alignment", &best_alignment, "get best alignments for ctc");
}