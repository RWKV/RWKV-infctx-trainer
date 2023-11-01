#include <torch/extension.h>


torch::Tensor discounted_cumsum_left_cuda(torch::Tensor x, torch::Tensor gamma);
torch::Tensor discounted_cumsum_right_cuda(torch::Tensor x, torch::Tensor gamma);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("discounted_cumsum_left_cuda", &discounted_cumsum_left_cuda, "Discounted Cumulative Sums CUDA (Left)");
    m.def("discounted_cumsum_right_cuda", &discounted_cumsum_right_cuda, "Discounted Cumulative Sums CUDA (Right)");
}