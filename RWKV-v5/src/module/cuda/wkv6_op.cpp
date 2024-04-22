#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

typedef at::BFloat16 bf16;
typedef at::Half fp16;
typedef float fp32;

void cuda_forward_bf16(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, bf16 *w, bf16 *u, bf16 *y);
void cuda_forward_fp16(int B, int T, int C, int H, fp16 *r, fp16 *k, fp16 *v, fp16 *w, fp16 *u, fp16 *y);
void cuda_forward_fp32(int B, int T, int C, int H, fp32 *r, fp32 *k, fp32 *v, fp32 *w, fp32 *u, fp32 *y);

void forward_bf16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward_bf16(B, T, C, H, r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), w.data_ptr<bf16>(), u.data_ptr<bf16>(), y.data_ptr<bf16>());
}
void forward_fp16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward_fp16(B, T, C, H, r.data_ptr<fp16>(), k.data_ptr<fp16>(), v.data_ptr<fp16>(), w.data_ptr<fp16>(), u.data_ptr<fp16>(), y.data_ptr<fp16>());
}
void forward_fp32(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward_fp32(B, T, C, H, r.data_ptr<fp32>(), k.data_ptr<fp32>(), v.data_ptr<fp32>(), w.data_ptr<fp32>(), u.data_ptr<fp32>(), y.data_ptr<fp32>());
}

void cuda_backward_bf16(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, bf16 *w, bf16 *u, bf16 *gy, bf16 *gr, bf16 *gk, bf16 *gv, bf16 *gw, bf16 *gu);
void cuda_backward_fp16(int B, int T, int C, int H, fp16 *r, fp16 *k, fp16 *v, fp16 *w, fp16 *u, fp16 *gy, fp16 *gr, fp16 *gk, fp16 *gv, fp16 *gw, fp16 *gu);
void cuda_backward_fp32(int B, int T, int C, int H, fp32 *r, fp32 *k, fp32 *v, fp32 *w, fp32 *u, fp32 *gy, fp32 *gr, fp32 *gk, fp32 *gv, fp32 *gw, fp32 *gu);

void backward_bf16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &gw, torch::Tensor &gu) {
    cuda_backward_bf16(B, T, C, H, r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), w.data_ptr<bf16>(), u.data_ptr<bf16>(), gy.data_ptr<bf16>(), gr.data_ptr<bf16>(), gk.data_ptr<bf16>(), gv.data_ptr<bf16>(), gw.data_ptr<bf16>(), gu.data_ptr<bf16>());
}
void backward_fp16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &gw, torch::Tensor &gu) {
    cuda_backward_fp16(B, T, C, H, r.data_ptr<fp16>(), k.data_ptr<fp16>(), v.data_ptr<fp16>(), w.data_ptr<fp16>(), u.data_ptr<fp16>(), gy.data_ptr<fp16>(), gr.data_ptr<fp16>(), gk.data_ptr<fp16>(), gv.data_ptr<fp16>(), gw.data_ptr<fp16>(), gu.data_ptr<fp16>());
}
void backward_fp32(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &gw, torch::Tensor &gu) {
    cuda_backward_fp32(B, T, C, H, r.data_ptr<fp32>(), k.data_ptr<fp32>(), v.data_ptr<fp32>(), w.data_ptr<fp32>(), u.data_ptr<fp32>(), gy.data_ptr<fp32>(), gr.data_ptr<fp32>(), gk.data_ptr<fp32>(), gv.data_ptr<fp32>(), gw.data_ptr<fp32>(), gu.data_ptr<fp32>());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_bf16", &forward_bf16, "rwkv6 forward_bf16");
    m.def("forward_fp16", &forward_fp16, "rwkv6 forward_fp16");
    m.def("forward_fp32", &forward_fp32, "rwkv6 forward_fp32");
    m.def("backward_bf16", &backward_bf16, "wkv6 backward bf16");
    m.def("backward_fp16", &backward_fp16, "wkv6 backward fp16");
    m.def("backward_fp32", &backward_fp32, "wkv6 backward fp32");}

TORCH_LIBRARY(wkv6b, m) {
    m.def("forward_bf16", forward_bf16);
    m.def("forward_fp16", forward_fp16);
    m.def("forward_fp32", forward_fp32);
    m.def("backward_bf16", backward_bf16);
    m.def("backward_fp16", backward_fp16);
    m.def("backward_fp32", backward_fp32);
}
