#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;
typedef at::Half fp16;
typedef float fp32;

//---------------
//
// CPU related flags
//
//---------------

// simd
#ifdef __AVX512F__  // This macro is defined if AVX-512 is supported
    #include <immintrin.h>
    #define SIMD_WIDTH 16
    #define LOAD(x) _mm512_load_ps(x)
    #define STORE(x, y) _mm512_store_ps(x, y)
    #define SET1(x) _mm512_set1_ps(x)
    #define MULTIPLY(x, y) _mm512_mul_ps(x, y)
    #define MULTADD(x, y, z) _mm512_fmadd_ps(x, y, z)
    // print out the SIMD width
    #pragma message("AVX-512 is supported")
#else
    // Fallback to AVX2 if AVX-512 is not supported
    #ifdef __AVX2__
        #include <immintrin.h>
        #define SIMD_WIDTH 8
        #define LOAD(x) _mm256_load_ps(x)
        #define STORE(x, y) _mm256_store_ps(x, y)
        #define SET1(x) _mm256_set1_ps(x)
        #define MULTIPLY(x, y) _mm256_mul_ps(x, y)
        #define MULTADD(x, y, z) _mm256_fmadd_ps(x, y, z)
        // print out the SIMD width
        #pragma message("AVX-512 is not supported")
    #else
        #if defined(__ARM_NEON) || defined(__ARM_NEON__)
            #include <arm_neon.h>
            #define SIMD_WIDTH 4  // NEON typically operates on 128-bit registers (4 floats)
            #define LOAD(x) vld1q_f32(x)
            #define STORE(x, y) vst1q_f32(x, y)
            #define SET1(x) vdupq_n_f32(x)
            #define MULTIPLY(x, y) vmulq_f32(x, y)
            #define MULTADD(x, y, z) vmlaq_f32(z, x, y)
            // Print out the SIMD width
            #pragma message("ARM NEON is supported")
        #else
            #pragma message("No SIMD is supported")
            #define SIMD_WIDTH 1
            #define LOAD(x) x
            #define STORE(x, y) x = y
            #define SET1(x) x
            #define MULTIPLY(x, y) x * y
            #define MULTADD(x, y, z) x * y + z
        #endif
    #endif
#endif

//---------------
//
// Optimized mm8 operations
//
//---------------

void cudac_mm8_one(unsigned long long N, unsigned long long M,
                   float *x,
                   uint8_t *w, unsigned long long w_stride,
                   float *y,
                   float *r,
                   float *o,
                   unsigned long long offset,
                   unsigned long long tokenlength);

void mm8_one_bf16(int64_t N, int64_t M, torch::Tensor &x, torch::Tensor &w, torch::Tensor &y, torch::Tensor &r, torch::Tensor &o, int64_t offset, int64_t tokenlength) {
    cudac_mm8_one(N, M, x.data_ptr<float>(), w.data_ptr<uint8_t>(), w.stride(0), y.data_ptr<float>(), r.data_ptr<float>(), o.data_ptr<float>(), offset, tokenlength);
}

//---------------
//
// Cuda forward ops
//
//---------------

void cuda_forward_bf16(int B, int T, int C, int H, float *state, bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *y);
void cuda_forward_fp16(int B, int T, int C, int H, float *state, fp16 *r, fp16 *k, fp16 *v, float *w, fp16 *u, fp16 *y);
void cuda_forward_fp32(int B, int T, int C, int H, float *state, fp32 *r, fp32 *k, fp32 *v, float *w, fp32 *u, fp32 *y);

void forward_bf16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward_bf16(B, T, C, H, state.data_ptr<float>(), r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), w.data_ptr<float>(), u.data_ptr<bf16>(), y.data_ptr<bf16>());
}
void forward_fp16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward_fp16(B, T, C, H, state.data_ptr<float>(), r.data_ptr<fp16>(), k.data_ptr<fp16>(), v.data_ptr<fp16>(), w.data_ptr<float>(), u.data_ptr<fp16>(), y.data_ptr<fp16>());
}
void forward_fp32(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward_fp32(B, T, C, H, state.data_ptr<float>(), r.data_ptr<fp32>(), k.data_ptr<fp32>(), v.data_ptr<fp32>(), w.data_ptr<float>(), u.data_ptr<fp32>(), y.data_ptr<fp32>());
}

//---------------
//
// Cuda backward ops
//
//---------------

void cuda_backward_bf16(int B, int T, int C, int H, float *state, bf16 *r, bf16 *k, bf16 *v, float *w, float *ww, bf16 *u, bf16 *gy, bf16 *gr, bf16 *gk, bf16 *gv, bf16 *gw, bf16 *gu);
void cuda_backward_fp16(int B, int T, int C, int H, float *state, fp16 *r, fp16 *k, fp16 *v, float *w, float *ww, fp16 *u, fp16 *gy, fp16 *gr, fp16 *gk, fp16 *gv, fp16 *gw, fp16 *gu);
void cuda_backward_fp32(int B, int T, int C, int H, float *state, fp32 *r, fp32 *k, fp32 *v, float *w, float *ww, fp32 *u, fp32 *gy, fp32 *gr, fp32 *gk, fp32 *gv, fp32 *gw, fp32 *gu);

void backward_bf16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &ww, torch::Tensor &u, torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &gw, torch::Tensor &gu) {
    cuda_backward_bf16(B, T, C, H, state.data_ptr<float>(), r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), w.data_ptr<float>(), ww.data_ptr<float>(), u.data_ptr<bf16>(), gy.data_ptr<bf16>(), gr.data_ptr<bf16>(), gk.data_ptr<bf16>(), gv.data_ptr<bf16>(), gw.data_ptr<bf16>(), gu.data_ptr<bf16>());
}
void backward_fp16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &ww, torch::Tensor &u, torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &gw, torch::Tensor &gu) {
    cuda_backward_fp16(B, T, C, H, state.data_ptr<float>(), r.data_ptr<fp16>(), k.data_ptr<fp16>(), v.data_ptr<fp16>(), w.data_ptr<float>(), ww.data_ptr<float>(), u.data_ptr<fp16>(), gy.data_ptr<fp16>(), gr.data_ptr<fp16>(), gk.data_ptr<fp16>(), gv.data_ptr<fp16>(), gw.data_ptr<fp16>(), gu.data_ptr<fp16>());
}
void backward_fp32(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &ww, torch::Tensor &u, torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &gw, torch::Tensor &gu) {
    cuda_backward_fp32(B, T, C, H, state.data_ptr<float>(), r.data_ptr<fp32>(), k.data_ptr<fp32>(), v.data_ptr<fp32>(), w.data_ptr<float>(), ww.data_ptr<float>(), u.data_ptr<fp32>(), gy.data_ptr<fp32>(), gr.data_ptr<fp32>(), gk.data_ptr<fp32>(), gv.data_ptr<fp32>(), gw.data_ptr<fp32>(), gu.data_ptr<fp32>());
}

//---------------
//
// Pytorch binding
//
//---------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_bf16", &forward_bf16, "rwkv5 forward_bf16");
    m.def("forward_fp16", &forward_fp16, "rwkv5 forward_fp16");
    m.def("forward_fp32", &forward_fp32, "rwkv5 forward_fp32");
    m.def("backward_bf16", &backward_bf16, "wkv5 backward bf16");
    m.def("backward_fp16", &backward_fp16, "wkv5 backward fp16");
    m.def("backward_fp32", &backward_fp32, "wkv5 backward fp32");
    // m.def("mm8_one", &mm8_one_bf16, "uint8 bf16 mm8_one");
}

TORCH_LIBRARY(wkv5, m) {
    m.def("forward_bf16", forward_bf16);
    m.def("forward_fp16", forward_fp16);
    m.def("forward_fp32", forward_fp32);
    m.def("backward_bf16", backward_bf16);
    m.def("backward_fp16", backward_fp16);
    m.def("backward_fp32", backward_fp32);
    // m.def("mm8_one", mm8_one_bf16);
}