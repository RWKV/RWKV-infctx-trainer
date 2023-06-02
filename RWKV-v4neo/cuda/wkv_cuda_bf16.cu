#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#define MIN_VALUE (-1e38)
typedef at::BFloat16 bf16;

__global__ void kernel_forward(const int B, const int T, const int C,
                               const float *__restrict__ const _w, const bf16 *__restrict__ const _u, const bf16 *__restrict__ const _k, const bf16 *__restrict__ const _v,
                               const float *__restrict__ const last_state, bf16 *__restrict__ const _y, float *__restrict__ const new_state) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;
    const int state_offset = (_b * C + _c)*3;

    float u = float(_u[_c]);
    float w = _w[_c];
    const bf16 *__restrict__ const k = _k + _offset;
    const bf16 *__restrict__ const v = _v + _offset;
    bf16 *__restrict__ const y = _y + _offset;

    float p, q, o;
    if (last_state == NULL) {
        p = 0, q = 0, o = MIN_VALUE;
    } else {
        p = last_state[state_offset+0];
        q = last_state[state_offset+1];
        o = last_state[state_offset+2];
    }
    // p and q are running sums divided by exp(o) (to avoid overflows)
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const float kk = float(k[ii]);
        const float vv = float(v[ii]);

        float no = max(o, u + kk);
        float A = exp(o - no);
        float B = exp(u + kk - no);
        y[ii] = bf16((A * p + B * vv) / (A * q + B));

        no = max(w + o, kk);
        A = exp(w + o - no);
        B = exp(kk - no);
        p = A * p + B * vv;
        q = A * q + B;
        o = no;
    }
    if (new_state != NULL) {
        new_state[state_offset+0] = p;
        new_state[state_offset+1] = q;
        new_state[state_offset+2] = o;
    }
}

__global__ void kernel_backward(const int B, const int T, const int C,
                                const float *__restrict__ const _w, const bf16 *__restrict__ const _u, const bf16 *__restrict__ const _k, const bf16 *__restrict__ const _v, const float *__restrict__ const last_state, 
                                const bf16 *__restrict__ const _gy, const float *__restrict__ const gnew_state,
                                bf16 *__restrict__ const _gw, bf16 *__restrict__ const _gu, bf16 *__restrict__ const _gk, bf16 *__restrict__ const _gv, float *__restrict__ const glast_state) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;
    const int state_offset  = (_b * C + _c)*3;

    float u = float(_u[_c]);
    float w = _w[_c];
    const bf16 *__restrict__ const k = _k + _offset;
    const bf16 *__restrict__ const v = _v + _offset;
    const bf16 *__restrict__ const gy = _gy + _offset;

    bf16 *__restrict__ const gk = _gk + _offset;
    bf16 *__restrict__ const gv = _gv + _offset;

    float y[Tmax], z[Tmax], zexp[Tmax];

    float gw = 0, gu = 0;
    float dpdw = 0, dqdw = 0;
    float p, q, o;
    if (last_state == NULL) {
        p = 0, q = 0, o = MIN_VALUE;
    } else {
        p = last_state[state_offset+0];
        q = last_state[state_offset+1];
        o = last_state[state_offset+2];
    }
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const float kk = float(k[ii]);
        const float vv = float(v[ii]);
        const float gyy = float(gy[ii]);

        float no = max(o, kk + u);
        float A = exp(o - no);
        float B = exp(kk + u - no);

        float num = A * p + B * vv;
        float iden = 1 / (A * q + B);

        y[i] = num * iden;
        z[i] = iden;
        zexp[i] = kk + u - no;

        gw += gyy * (dpdw - dqdw * y[i]) * iden * A;
        gu += gyy * (vv - y[i]) * B * iden;

        no = max(w + o, kk);
        A = exp(w + o - no);
        B = exp(kk - no);
        dpdw = A * (p + dpdw);
        dqdw = A * (q + dqdw);
        p = A * p + B * vv;
        q = A * q + B;
        o = no;
    }

    float gp = 0, gq = 0, go = MIN_VALUE;
    if (gnew_state != NULL) {
        gp = gnew_state[state_offset+0];
        gq = gnew_state[state_offset+1];
        go = gnew_state[state_offset+2];
        if (gp == 0 && gq == 0) go = MIN_VALUE;
        gw += (gp * dpdw + gq * dqdw) * exp(o+go);
    }

    for (int i = T - 1; i >= 0; i--) {
        const int ii = i * C;
        const float kk = float(k[ii]);
        const float vv = float(v[ii]);
        const float gyy = float(gy[ii]);

        float A = gyy * z[i] * exp(zexp[i]);
        float B = exp(kk + go);
        gk[ii] = bf16(A * (vv - y[i]) + B * (gp * vv + gq));
        gv[ii] = bf16(A + B * gp);

        float no = max(w + go, zexp[i] - kk - u);
        A = exp(w + go - no);
        B = gyy * z[i] * exp(zexp[i] - kk - u - no);
        gp = A * gp + B;
        gq = A * gq - B * y[i];
        go = no;
    }

    // glast_state[2] is not the gradient w.r.t of last_state[2]
    // o (index 2) in last_state is just an exponent for p and q
    // so there are really only 2 elements to differentiate on
    // Similary go (glast_state index 2) is just an exponent for gp and gq
    if (glast_state != NULL) {
        glast_state[state_offset+0] = gp;
        glast_state[state_offset+1] = gq;
        glast_state[state_offset+2] = go;
    }

    // Multiply by w because the w -> -exp(w) preprocessing is halfway in the backwards pass, even though it's not in the forward pass
    const int _offsetBC = _b * C + _c;
    _gw[_offsetBC] += bf16(gw * _w[_c]);
    _gu[_offsetBC] += gu;
}

void cuda_forward(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, float *last_state, bf16 *y, float *new_state) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, last_state, y, new_state);
}

void cuda_backward(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, float *last_state, bf16 *gy, float *gnew_state, bf16 *gw, bf16 *gu, bf16 *gk, bf16 *gv, float *glast_state) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, last_state, gy, gnew_state, gw, gu, gk, gv, glast_state);
}
