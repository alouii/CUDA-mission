#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void naive_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N,
    int D)
{
    int i = blockIdx.x;
    int d = threadIdx.x;
    if (i >= N || d >= D) return;

    float maxv = -CUDART_INF_F;

    for (int j = 0; j < N; ++j) {
        float dot = 0.f;
        for (int k = 0; k < D; ++k)
            dot += Q[i*D+k] * K[j*D+k];
        maxv = fmaxf(maxv, dot);
    }

    float sum = 0.f;
    float out = 0.f;

    for (int j = 0; j < N; ++j) {
        float dot = 0.f;
        for (int k = 0; k < D; ++k)
            dot += Q[i*D+k] * K[j*D+k];

        float p = expf(dot - maxv);
        sum += p;
        out += p * V[j*D + d];
    }

    O[i*D + d] = out / sum;
}
