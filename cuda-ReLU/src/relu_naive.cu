#include <cuda_runtime.h>

__global__ void relu_naive(const float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        y[i] = x[i] > 0.f ? x[i] : 0.f;
}
