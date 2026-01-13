#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error: "                            \
                  << cudaGetErrorString(err) << std::endl;   \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

// =====================================================
// Naive Vector Sum
// =====================================================
__global__ void vectorSumNaive(const float* A,
                               const float* B,
                               float* C,
                               int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

// =====================================================
// Vectorized Load (float4)
// =====================================================
__global__ void vectorSumFloat4(const float4* A,
                                const float4* B,
                                float4* C,
                                int N4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N4) {
        float4 a = A[idx];
        float4 b = B[idx];
        C[idx] = make_float4(
            a.x + b.x,
            a.y + b.y,
            a.z + b.z,
            a.w + b.w
        );
    }
}

// =====================================================
// Grid-stride Loop (Scalable)
// =====================================================
__global__ void vectorSumGridStride(const float* A,
                                    const float* B,
                                    float* C,
                                    int N) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x) {
        C[idx] = A[idx] + B[idx];
    }
}

// =====================================================
//  Benchmark helper (CUDA Events)
// =====================================================
float benchmarkKernel(const float* d_A,
                      const float* d_B,
                      float* d_C,
                      int N,
                      int kernelType,
                      int iters) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        if (kernelType == 0)
            vectorSumNaive<<<grid, block>>>(d_A, d_B, d_C, N);
        else if (kernelType == 1)
            vectorSumGridStride<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms / iters;
}

// =====================================================
// Main
// =====================================================
int main() {
    const int N = 1 << 24;
    const int iters = 100;
    size_t bytes = N * sizeof(float);

    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    float naive_ms = benchmarkKernel(d_A, d_B, d_C, N, 0, iters);
    float grid_ms  = benchmarkKernel(d_A, d_B, d_C, N, 1, iters);

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    std::cout << "Naive kernel      : " << naive_ms << " ms\n";
    std::cout << "Grid-stride kernel: " << grid_ms  << " ms\n";
    std::cout << "Speedup           : " << naive_ms / grid_ms << "x\n";

    std::cout << "\nSample output:\n";
    for (int i = 0; i < 5; i++)
        std::cout << "C[" << i << "] = " << h_C[i] << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}
