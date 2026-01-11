#include <iostream>
#include <cuda_runtime.h>
#include "benchmark.cuh"

#define CUDA_CHECK(call)                                     \
    do {                                                     \
        cudaError_t err = call;                              \
        if (err != cudaSuccess) {                            \
            std::cerr << "CUDA error: "                      \
                      << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":"           \
                      << __LINE__ << std::endl;              \
            exit(EXIT_FAILURE);                              \
        }                                                    \
    } while (0)

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main() {
    const int N = 1 << 22;  // ~4M elements
    const size_t size = N * sizeof(float);

    // Pinned host memory (faster async transfers)
    float *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocHost(&h_A, size));
    CUDA_CHECK(cudaMallocHost(&h_B, size));
    CUDA_CHECK(cudaMallocHost(&h_C, size));

    for (int i = 0; i < N; ++i) {
        h_A[i] = float(i);
        h_B[i] = float(2 * i);
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float ms = benchmarkKernel([&] {
        vectorAdd<<<blocks, threads, 0, stream>>>(d_A, d_B, d_C, N);
    });

    CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "Kernel time: " << ms << " ms\n";
    std::cout << "Throughput: "
              << (N / (ms * 1e6)) << " billion ops/sec\n";

    for (int i = 0; i < 5; ++i)
        std::cout << "C[" << i << "] = " << h_C[i] << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaStreamDestroy(stream);
    cudaDeviceReset();
}
