#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error: "                       \
                      << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":"            \
                      << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main() {
    const int N = 1 << 22;  // 4M elements
    const size_t size = N * sizeof(float);
    const int iterations = 1000;

    // Allocate pinned host memory
    float *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocHost(&h_A, size));
    CUDA_CHECK(cudaMallocHost(&h_B, size));
    CUDA_CHECK(cudaMallocHost(&h_C, size));

    for (int i = 0; i < N; ++i) {
        h_A[i] = float(i);
        h_B[i] = float(2 * i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ---------------- Warm-up ----------------
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));
    vectorAdd<<<(N+255)/256, 256, 0, stream>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ---------------- Normal kernel profiling with CUDA Events ----------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) {
        vectorAdd<<<(N+255)/256, 256, 0, stream>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float normalTimeMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&normalTimeMs, start, stop));
    std::cout << "Normal kernel avg time: " << (normalTimeMs / iterations) << " ms\n";

    // ---------------- CUDA Graph profiling with CUDA Events ----------------
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));
    vectorAdd<<<(N+255)/256, 256, 0, stream>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float graphTimeMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&graphTimeMs, start, stop));
    std::cout << "CUDA Graph avg time: " << (graphTimeMs / iterations) << " ms\n";

    // ---------------- Results ----------------
    std::cout << "C[0] = " << h_C[0] << "\n";
    std::cout << "Speedup: " << (normalTimeMs / graphTimeMs) << "x\n";

    // ---------------- Cleanup ----------------
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaStreamDestroy(stream);
    cudaDeviceReset();
}
