#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(call)                                     \
do {                                                         \
    cudaError_t err = call;                                  \
    if (err != cudaSuccess) {                                \
        std::cerr << "CUDA error: "                           \
                  << cudaGetErrorString(err) << std::endl;  \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while (0)

// =================================================
// Template Tensor
// =================================================
template<typename T>
struct Tensor {
    T* data;
    int rows;
    int cols;

    __device__ __forceinline__ T get(int r, int c) const {
        return data[r * cols + c];
    }

    __device__ __forceinline__ void set(int r, int c, T v) {
        data[r * cols + c] = v;
    }
};

// =================================================
// Optimized LayerNorm
// One block per row
// =================================================
__global__ void layerNormOptimized(Tensor<float> A) {
    extern __shared__ float shm[];

    float* s_sum = shm;
    float* s_sq  = shm + blockDim.x;

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.f;
    float sq  = 0.f;

    for (int j = tid; j < A.cols; j += blockDim.x) {
        float x = A.get(row, j);
        sum += x;
        sq  += x * x;
    }

    s_sum[tid] = sum;
    s_sq[tid]  = sq;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq[tid]  += s_sq[tid + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / A.cols;
    float var  = s_sq[0] / A.cols - mean * mean;
    float invstd = rsqrtf(var + 1e-5f);

    for (int j = tid; j < A.cols; j += blockDim.x) {
        float x = A.get(row, j);
        A.set(row, j, (x - mean) * invstd);
    }
}

// =================================================
// Benchmark helpers
// =================================================
float benchmarkKernel(Tensor<float>& d_tensor,
                      int iterations,
                      bool useGraph) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    dim3 block(256);
    dim3 grid(d_tensor.rows);
    size_t shm = 2 * block.x * sizeof(float);

    if (useGraph) {
        CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        layerNormOptimized<<<grid, block, shm, stream>>>(d_tensor);
        CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
        CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    }

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        if (useGraph)
            cudaGraphLaunch(graphExec, stream);
        else
            layerNormOptimized<<<grid, block, shm, stream>>>(d_tensor);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    return ms / iterations;
}

// =================================================
// Main
// =================================================
int main() {
    const int rows = 1024;
    const int cols = 1024;
    const int N = rows * cols;
    const int iters = 100;

    float* h_data = new float[N];
    for (int i = 0; i < N; i++)
        h_data[i] = static_cast<float>(i % 100);

    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(float),
                          cudaMemcpyHostToDevice));

    Tensor<float> d_tensor{d_data, rows, cols};

    float normal_ms = benchmarkKernel(d_tensor, iters, false);
    float graph_ms  = benchmarkKernel(d_tensor, iters, true);

    std::cout << "Normal kernel: " << normal_ms << " ms\n";
    std::cout << "CUDA Graph   : " << graph_ms  << " ms\n";
    std::cout << "Speedup     : " << normal_ms / graph_ms << "x\n";

    cudaFree(d_data);
    delete[] h_data;
}
