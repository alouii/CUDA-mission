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

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

// =====================================================
// Naive transpose (global memory only)
// =====================================================
__global__ void transposeNaive(const float* A, float* B,
                               int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        B[x * height + y] = A[y * width + x];
    }
}

// =====================================================
// Tiled transpose (shared memory)
// =====================================================
__global__ void transposeTiled(const float* A, float* B,
                               int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = A[y * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < height && y < width)
        B[y * height + x] = tile[threadIdx.x][threadIdx.y];
}

// =====================================================
// Conflict-free tiled transpose
// Padding avoids shared memory bank conflicts
// =====================================================
__global__ void transposeTiledNoBank(const float* A, float* B,
                                     int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = A[y * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < height && y < width)
        B[y * height + x] = tile[threadIdx.x][threadIdx.y];
}

// =====================================================
// Benchmark helper
// =====================================================
float benchmark(const float* d_A, float* d_B,
                int width, int height,
                int kernel, int iters) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM,
              (height + TILE_DIM - 1) / TILE_DIM);

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        if (kernel == 0)
            transposeNaive<<<grid, block>>>(d_A, d_B, width, height);
        else if (kernel == 1)
            transposeTiled<<<grid, block>>>(d_A, d_B, width, height);
        else
            transposeTiledNoBank<<<grid, block>>>(d_A, d_B, width, height);
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
    const int width = 2048;
    const int height = 2048;
    const int N = width * height;
    const int iters = 50;

    size_t bytes = N * sizeof(float);

    float* h_A = new float[N];
    float* h_B = new float[N];

    for (int i = 0; i < N; i++)
        h_A[i] = static_cast<float>(i);

    float *d_A, *d_B;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    float naive_ms = benchmark(d_A, d_B, width, height, 0, iters);
    float tiled_ms = benchmark(d_A, d_B, width, height, 1, iters);
    float nobank_ms = benchmark(d_A, d_B, width, height, 2, iters);

    CHECK_CUDA(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));

    std::cout << "Naive transpose        : " << naive_ms << " ms\n";
    std::cout << "Tiled transpose        : " << tiled_ms << " ms\n";
    std::cout << "No-bank-conflict tiled : " << nobank_ms << " ms\n";

    std::cout << "Speedup (naive → tiled): "
              << naive_ms / tiled_ms << "x\n";
    std::cout << "Speedup (tiled → nobank): "
              << tiled_ms / nobank_ms << "x\n";

    std::cout << "\nSample output:\n";
    for (int i = 0; i < 5; i++)
        std::cout << h_B[i] << " ";
    std::cout << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    delete[] h_A;
    delete[] h_B;
}
