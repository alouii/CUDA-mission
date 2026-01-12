#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(call)                                   \
do {                                                       \
    cudaError_t err = call;                                \
    if (err != cudaSuccess) {                              \
        std::cerr << "CUDA error at " << __FILE__ << ":"   \
                  << __LINE__ << " -> "                    \
                  << cudaGetErrorString(err) << std::endl;\
        exit(EXIT_FAILURE);                                \
    }                                                      \
} while (0)

// =======================================
// Tensor abstraction (device-side)
// =======================================
struct Tensor {
    float* data;
    int rows;
    int cols;

    __device__ int size() const {
        return rows * cols;
    }

    __device__ float get(int row, int col) const {
        return data[row * cols + col];
    }

    __device__ void set(int row, int col, float value) {
        data[row * cols + col] = value;
    }
};

// =======================================
// LayerNorm kernel (one thread per row)
// =======================================
__global__ void layerNorm(Tensor* A) {
    extern __shared__ float shm[];

    float* mean     = shm;
    float* variance = shm + A->rows;
    float* invstd   = shm + 2 * A->rows;

    int row = threadIdx.x + blockIdx.x * blockDim.x;

    if (row >= A->rows) return;

    // ---- Mean ----
    float m = 0.0f;
    for (int j = 0; j < A->cols; j++) {
        m += A->get(row, j);
    }
    m /= A->cols;
    mean[row] = m;

    // ---- Variance ----
    float v = 0.0f;
    for (int j = 0; j < A->cols; j++) {
        float diff = A->get(row, j) - m;
        v += diff * diff;
    }
    v /= A->cols;
    variance[row] = v;
    invstd[row] = rsqrtf(v + 1e-5f);

    __syncthreads();

    // ---- Normalize ----
    for (int j = 0; j < A->cols; j++) {
        float x = A->get(row, j);
        float y = (x - mean[row]) * invstd[row];
        A->set(row, j, y);
    }
}

// =======================================
// Utility print
// =======================================
void printMatrix(const float* data, int rows, int cols, const char* title) {
    std::cout << title << ":\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << data[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// =======================================
// Main
// =======================================
int main() {
    const int rows = 1;
    const int cols = 3;
    const int tensorSize = rows * cols;
    const size_t bytes = tensorSize * sizeof(float);

    // Host data
    float h_data[tensorSize] = {
        5.0f, 1.5f, 2.0f
    };

    printMatrix(h_data, rows, cols, "Input");

    // Device memory
    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // Tensor struct
    Tensor h_tensor;
    h_tensor.data = d_data;
    h_tensor.rows = rows;
    h_tensor.cols = cols;

    Tensor* d_tensor;
    CHECK_CUDA(cudaMalloc(&d_tensor, sizeof(Tensor)));
    CHECK_CUDA(cudaMemcpy(d_tensor, &h_tensor, sizeof(Tensor),
                          cudaMemcpyHostToDevice));

    // Kernel launch
    int blockSize = 128;
    int gridSize = (rows + blockSize - 1) / blockSize;
    size_t sharedMemBytes = 3 * rows * sizeof(float);

    layerNorm<<<gridSize, blockSize, sharedMemBytes>>>(d_tensor);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back
    CHECK_CUDA(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    printMatrix(h_data, rows, cols, "Normalized Output");

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_tensor);

    return 0;
}
