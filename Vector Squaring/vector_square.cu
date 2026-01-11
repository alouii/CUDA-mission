// ------------------------- Print function -------------------------
void printArray(const float* arr, const char* name, int N, int count=10) {
    std::cout << name << " = [";
    for (int i = 0; i < std::min(N, count); ++i) {
        std::cout << arr[i];
        if (i < std::min(N, count)-1) std::cout << ", ";
    }
    if (N > count) std::cout << ", ...";
    std::cout << "]\n";
}

// ------------------------- CUDA kernel for squaring -------------------------
__global__ void vectorSquare(const float* A, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] * A[idx];
}

int main() {
    const int N = 1 << 22; // 4M elements
    const size_t size = N * sizeof(float);
    const int iterations = 1000;

    // ------------------------- Host pinned memory -------------------------
    float *h_A, *h_C;