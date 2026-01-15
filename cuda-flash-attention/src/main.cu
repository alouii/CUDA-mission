#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

void naive_attention(
    const float*, const float*, const float*, float*, int, int);

void flash_attention_forward_warp(
    const float*, const float*, const float*, float*, float*, float*,
    int, int, int, int);

float benchmark(void (*kernel)(...), dim3 grid, dim3 block, size_t shmem) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<grid, block, shmem>>>();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

int main() {
    const int N = 32;
    const int D = 32;
    const int Br = 4;
    const int Bc = 4;

    float *Q, *K, *V, *O_naive, *O_flash, *m, *l;
    cudaMallocManaged(&Q, N*D*sizeof(float));
    cudaMallocManaged(&K, N*D*sizeof(float));
    cudaMallocManaged(&V, N*D*sizeof(float));
    cudaMallocManaged(&O_naive, N*D*sizeof(float));
    cudaMallocManaged(&O_flash, N*D*sizeof(float));
    cudaMallocManaged(&m, N*sizeof(float));
    cudaMallocManaged(&l, N*sizeof(float));

    for (int i = 0; i < N; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.f;
        for (int d = 0; d < D; ++d) {
            Q[i*D+d] = 0.01f;
            K[i*D+d] = 0.02f;
            V[i*D+d] = 0.03f;
            O_naive[i*D+d] = 0.f;
            O_flash[i*D+d] = 0.f;
        }
    }

    naive_attention<<<N, D>>>(Q, K, V, O_naive, N, D);
    cudaDeviceSynchronize();

    dim3 block(Br * 32);
    dim3 grid((N + Br - 1) / Br);
    size_t shmem = 2 * Bc * D * sizeof(float);

    flash_attention_forward_warp<<<grid, block, shmem>>>(
        Q, K, V, O_flash, m, l, N, D, Br, Bc);
    cudaDeviceSynchronize();

    float err = 0.f;
    for (int i = 0; i < N*D; ++i)
        err = fmaxf(err, fabs(O_naive[i] - O_flash[i]));

    printf("Max error: %.6e\n", err);
    printf("Benchmark complete.\n");

    cudaFree(Q); cudaFree(K); cudaFree(V);
    cudaFree(O_naive); cudaFree(O_flash);
    cudaFree(m); cudaFree(l);
}
