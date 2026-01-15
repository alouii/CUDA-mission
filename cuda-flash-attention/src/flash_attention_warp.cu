#include <cuda_runtime.h>
#include <math_constants.h>

#define WARP_SIZE 32

__device__ __forceinline__ float warpReduceSum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1)
        x += __shfl_down_sync(0xffffffff, x, offset);
    return x;
}

__global__ void flash_attention_forward_warp(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    float* m,
    float* l,
    int N,
    int D,
    int Br,
    int Bc)
{
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;

    int qi = blockIdx.x * Br + warp_id;
    if (warp_id >= Br || qi >= N) return;

    extern __shared__ float shmem[];
    float* Ksh = shmem;
    float* Vsh = shmem + Bc * D;

    float mi = m[qi];
    float li = l[qi];

    for (int j = 0; j < N; j += Bc) {

        // Load K,V blocks
        for (int idx = threadIdx.x; idx < Bc * D; idx += blockDim.x) {
            int p = idx / D;
            int d = idx % D;
            Ksh[idx] = K[(j + p) * D + d];
            Vsh[idx] = V[(j + p) * D + d];
        }
        __syncthreads();

        float sij_max = -CUDART_INF_F;
        float sij[16];  // Bc <= 16

        for (int p = 0; p < Bc; ++p) {
            float prod = Q[qi * D + lane] * Ksh[p * D + lane];
            float dot = warpReduceSum(prod);
            if (lane == 0) {
                sij[p] = dot;
                sij_max = fmaxf(sij_max, dot);
            }
        }

        sij_max = __shfl_sync(0xffffffff, sij_max, 0);

        float lij = 0.f;
        float pij[16];

        if (lane == 0) {
            for (int p = 0; p < Bc; ++p) {
                pij[p] = expf(sij[p] - sij_max);
                lij += pij[p];
            }
        }

        lij = __shfl_sync(0xffffffff, lij, 0);

        float m_new = fmaxf(mi, sij_max);
        float alpha = expf(mi - m_new);
        float beta  = expf(sij_max - m_new);

        float out = O[qi * D + lane] * li * alpha;
        for (int p = 0; p < Bc; ++p)
            out += beta * pij[p] * Vsh[p * D + lane];

        O[qi * D + lane] = out / (li * alpha + lij * beta);

        li = li * alpha + lij * beta;
        mi = m_new;

        __syncthreads();
    }

    if (lane == 0) {
        m[qi] = mi;
        l[qi] = li;
    }
}
