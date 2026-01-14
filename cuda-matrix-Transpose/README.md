CUDA Matrix Transpose (Memory Coalescing & Shared Memory)

This example demonstrates matrix transposition on the GPU using CUDA, focusing on memory access patterns, shared memory tiling, and bank conflict avoidance.

Matrix transpose is a classic CUDA benchmark because it is memory-bound and clearly shows the performance impact of proper memory usage.

✨ What This Example Shows

This project implements three versions of matrix transpose:

Naive transpose

Global memory only

Poor memory coalescing

Tiled transpose (shared memory)

Coalesced global memory accesses

Shared memory reuse

Bank-conflict-free tiled transpose

Padding to eliminate shared memory bank conflicts

Each version is benchmarked using CUDA Events.

📐 Problem Definition

Given an input matrix A (height × width), compute its transpose:

B(x,y)=A(y,x)
B(x,y)=A(y,x)
📁 Project Structure
.
├── matrixTranspose.cu
└── README.md

🧠 Kernel Overview
1.Naive Transpose

Each thread reads one element and writes one element

Writes are non-coalesced

Serves as a baseline

2. Tiled Transpose (Shared Memory)

Matrix is divided into TILE_DIM × TILE_DIM tiles

Tiles are loaded into shared memory

Reads and writes are coalesced

3.Conflict-Free Tiled Transpose

Shared memory tile is padded:

__shared__ float tile[TILE_DIM][TILE_DIM + 1];


Prevents shared memory bank conflicts

Typically slightly faster than the tiled version

⏱ Benchmarking

Timing is done using CUDA Events

Each kernel is executed multiple times

Average execution time per iteration is reported

Example output:

Naive transpose        : 1.85 ms
Tiled transpose        : 0.32 ms
No-bank-conflict tiled : 0.29 ms
Speedup (naive → tiled): 5.78x

🛠 Build Instructions
Requirements

CUDA Toolkit ≥ 11

NVIDIA GPU

nvcc

Compile
nvcc -O3 matrixTranspose.cu -o matrixTranspose

▶️ Run
./matrixTranspose

🧪 Correctness Check

After execution, a small portion of the transposed matrix is copied back to the host and printed to verify correctness.

⚠️ Notes

This example focuses on memory optimization, not arithmetic

Performance gains depend on:

GPU architecture

Matrix size

Memory bandwidth

The bank-conflict-free version is most beneficial on older architectures, but still good practice today

🚀 Possible Extensions

Add CUDA Graph execution

Add throughput calculation (GB/s)

Compare against cuBLAS transpose

Add half / FP16 version

Visualize performance with plots

🎯 Who This Is For

CUDA learners

GPU performance engineers

Anyone preparing for CUDA or HPC interviews

Developers working on memory-bound kernels

📄 License

MIT — free to use for learning, research, or projects.
