# CUDA ReLU Benchmark

This repository benchmarks two CUDA implementations of ReLU:

- Naive scalar kernel
- Vectorized memory-optimized kernel

## Motivation

ReLU is a **memory-bound kernel** and serves as a perfect
contrast to compute-heavy kernels like FlashAttention.

## Build

```bash
mkdir build && cd build
cmake ..
make

Run

mkdir -p data plots
./cuda_relu

Performance

The vectorized kernel achieves higher memory throughput
by using coalesced float4 loads and stores.
Key Concepts

    Memory-bound kernels

    Vectorized global memory access

    CUDA occupancy vs bandwidth

    Roofline analysis

Author

Lassaad Aloui
