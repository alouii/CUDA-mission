# CUDA FlashAttention Benchmark

This repository contains a **research-grade CUDA implementation**
of FlashAttention (forward pass) and a naive attention baseline.

## Features

- Warp-level FlashAttention
- Numerically stable online softmax
- No QKᵀ materialization
- Naive baseline for comparison
- Roofline-friendly structure

## Build

```bash
mkdir build && cd build
cmake ..
make
```
Run
./flash_attention

Output

Maximum numerical error vs naive attention

Kernel completes successfully

Architecture

1 warp = 1 query row

Shared memory tiling for K,V

Online softmax with (m, l) statistics

Next Steps

Tensor Cores (WMMA / MMA)

Backward pass (training-ready)

Nsight Compute roofline plots

Author
