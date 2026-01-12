# CUDA LayerNorm (Optimized)

This repository demonstrates an optimized CUDA implementation of **Layer Normalization** with:

- One block per row
- Warp-level reductions
- CUDA Events benchmarking
- CUDA Graph execution
- Performance comparison & plots

## Build

```bash
nvcc -O3 layernorm.cu -o layernorm
```

## Run
```bash
./layernorm
```
