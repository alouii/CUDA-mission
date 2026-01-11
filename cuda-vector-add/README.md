# CUDA Vector Add with CUDA Graphs & Nsight Profiling

High-performance CUDA project demonstrating:
- CUDA Streams & async memory transfers
- CUDA Graph capture & replay
- Kernel benchmarking with CUDA events
- Nsight Systems & Nsight Compute profiling

## Features
- Pinned host memory
- Reduced kernel launch overhead via CUDA Graphs
- Reproducible benchmarks
- CMake-based build system

## Build
```bash
mkdir build && cd build
cmake ..
make
