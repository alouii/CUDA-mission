# CUDA Vector Squaring Benchmark

This project demonstrates **element-wise squaring on GPU** using **CUDA** and compares:

- Normal kernel launches  
- CUDA Graph kernel-only execution  
- End-to-end runtime including Host ↔ Device memory copies  

It also prints the first 10 elements of arrays `A` and `C` and generates a **benchmark CSV and bar chart** automatically.

---

## Features

- Simple vector squaring kernel (`C[i] = A[i] * A[i]`)  
- Kernel-only and CUDA Graph execution  
- End-to-end timing including memory transfers  
- CUDA Events fallback profiling  
- CSV output for easy analysis  
- Automatic bar chart using Python/Matplotlib  
- First 10 elements of arrays printed for quick verification  
- Standalone example for easy reuse  

---

## Benchmark Results

| Benchmark                  | Avg Time (ms) | Speedup (vs KernelOnly) |
|----------------------------|---------------|------------------------|
| Kernel-only                | 0.33          | 1x                     |
| CUDA Graph kernel-only      | 0.30          | 1.10x                  |
| End-to-end (with memcpy)    | 4.44          | 0.07x                  |

![Benchmark Plot](benchmark_square_plot.png)

> CUDA Graph slightly accelerates kernel-only execution. End-to-end performance is dominated by memory transfers.

---

## Run Benchmark

You can compile, run, and plot the CUDA vector squaring benchmark as follows:

```bash
### 1.Compile CUDA program
nvcc vector_square_graph_events.cu -o vector_square_graph

### 2.Run benchmark
./vector_square_graph

### 3.Generate plot
python3 plot_results.py
```
# Example Output
```bash
A = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]
C = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, ...]
Kernel-only avg time: 0.33 ms
CUDA Graph kernel-only avg time: 0.30 ms
End-to-end avg time: 4.44 ms
```