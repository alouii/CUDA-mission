#!/bin/bash
set -e

# ------------------------- Step 1: Compile CUDA program -------------------------
echo "Compiling CUDA program..."
nvcc vector_add_graph_events.cu -o vector_add_graph

# ------------------------- Step 2: Run benchmark -------------------------
echo "Running CUDA benchmark..."
./vector_add_graph

# ------------------------- Step 3: Generate plot -------------------------
echo "Generating benchmark plot..."
python3 plot_results.py

# ------------------------- Step 4: Open plot (optional) -------------------------
if command -v xdg-open &> /dev/null
then
    echo "Opening benchmark_plot.png..."
    xdg-open benchmark_plot.png
fi

echo "Benchmark completed successfully!"
