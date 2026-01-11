import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("benchmark_square_results.csv")

plt.figure(figsize=(6,3.5))
colors = ['skyblue', 'orange', 'green']
plt.bar(df['Benchmark'], df['AvgTimeMs'], color=colors)
plt.ylabel("Average Time (ms)")
plt.title("CUDA Kernel Benchmark: Vector Squaring")
for i, v in enumerate(df['AvgTimeMs']):
    plt.text(i, v + 0.05, f"{v:.2f} ms", ha='center')
plt.tight_layout()
plt.savefig("benchmark_square_plot.png", dpi=100, bbox_inches='tight')
plt.show()
