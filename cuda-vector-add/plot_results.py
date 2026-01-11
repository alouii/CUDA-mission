import matplotlib.pyplot as plt
import pandas as pd

# Read CSV
df = pd.read_csv("benchmark_results.csv")

plt.figure(figsize=(8,5))
colors = ['skyblue', 'orange', 'green']
plt.bar(df['Benchmark'], df['AvgTimeMs'], color=colors)
plt.ylabel("Average Time (ms)")
plt.title("CUDA Kernel Benchmark: Normal vs CUDA Graph vs End-to-End")
for i, v in enumerate(df['AvgTimeMs']):
    plt.text(i, v + 0.05, f"{v:.2f} ms", ha='center')
plt.tight_layout()
plt.savefig("benchmark_plot.png", dpi=150)
plt.show()
