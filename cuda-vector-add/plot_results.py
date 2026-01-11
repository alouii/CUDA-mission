import matplotlib.pyplot as plt
import pandas as pd

# Read CSV
df = pd.read_csv("benchmark_results.csv")

# Plot
plt.figure(figsize=(6,4))
plt.bar(df['Kernel'], df['AvgTimeMs'], color=['skyblue', 'orange'])
plt.ylabel("Average Time (ms)")
plt.title("CUDA Kernel vs CUDA Graph Performance")
for i, v in enumerate(df['AvgTimeMs']):
    plt.text(i, v + 0.05, f"{v:.2f} ms", ha='center')
plt.tight_layout()
plt.savefig("benchmark_plot.png", dpi=150)
plt.show()
