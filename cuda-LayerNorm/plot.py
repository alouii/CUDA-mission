import matplotlib.pyplot as plt

labels = ['Normal', 'CUDA Graph']
times = [0.33, 0.12]  # replace with your real numbers

plt.bar(labels, times)
plt.ylabel('Time (ms)')
plt.title('LayerNorm Performance')
plt.tight_layout()
plt.savefig('benchmark_plot.png', dpi=120)
plt.show()
