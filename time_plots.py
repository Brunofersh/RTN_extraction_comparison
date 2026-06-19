import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# ── CSVs de los 3 tests ───────────────────────────────────────────
csv_files = [
    "Results_test_taue_estimation.csv",
    "Results_n_defect_estimation.csv",
    "Results_noise_robustness_estimation.csv",
]
test_labels = ["Test #1", "Test #2", "Test #3"]

methods = {
    "WTLP":   "time_wtlp",
    "KMeans": "time_kmeans",
    "FHMM":   "time_fhmm",
}

# ── Cargar datos ──────────────────────────────────────────────────
# data[method] = lista de 3 arrays (uno por test)
data = {m: [] for m in methods}

for csv in csv_files:
    df = pd.read_csv(csv)
    for method, col in methods.items():
        data[method].append(df[col].dropna().values)

# ── Plot ──────────────────────────────────────────────────────────
plt.style.use(['science', 'grid', 'nature', 'no-latex'])
fig, ax = plt.subplots(figsize=(6, 4))

tx = 12
x_pos = np.arange(len(test_labels))          # [0, 1, 2]
n_methods = len(methods)
width = 0.2                                   # separación entre métodos
offsets = np.linspace(-(n_methods-1)/2, (n_methods-1)/2, n_methods) * width

colors  = ["#1f77b4", "#2ca02c", "#d62728"]  # azul, verde, rojo
markers = ["o", "v", "s"]

for idx, (method, col) in enumerate(methods.items()):
    means = [np.mean(d) for d in data[method]]
    mins  = [np.min(d)  for d in data[method]]
    maxs  = [np.max(d)  for d in data[method]]

    xs = x_pos + offsets[idx]

    # Bigotes min-max
    yerr = np.array([
        [m - lo for m, lo in zip(means, mins)],   # lower
        [hi - m for m, hi in zip(means, maxs)],   # upper
    ])

    ax.errorbar(
        xs, means,
        yerr=yerr,
        fmt=markers[idx],
        color=colors[idx],
        label=method,
        capsize=4, capthick=1.2, elinewidth=1,
        markersize=5,
    )

ax.set_yscale("log")
ax.set_xticks(x_pos)
ax.set_xticklabels(test_labels, fontsize=tx)
ax.set_ylabel("Time [s]", fontsize=tx, fontweight='bold')
ax.tick_params(axis='y', labelsize=tx - 2)
ax.legend(fontsize=tx - 2)
ax.set_xlim(-0.5, len(test_labels) - 0.5)

plt.tight_layout()
plt.savefig("time_comparison.png", dpi=600)