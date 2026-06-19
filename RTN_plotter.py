import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Load CSV ────────────────────────────────────────────────────────────────

csv_path = "MILESTONE_I_tests\\data\\test_ALL_fresh_t_mcf_0.1_icc_6e-06_aper_0.0002\\traces_1_3_1_4_VT_3_6e-06_0.1_fresh.csv"

df = pd.read_csv(csv_path, header=None)

# ── Extract traces ──────────────────────────────────────────────────────────
# Col 0 -> row index
# Col 1 -> col index
# Col 2:end -> RTN samples

for i in range(len(df)):

    row_idx = int(df.iloc[i, 0])
    col_idx = int(df.iloc[i, 1])

    trace = np.asarray(
        df.iloc[i, 2:].values,
        dtype=np.float64
    ).flatten()

    # ── Plot ────────────────────────────────────────────────────────────────

    plt.figure(figsize=(14, 4))

    plt.plot(trace, linewidth=0.8)

    plt.title(f"RTN Trace - Transistor ({row_idx}, {col_idx})")

    plt.xlabel("Sample")

    plt.ylabel("Current")

    plt.grid(True)

    plt.tight_layout()

    plt.show()