"""
test_REAL_TRACES.py
-------------------
Processes real RTN traces from 512 CSV files (1024 transistors, 32×32 array).

CSV format:
  - Each file contains 2 rows (one per transistor).
  - Col 0 : transistor row index
  - Col 1 : transistor column index
  - Cols 2…: 100 000 time-domain samples of I_D (the RTN trace)

Filename convention:
  traces_<row1>_<col1>_<row2>_<col2>_VT_3_6e-06_10_fresh.csv

For each trace the algorithms (KMeans, FHMM) are run and their
estimated parameters are stored. KMeans is only executed for
traces with SNR > 20 dB.
"""

import os
import re
import time
import traceback
import tempfile

import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from kmeans_real import kmeans_test
from fhmm_real import fhmm_test


# ── helpers ───────────────────────────────────────────────────────────────────

def scalarize(x):
    """
    Convert lists/arrays/scalars into a float scalar.
    """

    if isinstance(x, (list, tuple, np.ndarray)):
        x = np.asarray(x).flatten()

        if len(x) == 0:
            return np.nan

        return float(x[0])

    return float(x)

def trace_to_tmpcsv(trace_values: np.ndarray, tmpdir: str) -> str:
    """Write a single 1-D trace to a temporary CSV and return the path."""
    fd, path = tempfile.mkstemp(suffix=".csv", dir=tmpdir)
    os.close(fd)
    np.savetxt(path, trace_values.reshape(-1, 1), delimiter=",")
    return path


def calculate_snr_db(trace_values: np.ndarray) -> float:
    """
    Estimate SNR in dB using mean/std approach:

        SNR = 20 * log10(|mean| / std)

    Parameters
    ----------
    trace_values : np.ndarray
        RTN trace samples

    Returns
    -------
    float
        SNR in dB
    """
    mean_signal = np.abs(np.mean(trace_values))
    noise_std = np.std(trace_values)

    if noise_std == 0:
        return np.inf

    return 20 * np.log10(mean_signal / noise_std)


# ── per-trace worker (runs in a separate process) ─────────────────────────────

def process_trace(args):
    """Run FHMM always and KMeans conditionally based on SNR."""

    trace_values, row, col, aii_0, aij_0, tolerance, max_runs, tmpdir = args

    # Write trace to a temporary file so the existing *_test() API is reused
    tmp_path = trace_to_tmpcsv(trace_values, tmpdir)

    try:

        # ── Compute SNR ───────────────────────────────────────────────────────
        snr_db = calculate_snr_db(trace_values-np.min(trace_values))
        #snr_db = calculate_snr_db(trace_values)

        # ── FHMM (always executed) ───────────────────────────────────────────
        t0 = time.time()
        res_fhmm = fhmm_test(
            tmp_path,
            aii_0,
            aij_0,
            tolerance,
            max_runs
        )
        time_fhmm = time.time() - t0

        # ── KMeans (only for high-SNR traces) ───────────────────────────────


        t0 = time.time()
        res_kmeans = kmeans_test(tmp_path,k_known=res_fhmm[3])
        time_kmeans = time.time() - t0

        kmeans_results = {
            "kmeans_shift": np.asarray(res_kmeans),
            "time_kmeans":  time_kmeans,
        }



        return {

            "row": row,
            "col": col,
            "snr_db": snr_db,

            # KMeans results
            **kmeans_results,

            # FHMM results
            "fhmm_shift": np.asarray(res_fhmm[0]),
            "fhmm_taue":  np.asarray(res_fhmm[1]),
            "fhmm_tauc":  np.asarray(res_fhmm[2]),
            "fhmm_ndef":  np.asarray(res_fhmm[3]),
            "time_fhmm":  time_fhmm,
        }

    finally:
        # clean up regardless of success / failure
        os.unlink(tmp_path)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── FHMM initialisation ──────────────────────────────────────────────────
    aii_0     = 0.9
    aij_0     = 1 - aii_0
    tolerance = 1e-1
    max_runs  = 100

    # ── Paths ────────────────────────────────────────────────────────────────
    root_dir    = "MILESTONE_I_tests\\data"
    output_dir  = Path("results\\real_traces")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "real_traces_results.csv"

    # Temporary directory for per-trace CSVs
    tmpdir = tempfile.mkdtemp(prefix="rtn_tmp_")

    # ── Collect all traces from all CSV files ────────────────────────────────
    pattern = re.compile(r"traces_(\d+)_(\d+)_(\d+)_(\d+)_")

    task_list = []

    for dirpath, _, filenames in os.walk(root_dir):

        for filename in sorted(filenames):

            if not filename.endswith(".csv"):
                continue

            csv_path = os.path.join(dirpath, filename)

            m = pattern.search(filename)

            if m is None:
                print(f"Skipping (no filename pattern): {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path, header=None)

            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                continue

            for i in range(len(df)):

                row_idx = int(df.iloc[i, 0])
                col_idx = int(df.iloc[i, 1])

                trace = df.iloc[i, 2:].values.astype(float)

                task_list.append((trace, row_idx, col_idx))

    print(f"Total traces to process: {len(task_list)}")

    # ── Launch in parallel ───────────────────────────────────────────────────
    fhmm_args = (aii_0, aij_0, tolerance, max_runs)

    rows_out = []

    n_workers = max(1, (os.cpu_count() or 4) - 2)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:

        futures = {
            executor.submit(
                process_trace,
                (trace, row, col, *fhmm_args, tmpdir)
            ): (row, col)

            for trace, row, col in task_list
        }

        n_total = len(futures)

        n_ok = 0
        n_error = 0

        for future in as_completed(futures):

            row, col = futures[future]

            try:

                r = future.result()

                rows_out.append(r)

                n_ok += 1

                kmeans_msg = (
                    f"{r['time_kmeans']:.2f}s"
                    if not np.isnan(r["time_kmeans"])
                    else "SKIPPED"
                )

                print(
                    f"✓ [{n_ok + n_error}/{n_total}] "
                    f"transistor ({row:2d},{col:2d}) "
                    f"| SNR:{r['snr_db']:.2f} dB "
                    f"| KMeans:{kmeans_msg} "
                    f"| FHMM:{r['time_fhmm']:.2f}s"
                )

            except Exception:

                n_error += 1

                print(
                    f"✗ [{n_ok + n_error}/{n_total}] "
                    f"Error on transistor ({row},{col}):\n"
                    + traceback.format_exc()
                )

    print(f"\nSummary: {n_ok} OK, {n_error} errors out of {n_total} total")

    # ── Save results ─────────────────────────────────────────────────────────
    if rows_out:

        df_out = pd.DataFrame(rows_out).sort_values(["row", "col"])

        df_out.to_csv(output_file, index=False)

        print(f"\nResults saved to {output_file}")

    else:
        print("No results to save.")

    # ── Clean up temp dir ────────────────────────────────────────────────────
    try:
        os.rmdir(tmpdir)

    except OSError:
        # non-empty if something went wrong; leave it for inspection
        pass