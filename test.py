import os
import re
import time
import traceback
import tempfile
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kmeans import kmeans_test
from fhmm import fhmm_test


# ════════════════════════════════════════════════════════════════════════
# Mode Selector
# ════════════════════════════════════════════════════════════════════════

MODE = "real"  # "synthetic" or "real"

#----Synthetic Mode Parameters----#
global_test_name = "trace_length_estimation_test"
mode = "trace_length"  # "n_events" or "variation" or "SNR" or "trace_length"
test_number_min=1
test_number_max=5

#----Real Mode Parameters----#

root_dir   = "MILESTONE_I_tests\\data"
output_dir = Path("results\\real_traces")



def asymmetric_errors(means, stds):
    lower = [min(m, s) for m, s in zip(means, stds)]
    upper = stds
    return np.array([lower, upper])


def scalarize(x):
    """Convierte listas/arrays/escalares en un float escalar."""
    if isinstance(x, (list, tuple, np.ndarray)):
        x = np.asarray(x).flatten()
        if len(x) == 0:
            return np.nan
        return float(x[0])
    return float(x)


# ════════════════════════════════════════════════════════════════════════
# SYNTHETIC TRACES MODE
# ════════════════════════════════════════════════════════════════════════


def process_csv_synthetic(args):
    csv_path, aii_0, aij_0, tolerance, max_runs = args

    from WTLP import WTLP_test  # import local: solo se necesita en modo sintético

    print(f"---STARTING WTLP --- {csv_path}")
    t0 = time.time()
    res_wtlp = WTLP_test(csv_path)
    time_wtlp = time.time() - t0

    print(f"---STARTING KMEANS--- {csv_path}")
    t0 = time.time()
    res_kmeans = kmeans_test(csv_path, real=False)
    time_kmeans = time.time() - t0

    print(f"---STARTING FHMM --- {csv_path}")
    t0 = time.time()
    res_fhmm = fhmm_test(csv_path, aii_0, aij_0, tolerance, max_runs, real=False)
    time_fhmm = time.time() - t0

    return {
        "csv_path":    csv_path,
        "res_wtlp":    tuple(float(v) for v in res_wtlp[:4]),
        "res_kmeans":  tuple(float(v) for v in res_kmeans[:4]),
        "res_fhmm":    tuple(float(v) for v in res_fhmm[:4]),
        "time_wtlp":   time_wtlp,
        "time_kmeans": time_kmeans,
        "time_fhmm":   time_fhmm,
    }


def run_synthetic(global_test_name,mode,test_number_min,test_number_max):
    # ── FHMM initialization ─────────────────────────────────────────────
    aii_0     = 0.9
    aij_0     = 1 - aii_0
    tolerance = 1e-1
    max_runs  = 100

    for test in range(test_number_min, test_number_max+1):
        test_name = f"{global_test_name}_d{test}"
        root_dir = f"data\\{test_name}"
        
        if mode == "n_events":
            pattern = r"events_(\d+)"
        elif mode == "variation":
            pattern = r"variation_([\d.]+)"
        elif mode == "trace_length":
            pattern = r"trace_length_([\d.]+)"
        else:
            pattern = r"SNR_(-?\d+)"

        task_list = [] 
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if not filename.endswith(".csv"):
                    continue
                csv_path = os.path.join(dirpath, filename)
                match = re.search(pattern, csv_path)
                if match is None:
                    print(f"Saltando (sin patrón): {csv_path}")
                    continue
                if mode == "n_events":
                    x_val = int(match.group(1))
                elif mode == "variation":
                    x_val = float(match.group(1))
                else:
                    x_val = match.group(1)
                task_list.append((csv_path, x_val))

        print(f"Total CSVs encontrados: {len(task_list)}")

        results_wtlp   = defaultdict(list)
        results_kmeans = defaultdict(list)
        results_fhmm   = defaultdict(list)
        results_times  = defaultdict(list)

        fhmm_args = (aii_0, aij_0, tolerance, max_runs)

        with ProcessPoolExecutor(max_workers=os.cpu_count() - 4) as executor:
            futures = {
                executor.submit(process_csv_synthetic, (csv_path, *fhmm_args)): (csv_path, x)
                for csv_path, x in task_list
            }

            n_total = len(futures)
            n_ok = 0
            n_error = 0

            for future in as_completed(futures):
                csv_path, x = futures[future]
                try:
                    r = future.result()
                    results_wtlp[x].append(r["res_wtlp"])
                    results_kmeans[x].append(r["res_kmeans"])
                    results_fhmm[x].append(r["res_fhmm"])
                    results_times[x].append((r["time_wtlp"], r["time_kmeans"], r["time_fhmm"]))
                    n_ok += 1
                    print(f"✓ [{n_ok+n_error}/{n_total}] {mode}_{x}: {os.path.basename(csv_path)} "
                          f"| WTLP:{r['time_wtlp']:.2f}s "
                          f"| KMeans:{r['time_kmeans']:.2f}s "
                          f"| FHMM:{r['time_fhmm']:.2f}s")
                except Exception:
                    n_error += 1
                    print(f"✗ [{n_ok+n_error}/{n_total}] Error en {csv_path}:")
                    print(traceback.format_exc())

            print(f"\nResumen: {n_ok} OK, {n_error} errores de {n_total} totales")

            procesados = {futures[future][0] for future in futures}
            for csv_path, x in task_list:
                if csv_path not in procesados:
                    print(f"⚠️ Nunca procesado: {csv_path}")

        for x in sorted(results_wtlp.keys()):
            results_wtlp[x]   = np.array(results_wtlp[x])
            results_kmeans[x] = np.array(results_kmeans[x])
            results_fhmm[x]   = np.array(results_fhmm[x])

        for x in sorted(results_wtlp.keys()):
            label = f"events_{x}" if mode == "n_events" else f"variation_{x}"
            print(f"\n=== {label} ===")
            print(f"  WTLP   -> {results_wtlp[x]}")
            print(f"  KMeans -> {results_kmeans[x]}")
            print(f"  FHMM   -> {results_fhmm[x]}")

        rows = []
        for x in sorted(results_wtlp.keys()):
            for i in range(len(results_wtlp[x])):
                shift_error_wtlp,   taue_error_wtlp,   tauc_error_wtlp,   n_def_error_wtlp   = results_wtlp[x][i][:4]
                shift_error_kmeans, taue_error_kmeans, tauc_error_kmeans, n_def_error_kmeans = results_kmeans[x][i][:4]
                shift_error_fhmm,   taue_error_fhmm,   tauc_error_fhmm,   n_def_error_fhmm   = results_fhmm[x][i][:4]
                time_wtlp, time_kmeans, time_fhmm = results_times[x][i]

                key = "events_x" if mode == "n_events" else "variation_x"
                rows.append({
                    key:                  x,
                    "run":                i + 1,
                    "shift_error_wtlp":   shift_error_wtlp,
                    "taue_error_wtlp":    taue_error_wtlp,
                    "tauc_error_wtlp":    tauc_error_wtlp,
                    "n_def_error_wtlp":   n_def_error_wtlp,
                    "time_wtlp":          time_wtlp,
                    "shift_error_kmeans": shift_error_kmeans,
                    "taue_error_kmeans":  taue_error_kmeans,
                    "tauc_error_kmeans":  tauc_error_kmeans,
                    "n_def_error_kmeans": n_def_error_kmeans,
                    "time_kmeans":        time_kmeans,
                    "shift_error_fhmm":   shift_error_fhmm,
                    "taue_error_fhmm":    taue_error_fhmm,
                    "tauc_error_fhmm":    tauc_error_fhmm,
                    "n_def_error_fhmm":   n_def_error_fhmm,
                    "time_fhmm":          time_fhmm,
                })

        df = pd.DataFrame(rows)
        output_file_path = Path(f"results\\{global_test_name}\\{test_name}.cvs")
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file_path, index=False)
        print("\nResultados guardados en resultados.csv")

    # ── Plotting ──────────────────────────────────────────────────────────
    methods = {
        "WTLP":   results_wtlp,
        "KMeans": results_kmeans,
        "FHMM":   results_fhmm,
    }

    x_values = sorted(results_wtlp.keys())

    IDX_TAUE = 1
    IDX_TAUC = 2

    taue_max = max(
        max(results[x][:, IDX_TAUE].mean() + results[x][:, IDX_TAUE].std() for x in x_values)
        for results in methods.values()
    )
    tauc_max = max(
        max(results[x][:, IDX_TAUC].mean() + results[x][:, IDX_TAUC].std() for x in x_values)
        for results in methods.values()
    )
    tau_max = max(tauc_max, taue_max)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for col, (method_name, results) in enumerate(methods.items()):
        taue_means = [results[x][:, IDX_TAUE].mean() for x in x_values]
        taue_stds  = [results[x][:, IDX_TAUE].std()  for x in x_values]
        tauc_means = [results[x][:, IDX_TAUC].mean() for x in x_values]
        tauc_stds  = [results[x][:, IDX_TAUC].std()  for x in x_values]

        ax = axes[0, col]
        ax.errorbar(x_values, taue_means, yerr=asymmetric_errors(taue_means, taue_stds),
                    fmt='o-', capsize=5, capthick=1.5, elinewidth=1.5)
        ax.set_title(method_name)
        ax.tick_params(labelbottom=False)
        ax.grid(True, linestyle="--", alpha=0.5)
        if col == 0:
            ax.set_ylabel("taue_error")
        else:
            ax.tick_params(labelleft=False)

        ax = axes[1, col]
        ax.errorbar(x_values, tauc_means, yerr=asymmetric_errors(tauc_means, tauc_stds),
                    fmt='s-', capsize=5, capthick=1.5, elinewidth=1.5, color="tomato")
        ax.set_xlabel("n_events" if mode == "n_events" else "variation")
        ax.set_xticks(x_values)
        ax.grid(True, linestyle="--", alpha=0.5)
        if col == 0:
            ax.set_ylabel("tauc_error")
        else:
            ax.tick_params(labelleft=False)

    for col in range(3):
        axes[0, col].set_ylim(-0.1, tau_max * 1.1)
        axes[1, col].set_ylim(-0.1, tau_max * 1.1)

    fig.suptitle("n_events vs error (media ± 1σ)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("resultados_plot.png", dpi=150)
    plt.show()


# ════════════════════════════════════════════════════════════════════════
# REAL MODE
# ════════════════════════════════════════════════════════════════════════


def trace_to_tmpcsv(trace_values: np.ndarray, tmpdir: str) -> str:
    """Escribe una traza 1-D en un CSV temporal y devuelve la ruta."""
    fd, path = tempfile.mkstemp(suffix=".csv", dir=tmpdir)
    os.close(fd)
    np.savetxt(path, trace_values.reshape(-1, 1), delimiter=",")
    return path


def calculate_snr_db(trace_values: np.ndarray) -> float:

    mean_signal = np.abs(np.mean(trace_values))
    noise_std = np.std(trace_values)

    if noise_std == 0:
        return np.inf

    return 20 * np.log10(mean_signal / noise_std)


def process_trace_real(args):
    
    trace_values, row, col, aii_0, aij_0, tolerance, max_runs, tmpdir = args

    tmp_path = trace_to_tmpcsv(trace_values, tmpdir)

    try:
        snr_db = calculate_snr_db(trace_values - np.min(trace_values))

        # ── FHMM (siempre se ejecuta) ────────────────────────────────────
        t0 = time.time()
        res_fhmm = fhmm_test(tmp_path, aii_0, aij_0, tolerance, max_runs, real=True)
        time_fhmm = time.time() - t0

        # ── KMeans (usa num_d estimado por FHMM) ─────────────────────────
        t0 = time.time()
        res_kmeans = kmeans_test(tmp_path, k_known=res_fhmm[3], real=True)
        time_kmeans = time.time() - t0

        kmeans_results = {
            "kmeans_shift": np.asarray(res_kmeans),
            "time_kmeans":  time_kmeans,
        }

        return {
            "row": row,
            "col": col,
            "snr_db": snr_db,

            **kmeans_results,

            "fhmm_shift": np.asarray(res_fhmm[0]),
            "fhmm_taue":  np.asarray(res_fhmm[1]),
            "fhmm_tauc":  np.asarray(res_fhmm[2]),
            "fhmm_ndef":  np.asarray(res_fhmm[3]),
            "time_fhmm":  time_fhmm,
        }

    finally:
        os.unlink(tmp_path)


def run_real(root_dir,output_dir):
    # ── FHMM initialization ─────────────────────────────────────────────
    aii_0     = 0.9
    aij_0     = 1 - aii_0
    tolerance = 1e-1
    max_runs  = 100

    # ── Paths ────────────────────────────────────────────────────────────

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "real_traces_results.csv"

    tmpdir = tempfile.mkdtemp(prefix="rtn_tmp_")

    # ── Recopilar todas las trazas de todos los CSV ──────────────────────
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

    fhmm_args = (aii_0, aij_0, tolerance, max_runs)

    rows_out = []
    n_workers = max(1, (os.cpu_count() or 4) - 2)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_trace_real, (trace, row, col, *fhmm_args, tmpdir)): (row, col)
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

    # ── Guardar resultados ────────────────────────────────────────────────
    if rows_out:
        df_out = pd.DataFrame(rows_out).sort_values(["row", "col"])
        df_out.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    else:
        print("No results to save.")

    # ── Limpiar directorio temporal ───────────────────────────────────────
    try:
        os.rmdir(tmpdir)
    except OSError:
        # no vacío si algo falló; se deja para inspección
        pass


# ════════════════════════════════════════════════════════════════════════
# Entrypoint
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if MODE == "synthetic":
        run_synthetic(global_test_name=global_test_name,mode=mode,test_number_min=test_number_min,test_number_max=test_number_max)
    elif MODE == "real":
        run_real(root_dir=root_dir,output_dir=output_dir)
    else:
        raise ValueError(f"Unknown mode: {MODE!r}. Use 'synthetic' or 'real'.")