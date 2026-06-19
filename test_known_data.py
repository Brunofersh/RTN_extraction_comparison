from WTLP import WTLP_test
from kmeans import kmeans_test
from fhmm import fhmm_test
import os
import traceback
import re
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def asymmetric_errors(means, stds):
    lower = [min(m, s) for m, s in zip(means, stds)]  # no puede bajar de 0
    upper = stds  # el bigote superior no tiene restricción
    return np.array([lower, upper])


# ── Worker: se ejecuta en un proceso separado ─────────────────────────────
def process_csv(args):
    csv_path, aii_0, aij_0, tolerance, max_runs = args

    print(f"---STARTING FHMM --- {csv_path}")
    t0 = time.time()
    res_fhmm = fhmm_test(csv_path, aii_0, aij_0, tolerance, max_runs)
    time_fhmm = time.time() - t0

    print(f"---STARTING WTLP --- {csv_path}")
    t0 = time.time()
    res_wtlp = WTLP_test(csv_path)
    time_wtlp = time.time() - t0

    print(f"---STARTING KMEANS--- {csv_path}")
    t0 = time.time()
    res_kmeans = kmeans_test(csv_path,len(res_fhmm[4]))
    time_kmeans = time.time() - t0

    #print(f"  FHMM result ({os.path.basename(csv_path)}): {res_fhmm}")  # <-- diagnóstico

    return {
        "csv_path":    csv_path,
        "res_wtlp":    tuple(float(v) for v in res_wtlp[:4]),
        "res_kmeans":  tuple(float(v) for v in res_kmeans[:4]),
        "res_fhmm":    tuple(float(v) for v in res_fhmm[:4]),
        "time_wtlp":   time_wtlp,
        "time_kmeans": time_kmeans,
        "time_fhmm":   time_fhmm,
    }


if __name__ == "__main__":

    #-------------FHMM INITIALIZATION--------------#
    aii_0     = 0.9
    aij_0     = 1 - aii_0
    std_dev   = 0.1
    tolerance = 1e-1
    max_runs  = 100
    #----------------------------------------------#
    for test in range(5,6):
        test_name=f"noise_robustness_test_d{test}"

        known_data=True

        root_dir = f"data\\{test_name}"
        mode     = "SNR"  # "n_events" or "variation" or "SNR"

        # Después — captura decimales también (e.g. "0.1" -> "0.1")
        if(mode=="n_events"):
            pattern=r"events_(\d+)"
        elif(mode=="variation"):
            pattern=r"variation_([\d.]+)"
        else:
            pattern=r"SNR_(-?\d+)"
        #pattern = r"events_(\d+)" if mode == "n_events" else r"variation_([\d.]+)"

        # ── Recopilar todos los CSVs ───────────────────────────────────────────
        task_list = []  # lista de (csv_path, x)
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if not filename.endswith(".csv"):
                    continue
                csv_path = os.path.join(dirpath, filename)
                match = re.search(pattern, csv_path)
                #print(match)
                if match is None:
                    print(f"Saltando (sin patrón): {csv_path}")
                    continue
                #x_val = float(match.group(1)) if mode == "variation" else if mode=="n_events" int(match.group(1))
                if(mode=="n_events"):
                    x_val=int(match.group(1))
                elif(mode=="variation"):
                    x_val=float(match.group(1))
                else:
                    x_val=match.group(1)
                task_list.append((csv_path, x_val))

        print(f"Total CSVs encontrados: {len(task_list)}")

        # ── Diccionarios para guardar resultados ──────────────────────────────
        results_wtlp   = defaultdict(list)
        results_kmeans = defaultdict(list)
        results_fhmm   = defaultdict(list)
        results_times  = defaultdict(list)

        fhmm_args = (aii_0, aij_0, tolerance, max_runs)

        # ── Lanzar en paralelo ────────────────────────────────────────────────
        with ProcessPoolExecutor(max_workers=os.cpu_count() - 4) as executor:
            futures = {
                executor.submit(process_csv, (csv_path, *fhmm_args)): (csv_path, x)
                for csv_path, x in task_list
            }

            n_total   = len(futures)
            n_ok      = 0
            n_error   = 0

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
                except Exception as e:
                    n_error += 1
                    print(f"✗ [{n_ok+n_error}/{n_total}] Error en {csv_path}:")
                    print(traceback.format_exc())

            print(f"\nResumen: {n_ok} OK, {n_error} errores de {n_total} totales")

            # Verificar que no falte ninguno
            procesados = set()
            for future in futures:
                procesados.add(futures[future][0])  # csv_path

            for csv_path, x in task_list:
                if csv_path not in procesados:
                    print(f"⚠️ Nunca procesado: {csv_path}")

        # ── Convertir a numpy ─────────────────────────────────────────────────
        for x in sorted(results_wtlp.keys()):
            results_wtlp[x]   = np.array(results_wtlp[x])
            results_kmeans[x] = np.array(results_kmeans[x])
            results_fhmm[x]   = np.array(results_fhmm[x])

        # ── Imprimir resumen ──────────────────────────────────────────────────
        for x in sorted(results_wtlp.keys()):
            label = f"events_{x}" if mode == "n_events" else f"variation_{x}"
            print(f"\n=== {label} ===")
            print(f"  WTLP   -> {results_wtlp[x]}")
            print(f"  KMeans -> {results_kmeans[x]}")
            print(f"  FHMM   -> {results_fhmm[x]}")

        # ── Guardar CSV ───────────────────────────────────────────────────────
        rows = []
        for x in sorted(results_wtlp.keys()):
            for i in range(len(results_wtlp[x])):

                shift_error_wtlp,   taue_error_wtlp,   tauc_error_wtlp,   n_def_error_wtlp   = results_wtlp[x][i][:4]
                shift_error_kmeans, taue_error_kmeans,  tauc_error_kmeans, n_def_error_kmeans = results_kmeans[x][i][:4]
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
        if(known_data):
            df.to_csv(test_name+"_known_data.csv", index=False)
        else:
            df.to_csv(test_name+".csv", index=False)
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

        # --- Fila 0: taue ---
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

        # --- Fila 1: tauc ---
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