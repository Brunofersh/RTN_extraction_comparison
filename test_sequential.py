from WTLP import WTLP_test
from kmeans import kmeans_test
from fhmm_mine import fhmm_test
import os
import re
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import time



def asymmetric_errors(means, stds):
    lower = [min(m, s) for m, s in zip(means, stds)]  # no puede bajar de 0
    upper = stds  # el bigote superior no tiene restricción
    return np.array([lower, upper])

#-------------FHMM INITIALIZATION--------------#
aii_0=0.9
aij_0=1-aii_0
std_dev=0.1
tolerance=1e-2
max_runs=1000
#----------------------------------------------#

# Diccionarios para guardar resultados por valor de x
results_wtlp  = defaultdict(list)
results_kmeans = defaultdict(list)
results_fhmm   = defaultdict(list)
results_times = defaultdict(list)

# Buscar todos los CSV recursivamente
root_dir = r"data\n_defect_estimation_test\variation_0.1"
mode="variation" # "n_events" or "variation"

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".csv"):
            csv_path = os.path.join(dirpath, filename)

            # Extraer x de "events_x" en la ruta
            if(mode=="n_events"):
                match = re.search(r"events_(\d+)", csv_path)
                if match is None:
                    print(f"No se encontró 'events_x' en: {csv_path}, saltando...")
                    continue
                    
                x = int(match.group(1))

                print(f"\n--- Procesando events_{x}: {csv_path} ---")
            elif(mode=="variation"):
                match = re.search(r"variation_(\d+)", csv_path)
                if match is None:
                    print(f"No se encontró 'events_x' en: {csv_path}, saltando...")
                    continue
                    
                x = int(match.group(1))

                print(f"\n--- Procesando events_{x}: {csv_path} ---")

                # Ejecutar funciones y guardar resultados
                print("---STARTING WTLP---")
                t0 = time.time()
                res_wtlp = WTLP_test(csv_path)
                time_wtlp = time.time() - t0
                

                print("---STARTING KMEANS---")
                t0 = time.time()
                res_kmeans = kmeans_test(csv_path)
                time_kmeans = time.time() - t0
                

                print("---STARTING FHMM---")
                t0 = time.time()
                res_fhmm = fhmm_test(csv_path, aii_0, aij_0, tolerance, max_runs)
                time_fhmm = time.time() - t0
                

                print(f"  Tiempos -> WTLP: {time_wtlp:.3f}s | KMeans: {time_kmeans:.3f}s | FHMM: {time_fhmm:.3f}s")

                results_wtlp[x].append(tuple(float(v) for v in res_wtlp[:4]))
                results_kmeans[x].append(tuple(float(v) for v in res_kmeans[:4]))
                results_fhmm[x].append(tuple(float(v) for v in res_fhmm[:4]))
                results_times[x].append((time_wtlp, time_kmeans, time_fhmm))

# Convertir listas a arrays de numpy para procesamiento
for x in sorted(results_wtlp.keys()):
    results_wtlp[x]   = np.array(results_wtlp[x])
    results_kmeans[x] = np.array(results_kmeans[x])
    results_fhmm[x]   = np.array(results_fhmm[x])

# Ejemplo de acceso a resultados
for x in sorted(results_wtlp.keys()):
    if(mode=="n_events"):
        print(f"\n=== events_{x} ===")
    elif(mode=="variation"):
        print(f"\n=== variation_{x} ===")
    print(f"  WTLP   -> {results_wtlp[x]}")
    print(f"  KMeans -> {results_kmeans[x]}")
    print(f"  FHMM   -> {results_fhmm[x]}")


rows = []

for x in sorted(results_wtlp.keys()):
    for i in range(len(results_wtlp[x])):
        
        shift_error_wtlp,  taue_error_wtlp,  tauc_error_wtlp,  n_def_error_wtlp  = results_wtlp[x][i][:4]
        shift_error_kmeans, taue_error_kmeans, tauc_error_kmeans, n_def_error_kmeans = results_kmeans[x][i][:4]
        shift_error_fhmm,  taue_error_fhmm,  tauc_error_fhmm,  n_def_error_fhmm  = results_fhmm[x][i][:4]
        time_wtlp, time_kmeans, time_fhmm = results_times[x][i]
        

        if(mode=="n_events"):

            rows.append({
                "events_x": x,
                "run":      i + 1,

                "shift_error_wtlp":   shift_error_wtlp,
                "taue_error_wtlp":    taue_error_wtlp,
                "tauc_error_wtlp":    tauc_error_wtlp,
                "n_def_error_wtlp":   n_def_error_wtlp,
                "time_wtlp":          time_wtlp,          # <---

                "shift_error_kmeans": shift_error_kmeans,
                "taue_error_kmeans":  taue_error_kmeans,
                "tauc_error_kmeans":  tauc_error_kmeans,
                "n_def_error_kmeans": n_def_error_kmeans,
                "time_kmeans":        time_kmeans,         # <---

                "shift_error_fhmm":   shift_error_fhmm,
                "taue_error_fhmm":    taue_error_fhmm,
                "tauc_error_fhmm":    tauc_error_fhmm,
                "n_def_error_fhmm":   n_def_error_fhmm,
                "time_fhmm":          time_fhmm,           # <---
            })
        elif(mode=="variation"):
            rows.append({
                "variation_x": x,
                "run":      i + 1,

                "shift_error_wtlp":   shift_error_wtlp,
                "taue_error_wtlp":    taue_error_wtlp,
                "tauc_error_wtlp":    tauc_error_wtlp,
                "n_def_error_wtlp":   n_def_error_wtlp,
                "time_wtlp":          time_wtlp,          # <---

                "shift_error_kmeans": shift_error_kmeans,
                "taue_error_kmeans":  taue_error_kmeans,
                "tauc_error_kmeans":  tauc_error_kmeans,
                "n_def_error_kmeans": n_def_error_kmeans,
                "time_kmeans":        time_kmeans,         # <---

                "shift_error_fhmm":   shift_error_fhmm,
                "taue_error_fhmm":    taue_error_fhmm,
                "tauc_error_fhmm":    tauc_error_fhmm,
                "n_def_error_fhmm":   n_def_error_fhmm,
                "time_fhmm":          time_fhmm,           # <---
            })

df = pd.DataFrame(rows)
df.to_csv("resultados.csv", index=False)
print("\nResultados guardados en resultados.csv")

methods = {
    "WTLP":   results_wtlp,
    "KMeans": results_kmeans,
    "FHMM":   results_fhmm,
}

x_values = sorted(results_wtlp.keys())  # n_events ordenados

# Índices en el array de resultados
IDX_TAUE = 1
IDX_TAUC = 2

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for col, (method_name, results) in enumerate(methods.items()):

    taue_means = [results[x][:, IDX_TAUE].mean() for x in x_values]
    taue_stds  = [results[x][:, IDX_TAUE].std()  for x in x_values]

    tauc_means = [results[x][:, IDX_TAUC].mean() for x in x_values]
    tauc_stds  = [results[x][:, IDX_TAUC].std()  for x in x_values]

    ax = axes[0, col]
    taue_errors = asymmetric_errors(taue_means, taue_stds)
    ax.errorbar(x_values, taue_means, yerr=taue_errors,
                fmt='o-', capsize=5, capthick=1.5, elinewidth=1.5)
    ax.set_ylim(bottom=-0.1)  # fuerza el eje Y a empezar en 0
    ax.set_title(method_name)
    ax.set_xlabel("n_events")
    ax.set_xticks(x_values)
    ax.grid(True, linestyle="--", alpha=0.5)
    if col == 0:
        ax.set_ylabel("taue_error")

    # --- Fila 1: tauc ---
    ax = axes[1, col]
    tauc_errors = asymmetric_errors(tauc_means, tauc_stds)
    ax.errorbar(x_values, tauc_means, yerr=tauc_errors,
                fmt='s-', capsize=5, capthick=1.5, elinewidth=1.5, color="tomato")
    ax.set_ylim(bottom=-0.1)
    ax.set_xlabel("n_events")
    ax.set_xticks(x_values)
    ax.grid(True, linestyle="--", alpha=0.5)
    if col == 0:
        ax.set_ylabel("tauc_error")

        # Calcular el máximo global por fila para fijar ylim
    taue_max = max(
        max(results[x][:, IDX_TAUE].mean() + results[x][:, IDX_TAUE].std() for x in x_values)
        for results in methods.values()
    )
    tauc_max = max(
        max(results[x][:, IDX_TAUC].mean() + results[x][:, IDX_TAUC].std() for x in x_values)
        for results in methods.values()
    )
    tau_max=max(tauc_max,taue_max)
    # Luego al final del bucle, aplicar el mismo ylim a todos
    for col in range(3):
        axes[0, col].set_ylim(-0.1, tau_max * 1.1)  # 10% de margen
        axes[1, col].set_ylim(-0.1, tau_max * 1.1)

axes[0, 1].set_title(f"KMeans\n(taue_error)")  
fig.suptitle("n_events vs error (media ± 1σ)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("resultados_plot.png", dpi=150)
plt.show()