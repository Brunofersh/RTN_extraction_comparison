from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", module="sklearn")


# ──────────────────────────────────────────────────────────────────────────
# Utilidades comunes
# ──────────────────────────────────────────────────────────────────────────

def robust_kmeans(X, n_clusters, n_init=10, max_retries=20, random_state=42):
    for seed in range(random_state, random_state + max_retries):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            try:
                km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init).fit(X)
                if len(np.unique(km.labels_)) == n_clusters:
                    return km
            except ConvergenceWarning:
                continue

    # Fallback: forzar centroides manualmente splitting el cluster más grande
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init).fit(X)
    centers = km.cluster_centers_.copy()

    while len(np.unique(km.labels_)) < n_clusters:
        labels = km.labels_
        unique, counts = np.unique(labels, return_counts=True)
        largest_cluster = unique[np.argmax(counts)]

        std = X[labels == largest_cluster].std(axis=0).mean()
        missing_idx = next(i for i in range(n_clusters) if i not in unique)
        centers[missing_idx] = centers[largest_cluster] + np.random.normal(0, std * 0.1, centers.shape[1])

        km = KMeans(n_clusters=n_clusters, init=centers, n_init=1, random_state=random_state).fit(X)
        centers = km.cluster_centers_.copy()

    return km


def gap_statistic(X, k_min=1, k_max=10, n_refs=10):
    gaps = []
    for k in range(k_min, min(k_max + 1, len(X))):
        km = robust_kmeans(X=X, n_clusters=k, random_state=42)
        Wk = km.inertia_

        ref_inertias = []
        for _ in range(n_refs):
            X_ref = np.random.uniform(X.min(0), X.max(0), X.shape)
            km_ref = robust_kmeans(X=X_ref, n_clusters=k, random_state=42)
            ref_inertias.append(km_ref.inertia_)

        gap = np.log(np.mean(ref_inertias)) - np.log(Wk)
        gaps.append(gap)

    return np.argmax(gaps) + k_min  # K con mayor gap


def num_sim(n1, n2):
    """ calculates a similarity score between 2 numbers """
    return 1 - abs(n1 - n2) / (n1 + n2 + 1e-15)


def fit_dwell_cdf(times, label="", n_tail_fraction=0.95):

    t = np.sort(times)
    n = len(t)

    # CDF empírica con corrección de Hazen (evita ln(0))
    k = np.arange(1, n + 1)
    S = (n - k + 0.5) / n          # supervivencia empírica
    lnS = np.log(S)

    # Usar solo la fracción estable de la cola
    n_fit = max(3, int(n * n_tail_fraction))
    t_fit = t[:n_fit]
    lnS_fit = lnS[:n_fit]

    # Regresión lineal forzada por el origen: lnS = m * t
    m_hat = np.dot(t_fit, lnS_fit) / np.dot(t_fit, t_fit)
    tau = -1.0 / m_hat

    # R² en escala log (bondad del ajuste -> 1 = exponencial pura)
    lnS_pred = m_hat * t_fit
    ss_res = np.sum((lnS_fit - lnS_pred) ** 2)
    ss_tot = np.sum((lnS_fit - lnS_fit.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return tau, r2, t, lnS, m_hat


def top_n_valores_e_indices(arr, n):
    flat = arr.flatten()
    unicos = np.unique(flat)
    ordenados = np.sort(unicos)[::-1]
    top_valores = ordenados[:n]

    resultado = []
    value_list = []

    for val in top_valores:
        pos = np.argwhere(arr == val)[0]
        resultado.append(tuple(pos))
        value_list.append(val)

    return value_list, resultado


def _load_trace(csv_path, real):
    """
    Carga la traza desde el CSV según el formato esperado.

    - real=False : CSV con columna "mag_trace" (formato sintético, con GT)
    - real=True  : CSV con la traza en bruto (filas/columnas de muestras)
    """
    trace = pd.read_csv(csv_path)

    if not real:
        trace_a = np.array(trace["mag_trace"])
    else:
        trace_a = np.array(trace)
        trace_a = np.transpose(trace_a)
        trace_a = np.asarray(trace_a).flatten()
        trace_a = trace_a - np.min(trace_a)

    return trace_a, trace


# ──────────────────────────────────────────────────────────────────────────
# Extracción de shifts vía clustering (común a ambos modos)
# ──────────────────────────────────────────────────────────────────────────

def _cluster_and_extract_shifts(data_array, k_optimo, sim_threshold):
    km = robust_kmeans(X=data_array.reshape(-1, 1), n_clusters=k_optimo, random_state=42)
    centroids = km.cluster_centers_

    data_clean = np.zeros(len(data_array))
    i_shift = np.zeros(len(data_array))
    shift = []

    for value in range(len(data_array)):
        data_clean[value] = centroids[np.argmin(np.abs(centroids - data_array[value]))]
        if value > 0:
            i_shift[value] = data_clean[value] - data_clean[value - 1]
            if (i_shift[value] != np.float64(0)) and (np.abs(i_shift[value]) not in shift):
                shift.append(np.abs(i_shift[value]))

    shift = np.float32(shift)
    rel_table = np.zeros((len(shift), 2))
    rel = 0
    for i in range(len(shift)):
        for j in range(len(shift)):
            if i != j:
                if num_sim(shift[i], shift[j]) >= sim_threshold:
                    if shift[i] not in rel_table:
                        rel_table[rel, 0] = shift[i]
                        rel_table[rel, 1] = shift[j]
                        shift[i] = 0

    shift = [i for i in shift if i != 0]

    return shift, rel_table, i_shift, centroids, km, data_clean


# ──────────────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────────────

def kmeans_test(csv_path, k_known=None, real=False):
    trace_a, trace = _load_trace(csv_path, real)
    data_array = np.array(trace_a, dtype=float)
    vmax = np.max(data_array) * 1.1
    vmin = np.min(data_array) * 1.1

    if not real:
        # ── Modo sintético ──────────────────────────────────────────────
        gt_shift = np.fromstring(trace["shift"][0][1:-2], dtype=float, sep=',')
        gt_taue = np.fromstring(trace["taue"][0][1:-2], dtype=float, sep=',')
        gt_tauc = np.fromstring(trace["tauc"][0][1:-2], dtype=float, sep=',')

        if vmax != vmin:
            k_max = 2 ** len(gt_shift)

            if k_known:
                k_optimo = gap_statistic(data_array.reshape(-1, 1), 2 ** (k_known - 1) + 1, 2 ** k_known)
            else:
                k_optimo = gap_statistic(data_array.reshape(-1, 1), 1, k_max)

            shift, rel_table, i_shift, centroids, km, data_clean = _cluster_and_extract_shifts(
                data_array, k_optimo, sim_threshold=0.99
            )

            for value in range(len(shift)):
                if rel_table[value, 0] != 0:
                    for i in range(len(i_shift)):
                        if np.abs(i_shift[i]) == rel_table[value, 0]:
                            i_shift[i] = rel_table[value, 1] * (i_shift[i] / np.abs(i_shift[i]))

            te_list = [[] for _ in range(len(shift))]
            tc_list = [[] for _ in range(len(shift))]
            t_last = np.zeros(len(shift))

            i_shift = np.float32(i_shift)
            for value in range(len(data_array)):
                if i_shift[value] < 0:
                    try:
                        defect_id = shift.index(np.abs(i_shift[value]))
                    except ValueError:
                        defect_id = np.argmin(np.abs(shift - np.abs(i_shift[value])))
                    tc_list[defect_id].append(value - t_last[defect_id])
                    t_last[defect_id] = value
                elif i_shift[value] > 0:
                    try:
                        defect_id = shift.index(np.abs(i_shift[value]))
                    except ValueError:
                        defect_id = np.argmin(np.abs(shift - np.abs(i_shift[value])))
                    te_list[defect_id].append(value - t_last[defect_id])
                    t_last[defect_id] = value

            tauc_array = np.zeros(len(shift))
            taue_array = np.zeros(len(shift))

            for value in range(len(shift)):
                if len(te_list[value]) != 0 and len(tc_list[value]) != 0:
                    tauc_array[value], r2c, a, b, c = fit_dwell_cdf(np.array(tc_list[value]))
                    taue_array[value], r2e, a, b, c = fit_dwell_cdf(np.array(te_list[value]))

            shift = [shift[i] for i in range(len(shift)) if tauc_array[i] != 0]
            tauc_array = [i for i in tauc_array if i != 0]
            taue_array = [i for i in taue_array if i != 0]

            shift_aux = np.array(shift)
            tauc_array_aux = np.array(tauc_array)
            taue_array_aux = np.array(taue_array)

            comparison_array = np.zeros((len(shift), len(gt_shift)))
            for i in range(len(shift)):
                for j in range(len(gt_shift)):
                    comparison_array[i, j] = num_sim(shift[i], np.array(gt_shift[j]))

            if k_known:
                e_n_defects = np.abs(len(gt_shift) - np.ceil(np.log2(k_optimo)))
            else:
                e_n_defects = np.abs(len(gt_shift) - len(shift))

            if len(shift) >= len(gt_shift):
                val_array, top_index = top_n_valores_e_indices(comparison_array, len(gt_shift))
                e_shift = e_taue = e_tauc = 0
                for i in range(len(gt_shift)):
                    e_shift += np.abs(shift[top_index[i][0]] - gt_shift[top_index[i][1]]) / np.abs(gt_shift[top_index[i][1]])
                    shift[top_index[i][0]] = 0
                    e_taue += np.abs(taue_array[top_index[i][0]] - gt_taue[top_index[i][1]]) / np.abs(gt_taue[top_index[i][1]])
                    taue_array[top_index[i][0]] = 0
                    e_tauc += np.abs(tauc_array[top_index[i][0]] - gt_tauc[top_index[i][1]]) / np.abs(gt_tauc[top_index[i][1]])
                    tauc_array[top_index[i][0]] = 0
                e_shift += e_n_defects * 10
                e_taue += e_n_defects * 10
                e_tauc += e_n_defects * 10
            else:
                e_shift = e_taue = e_tauc = 0
                val_array, top_index = top_n_valores_e_indices(comparison_array, len(shift))
                for i in range(len(shift)):
                    e_shift += np.abs(shift[top_index[i][0]] - gt_shift[top_index[i][1]]) / np.abs(gt_shift[top_index[i][1]])
                    gt_shift[top_index[i][1]] = 0
                    e_taue += np.abs(taue_array[top_index[i][0]] - gt_taue[top_index[i][1]]) / np.abs(gt_taue[top_index[i][1]])
                    gt_taue[top_index[i][1]] = 0
                    e_tauc += np.abs(tauc_array[top_index[i][0]] - gt_tauc[top_index[i][1]]) / np.abs(gt_tauc[top_index[i][1]])
                    gt_tauc[top_index[i][1]] = 0
                e_shift += e_n_defects * 10
                e_taue += e_n_defects * 10
                e_tauc += e_n_defects * 10

            cap = max(len(shift), len(gt_shift)) * 10
            e_taue = min(e_taue, cap)
            e_tauc = min(e_tauc, cap)
            e_shift = min(e_shift, cap)
        else:
            e_n_defects = len(gt_shift)
            e_shift = e_n_defects * 10
            e_taue = e_n_defects * 10
            e_tauc = e_n_defects * 10
            shift_aux = np.nan
            taue_array_aux = np.nan
            tauc_array_aux = np.nan

        return e_shift, e_taue, e_tauc, e_n_defects, shift_aux, taue_array_aux, tauc_array_aux

    else:
        # ── Modo real ────────────────────────────────────────────────────
        if vmax == vmin:
            return []

        if not k_known:
            raise ValueError("kmeans_test(real=True) requiere k_known (p.ej. num_d estimado por FHMM)")

        k_optimo = gap_statistic(data_array.reshape(-1, 1), 2 ** (k_known - 1) + 1, 2 ** k_known)

        shift, rel_table, i_shift, centroids, km, data_clean = _cluster_and_extract_shifts(
            data_array, k_optimo, sim_threshold=0.97
        )

        return shift