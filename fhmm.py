import numpy as np
import pandas as pd
import scipy
from itertools import product




def num_sim(n1, n2):
    """ calculates a similarity score between 2 numbers """
    return 1 - abs(n1 - n2) / (n1 + n2)


def top_n_valores_e_indices(arr, n):
    flat = arr.flatten()
    unicos = np.unique(flat)
    ordenados = np.sort(unicos)[::-1]
    top_valores = ordenados[:n]

    resultado = []
    value_list = []

    for val in top_valores:
        # coger solo la primera ocurrencia
        pos = np.argwhere(arr == val)[0]
        resultado.append(tuple(pos))
        value_list.append(val)

    return value_list, resultado


def _load_trace(csv_path, real):
    """
    Carga la traza desde el CSV según el formato esperado.

    - real=False : CSV con columna "mag_trace" (formato sintético, con GT)
    - real=True  : CSV con la traza en bruto (filas/columnas de muestras),
                   sin cabecera con nombres ("mag_trace" no existe).
    """
    trace = pd.read_csv(csv_path)

    if not real:
        trace_a = np.array(trace["mag_trace"])
    else:
        trace_a = np.array(trace)
        trace_a = np.transpose(trace_a)
        trace_a = np.asarray(trace_a).flatten()
        trace_a = trace_a - np.min(trace_a)

    return trace_a




def fhmm_init(trace_a, aii_0, aij_0, num_d):
    """Inicializa los parámetros del FHMM dado num_d (número de defectos)."""
    A_m = np.zeros((2, 2, num_d))      # Size: 2 x 2 x Number of defects
    pi_m = np.ones((num_d)) * 0.5      # Size: number of defects

    for m in range(num_d):
        A_m[:, :, m] = [[aii_0, aij_0], [aij_0, aii_0]]

    std_m = np.std(trace_a)
    mu_0 = np.mean(trace_a)                                  # nivel basal
    mu_m = np.random.randn(num_d) * np.std(trace_a) * 0.5    # perturbaciones

    comb_array = np.array(list(product([0, 1], repeat=num_d)))
    return A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, num_d


def fhmm_EM(A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, tolerance, max_run, num_d):
    run = 0
    log_like_ant = 0.0

    alf = np.zeros((len(trace_a), len(comb_array)))
    b = np.zeros((len(trace_a), len(comb_array)))

    while True:
        run += 1
        # b precomputado: shape (T, 2^M) — vectorizado
        for n_comb, comb in enumerate(comb_array):
            mu_s = mu_0 + mu_m @ comb
            b[:, n_comb] = scipy.stats.norm.logpdf(trace_a, mu_s, std_m)
        b = np.exp(b - b.max(axis=1, keepdims=True))  # estabilidad numérica

        # matriz de transición conjunta: shape (2^M, 2^M)
        # T_joint[s, s'] = prod_m A^m[s^m, s'^m]
        T_joint = np.ones((len(comb_array), len(comb_array)))
        for n_s, s in enumerate(comb_array):
            for n_sp, sp in enumerate(comb_array):
                for m in range(num_d):
                    T_joint[n_s, n_sp] *= A_m[s[m], sp[m], m]

        # forward vectorizado: un matmul por instante en lugar de bucles anidados
        log_likelihood = 0.0
        alf = np.zeros((len(trace_a), len(comb_array)))
        alf[0, :] = b[0, :]
        for m in range(num_d):
            comb = comb_array[:, m]
            alf[0, :] *= (pi_m[m] ** comb) * ((1 - pi_m[m]) ** (1 - comb))
        c_0 = alf[0, :].sum()
        alf[0, :] /= c_0
        log_likelihood += np.log(c_0)

        for t in range(1, len(trace_a)):
            alf[t, :] = (alf[t - 1, :] @ T_joint) * b[t, :]
            c_t = alf[t, :].sum()
            alf[t, :] /= c_t
            log_likelihood += np.log(c_t)

        if np.abs(log_like_ant - log_likelihood) < tolerance:
            print(f'Run number {run} , with log_likelihood: {log_likelihood}, EXITING EM, CONVERGED!')
            return A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, log_likelihood, run
        elif run > max_run:
            print(f'Run number {run} , with log_likelihood: {log_likelihood}, DID NOT CONVERGE')
            return A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, log_likelihood, run
        log_like_ant = log_likelihood

        # backward vectorizado
        beta = np.ones((len(trace_a), len(comb_array)))
        for t in range(len(trace_a) - 2, -1, -1):
            beta[t, :] = T_joint @ (b[t + 1, :] * beta[t + 1, :])
            c_t = beta[t, :].sum()
            beta[t, :] /= c_t

        # -- E step: Gamma calculation
        gamma = np.zeros((len(trace_a), len(comb_array)))

        for t in range(len(trace_a)):
            for n_comb in range(len(comb_array)):
                gamma[t, n_comb] = alf[t, n_comb] * beta[t, n_comb]
            gamma[t, :] /= np.sum(gamma[t, :])

        # -- E step: Xi calculation
        # xi: shape (T-1, num_d, 2, 2)
        # xi[t, m, k, k'] = P(s^m_t=k, s^m_{t+1}=k' | Y)
        xi = np.zeros((len(trace_a) - 1, num_d, 2, 2))

        for t in range(len(trace_a) - 1):
            for n_s, s in enumerate(comb_array):        # estado en t
                for n_sp, sp in enumerate(comb_array):  # estado en t+1
                    trans_prod = 1.0
                    for m in range(num_d):
                        trans_prod *= A_m[s[m], sp[m], m]

                    val = alf[t, n_s] * trans_prod * b[t + 1, n_sp] * beta[t + 1, n_sp]

                    for m in range(num_d):
                        xi[t, m, s[m], sp[m]] += val

            for m in range(num_d):
                total = np.sum(xi[t, m, :, :])
                if total > 0:
                    xi[t, m, :, :] /= total

        # -- M step: Initial Probabilities Update
        for m in range(num_d):
            pi_m[m] = np.sum([gamma[0, n] for n, c in enumerate(comb_array) if c[m] == 1])

        # -- M step: Transition Matrix Update
        for m in range(num_d):
            for k in range(2):
                denom = np.sum(xi[:, m, k, :])
                for kp in range(2):
                    A_m[k, kp, m] = np.sum(xi[:, m, k, kp]) / denom

        # -- M step: Means Update
        for m in range(num_d):
            num = 0.0
            den = 0.0
            for t in range(len(trace_a)):
                for n_s, s in enumerate(comb_array):
                    if s[m] == 1:
                        residual = trace_a[t] - mu_0 - sum(
                            s[l] * mu_m[l] for l in range(num_d) if l != m
                        )
                        num += gamma[t, n_s] * residual
                        den += gamma[t, n_s]
            mu_m[m] = num / den

        # Actualizar mu_0 después
        num = 0.0
        for t in range(len(trace_a)):
            for n_s, s in enumerate(comb_array):
                num += gamma[t, n_s] * (trace_a[t] - mu_m @ s)
        mu_0 = num / len(trace_a)

        # -- M step: Standard Deviation Update
        num = 0.0
        for t in range(len(trace_a)):
            for n_s, s in enumerate(comb_array):
                mu_s = mu_0 + mu_m @ s
                num += gamma[t, n_s] * (trace_a[t] - mu_s) ** 2
        sigma2 = num / len(trace_a)
        std_dev = np.sqrt(sigma2)


def normalizar_convencion(A_m, pi_m, mu_m, num_d):
    """
    Fuerza que s^m=1 signifique captura (mu_m[m] < 0) para todas las cadenas.
    Si mu_m[m] > 0, intercambia los estados 0 y 1 de esa cadena.
    """
    for m in range(num_d):
        if mu_m[m] > 0:
            mu_m[m] = -mu_m[m]

            # intercambiar filas Y columnas de A^m
            A_m[:, :, m] = np.array([[A_m[1, 1, m], A_m[1, 0, m]],
                                      [A_m[0, 1, m], A_m[0, 0, m]]])

            # pi era P(s=1), ahora P(s=1) es el antiguo P(s=0)
            pi_m[m] = 1 - pi_m[m]

    return A_m, pi_m, mu_m


def _dwell_times_from_A(A_m, mu_m, num_d):
    """Extrae shift/taue/tauc a partir de A_m y mu_m, descartando defectos
    con amplitud despreciable. Devuelve (shift, taue, tauc, defect_is_small)."""
    shift, taue_array, tauc_array = [], [], []
    small_amplitude = False

    for m in range(num_d):
        p_salida_0 = A_m[0, 1, m]   # a^m_01: inactivo -> activo
        p_salida_1 = A_m[1, 0, m]   # a^m_10: activo -> inactivo

        tau_0 = 1.0 / p_salida_0 if p_salida_0 > 0 else np.inf
        tau_1 = 1.0 / p_salida_1 if p_salida_1 > 0 else np.inf

        if abs(mu_m[m]) > 1e-4:
            shift.append(abs(mu_m[m]))
            taue_array.append(tau_1)
            tauc_array.append(tau_0)
        else:
            small_amplitude = True

    return shift, taue_array, tauc_array, small_amplitude




def _compute_errors(shift, taue_array, tauc_array, gt_shift, gt_taue, gt_tauc):
    shift = list(shift)
    taue_array = list(taue_array)
    tauc_array = list(tauc_array)

    comparison_array = np.zeros((len(shift), len(gt_shift)))
    for i in range(len(shift)):
        for j in range(len(gt_shift)):
            comparison_array[i, j] = num_sim(shift[i], gt_shift[j])

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

    return e_shift, e_taue, e_tauc, e_n_defects




def fhmm_test(csv_path, aii_0, aij_0, tolerance, max_runs, real=False):
    """

    Parameters
    ----------
    real : bool

    Returns
    -------
    If real=False:
        e_shift, e_taue, e_tauc, e_n_defects, shift, taue_array, tauc_array
    If real=True:
        shift, taue_array, tauc_array, num_d_estimado
    """
    trace_a = _load_trace(csv_path, real)
    data_array = np.array(trace_a, dtype=float)
    vmax = np.max(data_array) * 1.1
    vmin = np.min(data_array) * 1.1

    if not real:
        # ── Modo sintético: num_d conocido por ground truth ────────────────
        trace = pd.read_csv(csv_path)
        gt_shift = np.fromstring(trace["shift"][0][1:-2], dtype=float, sep=',')

        if vmax != vmin:
            num_d = len(gt_shift)
            A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, num_d = fhmm_init(
                trace_a, aii_0, aij_0, num_d
            )
            A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, log_likelihood, run = fhmm_EM(
                A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, tolerance, max_runs, num_d
            )
            A_m, pi_m, mu_m = normalizar_convencion(A_m, pi_m, mu_m, num_d)

            shift, taue_array, tauc_array, _ = _dwell_times_from_A(A_m, mu_m, num_d)

            gt_taue = np.fromstring(trace["taue"][0][1:-2], dtype=float, sep=',')
            gt_tauc = np.fromstring(trace["tauc"][0][1:-2], dtype=float, sep=',')

            shift_aux = np.array(shift)
            taue_array_aux = np.array(taue_array)
            tauc_array_aux = np.array(tauc_array)

            e_shift, e_taue, e_tauc, e_n_defects = _compute_errors(
                shift, taue_array, tauc_array, gt_shift, gt_taue, gt_tauc
            )
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
        # ── Modo real: num_d desconocido, se incrementa hasta saturar ──────
        if vmax == vmin:
            return [], [], [], 0

        fhmm_flag = 0
        num_d = 0
        shift_ant, taue_array_ant, tauc_array_ant = [], [], []

        while num_d < 10 and fhmm_flag == 0:
            num_d += 1
            A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, num_d = fhmm_init(
                trace_a, aii_0, aij_0, num_d
            )
            A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, log_likelihood, run = fhmm_EM(
                A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, tolerance, max_runs, num_d
            )
            A_m, pi_m, mu_m = normalizar_convencion(A_m, pi_m, mu_m, num_d)

            shift, taue_array, tauc_array = [], [], []
            for m in range(num_d):
                if fhmm_flag:
                    break
                p_salida_0 = A_m[0, 1, m]
                p_salida_1 = A_m[1, 0, m]
                tau_0 = 1.0 / p_salida_0 if p_salida_0 > 0 else np.inf
                tau_1 = 1.0 / p_salida_1 if p_salida_1 > 0 else np.inf

                if abs(mu_m[m]) > 1e-4:
                    shift.append(abs(mu_m[m]))
                    taue_array.append(tau_1)
                    tauc_array.append(tau_0)
                else:
                    fhmm_flag = 1

            if not fhmm_flag:
                shift_ant = shift
                taue_array_ant = taue_array
                tauc_array_ant = tauc_array

        return shift_ant, taue_array_ant, tauc_array_ant, num_d - 1