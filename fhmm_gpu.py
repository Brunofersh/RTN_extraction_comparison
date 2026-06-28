"""
fhmm.py
-------
Factorial Hidden Markov Model (FHMM) para análisis de trazas RTN.

Modos de datos:
  - real=False  (por defecto): CSV con columna "mag_trace" + ground truth
                ("shift", "taue", "tauc"). num_d se conoce de antemano.
                fhmm_test() devuelve errores frente a la ground truth.
  - real=True : CSV con traza en bruto (sin ground truth ni num_d conocido).
                Se prueba num_d = 1, 2, 3... hasta que el último defecto
                añadido deja de ser significativo.
                fhmm_test() devuelve los parámetros estimados directamente.

Aceleración GPU:
  Si CuPy está instalado y hay una GPU CUDA disponible, todo el núcleo
  numérico del EM (forward, backward, gamma, xi, M-step) se ejecuta en
  GPU mediante CuPy como drop-in de NumPy. Si no hay GPU, el código cae
  automáticamente en NumPy puro sin necesidad de cambiar nada.

  Los bucles Python más costosos (O(T·S²·M) en la versión original) han
  sido reemplazados por operaciones matriciales densas que la GPU ejecuta
  en un solo kernel, lo que hace la diferencia para M > 4.

Instalación de CuPy (elegir según versión de CUDA):
  pip install cupy-cuda12x   # CUDA 12.x
  pip install cupy-cuda11x   # CUDA 11.x
"""

import numpy as np
import pandas as pd
import scipy.stats
from itertools import product

# ── Selección automática de backend (GPU vs CPU) ───────────────────────────
try:
    import cupy as cp
    # Verificar que haya al menos una GPU accesible
    cp.cuda.runtime.getDeviceCount()
    _GPU = True
    print("[fhmm] Backend: CuPy (GPU)")
except Exception:
    cp = np          # CuPy ausente o sin GPU → NumPy como fallback
    _GPU = False
    print("[fhmm] Backend: NumPy (CPU) — instala CuPy para usar GPU")

# xp es el módulo activo (cupy o numpy); permite escribir xp.zeros(...) etc.
xp = cp


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades comunes
# ══════════════════════════════════════════════════════════════════════════════

def num_sim(n1, n2):
    """Similarity score entre dos números (escala 0-1)."""
    return 1 - abs(n1 - n2) / (n1 + n2)


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
    Carga la traza desde el CSV según el formato:
      real=False → columna "mag_trace"  (CSV sintético con ground truth)
      real=True  → todas las filas/columnas como muestras (CSV real, sin GT)
    """
    trace = pd.read_csv(csv_path)
    if not real:
        trace_a = np.array(trace["mag_trace"], dtype=np.float64)
    else:
        trace_a = np.array(trace, dtype=np.float64)
        trace_a = np.ascontiguousarray(trace_a.T.flatten())
        trace_a = trace_a - trace_a.min()
    return trace_a, trace


# ══════════════════════════════════════════════════════════════════════════════
# Inicialización del FHMM
# ══════════════════════════════════════════════════════════════════════════════

def fhmm_init(trace_a, aii_0, aij_0, num_d):
    """
    Inicializa parámetros del FHMM dado num_d (número de defectos).
    Devuelve todo en NumPy; la transferencia a GPU ocurre dentro de fhmm_EM.
    """
    A_m = np.zeros((2, 2, num_d))
    for m in range(num_d):
        A_m[:, :, m] = [[aii_0, aij_0], [aij_0, aii_0]]

    pi_m  = np.ones(num_d) * 0.5
    std_m = np.std(trace_a)
    mu_0  = np.mean(trace_a)
    mu_m  = np.random.randn(num_d) * std_m * 0.5

    comb_array = np.array(list(product([0, 1], repeat=num_d)), dtype=np.float64)
    return A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, num_d


# ══════════════════════════════════════════════════════════════════════════════
# Kernel EM — totalmente vectorizado, ejecutable en GPU con CuPy
# ══════════════════════════════════════════════════════════════════════════════

def _build_T_joint(A_m_x, comb_x, num_d):
    """
    Construye la matriz de transición conjunta T_joint[s, s'] = ∏_m A^m[s^m, s'^m].

    Versión totalmente vectorizada: sin ningún bucle Python.
    Shape de comb_x: (S, M) con S = 2^M.

    Estrategia:
      Para cada cadena m, A_m_x[:,:,m][comb_x[:,m], comb_x[:,m]] da un vector
      de longitud S con las probabilidades de permanecer/salir de cada estado.
      El producto exterior entre esos vectores fila y columna da la contribución
      de la cadena m a T_joint, y el producto sobre m da T_joint.
    """
    S = comb_x.shape[0]
    # índices enteros de los estados de cada cadena para cada combinación
    idx = comb_x.astype(xp.int32)          # (S, M)

    T_joint = xp.ones((S, S), dtype=xp.float64)
    for m in range(num_d):
        # A_row[s]  = probabilidad de salir del estado s^m (fila de A^m)
        # A_col[s'] = probabilidad de entrar al estado s'^m (columna de A^m)
        # contribución_m[s, s'] = A^m[s^m, s'^m]
        row_idx = idx[:, m]                 # (S,)
        col_idx = idx[:, m]                 # (S,)
        A_m_slice = A_m_x[:, :, m]         # (2, 2)
        contrib = A_m_slice[row_idx[:, None], col_idx[None, :]]   # (S, S)
        T_joint *= contrib
    return T_joint


def _build_b(trace_x, comb_x, mu_m_x, mu_0, std_m):
    """
    Calcula la matriz de emisión b[t, s] = p(y_t | estado combinado s).
    Vectorizado sobre t y s simultáneamente.
    Shape: (T, S)
    """
    # mu_s[s] = mu_0 + mu_m · comb[s]   →  (S,)
    mu_s = mu_0 + comb_x @ mu_m_x          # (S,)

    # diff[t, s] = trace[t] - mu_s[s]    →  (T, S)
    diff = trace_x[:, None] - mu_s[None, :]

    # log-pdf gaussiana, estabilizada por max por fila
    log_b = -0.5 * (diff / std_m) ** 2 - xp.log(std_m) - 0.5 * xp.log(2 * xp.pi)
    log_b -= log_b.max(axis=1, keepdims=True)
    return xp.exp(log_b)                    # (T, S)


def _forward(b_x, T_joint, pi_m_x, comb_x, num_d):
    """
    Paso forward con escalado (log-verosimilitud acumulada).
    Devuelve alf (T, S) normalizada y log_likelihood.
    El bucle en t es inherentemente secuencial; cada paso es un matmul (S,)@(S,S).
    """
    T, S = b_x.shape
    alf = xp.zeros((T, S), dtype=xp.float64)

    # t = 0: alf[0, s] = b[0, s] * ∏_m pi_m^s^m * (1-pi_m)^(1-s^m)
    alf[0, :] = b_x[0, :]
    idx = comb_x.astype(xp.int32)          # (S, M)
    for m in range(num_d):
        s_m = comb_x[:, m]
        alf[0, :] *= (pi_m_x[m] ** s_m) * ((1 - pi_m_x[m]) ** (1 - s_m))

    c_0 = alf[0, :].sum()
    alf[0, :] /= c_0
    log_likelihood = float(xp.log(c_0))

    for t in range(1, T):
        alf[t, :] = (alf[t - 1, :] @ T_joint) * b_x[t, :]
        c_t = alf[t, :].sum()
        alf[t, :] /= c_t
        log_likelihood += float(xp.log(c_t))

    return alf, log_likelihood


def _backward(b_x, T_joint):
    """
    Paso backward con escalado.
    Devuelve beta (T, S) normalizada.
    """
    T, S = b_x.shape
    beta = xp.ones((T, S), dtype=xp.float64)

    for t in range(T - 2, -1, -1):
        beta[t, :] = T_joint @ (b_x[t + 1, :] * beta[t + 1, :])
        c_t = beta[t, :].sum()
        beta[t, :] /= c_t

    return beta


def _gamma_from_alf_beta(alf, beta):
    """gamma[t, s] = alf[t,s]*beta[t,s] normalizada. Vectorizado sobre t y s."""
    g = alf * beta                          # (T, S)
    g /= g.sum(axis=1, keepdims=True)
    return g


def _xi_vectorized(alf, beta, b_x, T_joint, comb_x, num_d):
    """
    Calcula xi[t, m, k, k'] = P(s^m_t=k, s^m_{t+1}=k' | Y) de forma vectorizada.

    La versión original tenía un triple bucle Python: O(T·S²·M).
    Aquí la dimensión T y la dimensión S² se calculan de golpe con broadcasting:

      xi_joint[t, s, s'] = alf[t,s] * T_joint[s,s'] * b[t+1,s'] * beta[t+1,s']
                                                                    (sin normalizar)

    Shape: (T-1, S, S) → luego se marginaliza por cadena m con máscaras binarias.

    Nota de memoria: para M=8, S=256 y T=100000 esto serían 256²×100000×8B ≈ 52 GB.
    Para M≤6 (S≤64) y T=100000 el tensor tiene 64²×100000×8B ≈ 3.3 GB, manejable
    en una GPU moderna. Para M>6 se usa la versión por lotes en t (CHUNK_T).
    """
    T, S = alf.shape
    CHUNK_T = 2000   # filas de t procesadas en paralelo (ajustar según VRAM)

    # Máscaras: mask_m_k[m, k, s] = 1 si comb[s, m] == k
    mask = xp.zeros((num_d, 2, S), dtype=xp.float64)
    for m in range(num_d):
        for k in range(2):
            mask[m, k, :] = (comb_x[:, m] == k).astype(xp.float64)

    # xi acumulado (ya marginalizado): (num_d, 2, 2)
    xi_sum = xp.zeros((num_d, 2, 2), dtype=xp.float64)

    for t_start in range(0, T - 1, CHUNK_T):
        t_end = min(t_start + CHUNK_T, T - 1)
        chunk = t_end - t_start

        # xi_joint[chunk, S, S] (sin normalizar por t)
        # alf_chunk[:, :, None] * T_joint[None, :, :] * (b * beta)[None, None, :]
        b_beta_next = b_x[t_start + 1: t_end + 1, :] * beta[t_start + 1: t_end + 1, :]
        # (chunk, S, S)
        xi_joint = (alf[t_start:t_end, :, None]
                    * T_joint[None, :, :]
                    * b_beta_next[:, None, :])

        # Normalizar por (t): dividir por suma sobre (s, s')
        norm = xi_joint.sum(axis=(1, 2), keepdims=True)
        norm = xp.where(norm > 0, norm, xp.ones_like(norm))
        xi_joint /= norm                    # (chunk, S, S)

        # Marginalizar: xi_sum[m, k, k'] += sum_t sum_{s:s^m=k} sum_{s':s'^m=k'} xi_joint[t,s,s']
        for m in range(num_d):
            for k in range(2):
                for kp in range(2):
                    # mask[m,k]: (S,)  mask[m,kp]: (S,)
                    # producto exterior de máscaras: (S, S)
                    outer = mask[m, k, :, None] * mask[m, kp, None, :]   # (S, S)
                    # suma sobre t, s, s': escalar
                    xi_sum[m, k, kp] += (xi_joint * outer[None, :, :]).sum()

    # Normalizar xi_sum por cadena m
    for m in range(num_d):
        total = xi_sum[m].sum()
        if total > 0:
            xi_sum[m] /= total

    return xi_sum   # (num_d, 2, 2)   — xi marginalizado y sumado sobre t


def _mstep_pi(gamma, comb_x, num_d):
    """M-step: actualización de pi_m. Vectorizado."""
    # gamma[0, s] * (comb[s, m] == 1)
    pi_m = xp.zeros(num_d, dtype=xp.float64)
    for m in range(num_d):
        mask = (comb_x[:, m] == 1)
        pi_m[m] = gamma[0, mask].sum()
    return pi_m


def _mstep_A(xi_sum):
    """M-step: actualización de A_m desde xi ya marginalizado y sumado sobre t."""
    num_d = xi_sum.shape[0]
    A_m = xp.zeros((2, 2, num_d), dtype=xp.float64)
    for m in range(num_d):
        for k in range(2):
            denom = xi_sum[m, k, :].sum()
            for kp in range(2):
                A_m[k, kp, m] = xi_sum[m, k, kp] / denom if denom > 0 else 0.0
    return A_m


def _mstep_mu(gamma, trace_x, comb_x, mu_m_x, mu_0, num_d):
    """
    M-step: actualización de mu_m y mu_0.
    Vectorizado sobre t y s:
      mu_m[m] = sum_{t,s: s^m=1} gamma[t,s] * (y_t - mu_0 - sum_{l≠m} s^l * mu_l)
                / sum_{t,s: s^m=1} gamma[t,s]
    """
    T = trace_x.shape[0]
    # mu_s[s] = mu_0 + comb @ mu_m   →  (S,)
    mu_s_all = mu_0 + comb_x @ mu_m_x          # (S,)

    # residual_full[t, s] = y_t - mu_s[s]       →  (T, S)
    residual_full = trace_x[:, None] - mu_s_all[None, :]

    mu_m_new = xp.zeros(num_d, dtype=xp.float64)
    for m in range(num_d):
        mask_1 = (comb_x[:, m] == 1)           # (S,) estados donde s^m=1
        # Contribución de la cadena m al residual (sin el término m):
        # residual_m[t, s] = y_t - mu_0 - sum_{l≠m} s^l * mu_l
        #                  = residual_full[t,s] + s^m * mu_m[m]
        # Como mask_1 filtra s con s^m=1:
        #   residual_m[t, s] = residual_full[t,s] + mu_m[m]
        g_masked = gamma[:, mask_1]             # (T, S_1)
        r_masked = residual_full[:, mask_1] + mu_m_x[m]  # (T, S_1)

        num = (g_masked * r_masked).sum()
        den = g_masked.sum()
        mu_m_new[m] = num / den if den > 0 else mu_m_x[m]

    # mu_0: sum_{t,s} gamma[t,s] * (y_t - mu_m @ comb[s])
    mu_contribution = comb_x @ mu_m_new        # (S,)   suma de mu_m ponderada por comb
    residual_0 = trace_x[:, None] - mu_contribution[None, :]   # (T, S)
    mu_0_new = (gamma * residual_0).sum() / T

    return mu_m_new, float(mu_0_new)


def _mstep_std(gamma, trace_x, comb_x, mu_m_x, mu_0):
    """M-step: actualización de std. Vectorizado sobre t y s."""
    T = trace_x.shape[0]
    mu_s = mu_0 + comb_x @ mu_m_x              # (S,)
    diff = trace_x[:, None] - mu_s[None, :]    # (T, S)
    sigma2 = (gamma * diff ** 2).sum() / T
    return float(xp.sqrt(sigma2))


# ══════════════════════════════════════════════════════════════════════════════
# Loop EM principal
# ══════════════════════════════════════════════════════════════════════════════

def fhmm_EM(A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, tolerance, max_run, num_d):
    """
    Ejecuta el algoritmo Baum-Welch (EM) para el FHMM.
    Si hay GPU disponible, todos los tensores se alojan en GPU durante el bucle.
    Al terminar devuelve todo en NumPy (igual que la versión original).
    """
    # ── Transferir a GPU (o no-op si xp == np) ────────────────────────────
    trace_x  = xp.array(trace_a,    dtype=xp.float64)
    comb_x   = xp.array(comb_array, dtype=xp.float64)
    A_m_x    = xp.array(A_m,        dtype=xp.float64)
    pi_m_x   = xp.array(pi_m,       dtype=xp.float64)
    mu_m_x   = xp.array(mu_m,       dtype=xp.float64)

    run = 0
    log_like_ant = 0.0

    while True:
        run += 1

        # ── E-step ────────────────────────────────────────────────────────
        b_x       = _build_b(trace_x, comb_x, mu_m_x, mu_0, std_m)
        T_joint   = _build_T_joint(A_m_x, comb_x, num_d)
        alf, log_likelihood = _forward(b_x, T_joint, pi_m_x, comb_x, num_d)

        # Criterio de parada (antes de backward para ahorrar tiempo si converge)
        converged = abs(log_like_ant - log_likelihood) < tolerance
        if converged or run > max_run:
            status = "CONVERGED" if converged else "DID NOT CONVERGE"
            print(f'Run {run:3d} | log_likelihood: {log_likelihood:.4f} | {status}')
            # Devolver en NumPy
            if _GPU:
                return (cp.asnumpy(A_m_x), cp.asnumpy(pi_m_x), trace_a,
                        cp.asnumpy(mu_m_x), mu_0, std_m, comb_array,
                        log_likelihood, run)
            else:
                return A_m_x, pi_m_x, trace_a, mu_m_x, mu_0, std_m, comb_array, log_likelihood, run

        log_like_ant = log_likelihood

        beta      = _backward(b_x, T_joint)
        gamma     = _gamma_from_alf_beta(alf, beta)
        xi_sum    = _xi_vectorized(alf, beta, b_x, T_joint, comb_x, num_d)

        # ── M-step ───────────────────────────────────────────────────────
        pi_m_x  = _mstep_pi(gamma, comb_x, num_d)
        A_m_x   = _mstep_A(xi_sum)
        mu_m_x, mu_0 = _mstep_mu(gamma, trace_x, comb_x, mu_m_x, mu_0, num_d)
        std_m   = _mstep_std(gamma, trace_x, comb_x, mu_m_x, mu_0)


# ══════════════════════════════════════════════════════════════════════════════
# Post-procesado (CPU, NumPy puro)
# ══════════════════════════════════════════════════════════════════════════════

def normalizar_convencion(A_m, pi_m, mu_m, num_d):
    """
    Fuerza que s^m=1 signifique captura (mu_m[m] < 0) para todas las cadenas.
    Si mu_m[m] > 0, intercambia los estados 0 y 1 de esa cadena.
    """
    for m in range(num_d):
        if mu_m[m] > 0:
            mu_m[m] = -mu_m[m]
            A_m[:, :, m] = np.array([[A_m[1, 1, m], A_m[1, 0, m]],
                                      [A_m[0, 1, m], A_m[0, 0, m]]])
            pi_m[m] = 1 - pi_m[m]
    return A_m, pi_m, mu_m


def _dwell_times_from_A(A_m, mu_m, num_d):
    """Extrae shift/taue/tauc descartando defectos con amplitud despreciable."""
    shift, taue_array, tauc_array = [], [], []
    small_amplitude = False
    for m in range(num_d):
        p01 = A_m[0, 1, m]   # inactivo → activo
        p10 = A_m[1, 0, m]   # activo → inactivo
        tau_0 = 1.0 / p01 if p01 > 0 else np.inf
        tau_1 = 1.0 / p10 if p10 > 0 else np.inf
        if abs(mu_m[m]) > 1e-4:
            shift.append(abs(mu_m[m]))
            taue_array.append(tau_1)
            tauc_array.append(tau_0)
        else:
            small_amplitude = True
    return shift, taue_array, tauc_array, small_amplitude


def _compute_errors(shift, taue_array, tauc_array, gt_shift, gt_taue, gt_tauc):
    shift      = list(shift)
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
            e_shift += np.abs(shift[top_index[i][0]]      - gt_shift[top_index[i][1]]) / np.abs(gt_shift[top_index[i][1]])
            e_taue  += np.abs(taue_array[top_index[i][0]] - gt_taue [top_index[i][1]]) / np.abs(gt_taue [top_index[i][1]])
            e_tauc  += np.abs(tauc_array[top_index[i][0]] - gt_tauc [top_index[i][1]]) / np.abs(gt_tauc [top_index[i][1]])
            shift[top_index[i][0]] = taue_array[top_index[i][0]] = tauc_array[top_index[i][0]] = 0
        e_shift += e_n_defects * 10
        e_taue  += e_n_defects * 10
        e_tauc  += e_n_defects * 10
    else:
        e_shift = e_taue = e_tauc = 0
        val_array, top_index = top_n_valores_e_indices(comparison_array, len(shift))
        for i in range(len(shift)):
            e_shift += np.abs(shift[top_index[i][0]]      - gt_shift[top_index[i][1]]) / np.abs(gt_shift[top_index[i][1]])
            e_taue  += np.abs(taue_array[top_index[i][0]] - gt_taue [top_index[i][1]]) / np.abs(gt_taue [top_index[i][1]])
            e_tauc  += np.abs(tauc_array[top_index[i][0]] - gt_tauc [top_index[i][1]]) / np.abs(gt_tauc [top_index[i][1]])
            gt_shift[top_index[i][1]] = gt_taue[top_index[i][1]] = gt_tauc[top_index[i][1]] = 0
        e_shift += e_n_defects * 10
        e_taue  += e_n_defects * 10
        e_tauc  += e_n_defects * 10

    cap = max(len(shift), len(gt_shift)) * 10
    return min(e_shift, cap), min(e_taue, cap), min(e_tauc, cap), e_n_defects


# ══════════════════════════════════════════════════════════════════════════════
# API pública
# ══════════════════════════════════════════════════════════════════════════════

def fhmm_test(csv_path, aii_0, aij_0, tolerance, max_runs, real=False):
    """
    Ejecuta el FHMM sobre una traza RTN.

    Parameters
    ----------
    real : bool
        False → modo sintético: CSV con "mag_trace" + ground truth.
                num_d conocido por GT. Devuelve errores frente a GT.
        True  → modo real: sin ground truth. Prueba num_d = 1,2,3...
                hasta que el defecto añadido deja de ser significativo.
                Devuelve los parámetros estimados.

    Returns
    -------
    real=False : e_shift, e_taue, e_tauc, e_n_defects, shift, taue, tauc
    real=True  : shift, taue, tauc, num_d_estimado
    """
    trace_a, trace_df = _load_trace(csv_path, real)
    vmax = np.max(trace_a) * 1.1
    vmin = np.min(trace_a) * 1.1

    if not real:
        # ── Modo sintético ─────────────────────────────────────────────
        gt_shift = np.fromstring(trace_df["shift"][0][1:-2], dtype=float, sep=',')

        if vmax != vmin:
            num_d = len(gt_shift)
            A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, num_d = fhmm_init(
                trace_a, aii_0, aij_0, num_d)
            A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, log_likelihood, run = fhmm_EM(
                A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, tolerance, max_runs, num_d)
            A_m, pi_m, mu_m = normalizar_convencion(A_m, pi_m, mu_m, num_d)

            shift, taue_array, tauc_array, _ = _dwell_times_from_A(A_m, mu_m, num_d)

            gt_taue = np.fromstring(trace_df["taue"][0][1:-2], dtype=float, sep=',')
            gt_tauc = np.fromstring(trace_df["tauc"][0][1:-2], dtype=float, sep=',')

            e_shift, e_taue, e_tauc, e_n_defects = _compute_errors(
                shift, taue_array, tauc_array, gt_shift, gt_taue, gt_tauc)

            return e_shift, e_taue, e_tauc, e_n_defects, np.array(shift), np.array(taue_array), np.array(tauc_array)
        else:
            e_n_defects = len(gt_shift)
            penalty = e_n_defects * 10
            return penalty, penalty, penalty, e_n_defects, np.nan, np.nan, np.nan

    else:
        # ── Modo real (num_d desconocido, se incrementa hasta saturar) ──
        if vmax == vmin:
            return [], [], [], 0

        fhmm_flag = 0
        num_d = 0
        shift_ant = taue_array_ant = tauc_array_ant = []

        while num_d < 10 and fhmm_flag == 0:
            num_d += 1
            A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, num_d = fhmm_init(
                trace_a, aii_0, aij_0, num_d)
            A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, log_likelihood, run = fhmm_EM(
                A_m, pi_m, trace_a, mu_m, mu_0, std_m, comb_array, tolerance, max_runs, num_d)
            A_m, pi_m, mu_m = normalizar_convencion(A_m, pi_m, mu_m, num_d)

            shift, taue_array, tauc_array = [], [], []
            for m in range(num_d):
                if fhmm_flag:
                    break
                p01 = A_m[0, 1, m]
                p10 = A_m[1, 0, m]
                tau_0 = 1.0 / p01 if p01 > 0 else np.inf
                tau_1 = 1.0 / p10 if p10 > 0 else np.inf
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