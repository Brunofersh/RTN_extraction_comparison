"""
TVD Analysis Script
-------------------
- Busca CSVs en carpetas SNR_{nivel} dentro de una carpeta raiz.
- Extrae mag_trace (ruidoso) y mag_trace_clean de cada CSV.
- Aplica denoising (Chambolle-Pock o Condat) con distintos valores de lambda.
- Calcula el error relativo medio entre output y mag_trace_clean.
- Guarda resultados en CSV y genera plots.
"""

import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


# ============================================================
#  PARAMETROS DE ENTRADA — editar aqui
# ============================================================
TEST_NAME     = "tv_denoising_test_d2"
ROOT_FOLDER   = f"data\\{TEST_NAME}"
LAMBDA_VALUES = np.logspace(-3, 3, 60)
OUTPUT_FOLDER = f"results\\tvd_analysis\\{TEST_NAME}"

# Si SKIP_TO_PLOT = True y existe tvd_results.csv en OUTPUT_FOLDER,
# se salta el analisis y se generan los plots directamente.
SKIP_TO_PLOT  = False

# Modo traza individual
# Si SINGLE_TRACE = True, evalua un unico CSV con un unico lambda y lo plotea.
# Si SINGLE_TRACE_FIND_LAMBDA = True, ignora SINGLE_TRACE_LAMBDA y busca
# automaticamente el lambda optimo tal que Var(residuo)/sigma^2_est = 1.
SINGLE_TRACE             = True
SINGLE_TRACE_CSV         = r"data\tv_denoising_test_d2\SNR_20\rtn_params_0.csv"
SINGLE_TRACE_LAMBDA      = 1.0
SINGLE_TRACE_FIND_LAMBDA = True   # True: busqueda automatica del lambda optimo

# Modo de denoising
# "chambolle_pock" : TVD clasico, salida suavizada      (L2 fidelity)
# "condat"         : Fused Lasso 1D, salida escalonada  (ideal para RTN)
DENOISE_MODE = "condat"
# ============================================================


# ------------------------------------------------------------
#  ALGORITMO CHAMBOLLE-POCK  (TVD clasico, salida suavizada)
# ------------------------------------------------------------
def tvd_denoise(signal: np.ndarray, lam: float, n_iter: int = 500) -> np.ndarray:
    """
    Total Variation Denoising 1D mediante Chambolle-Pock.
    Minimiza:  0.5*||x-y||^2 + lam*TV(x)
    Salida: senal suavizada (NO escalonada).
    """
    y  = signal.astype(float)
    n  = len(y)

    def D(v):
        return np.diff(v)

    def Dt(w):
        out = np.zeros(n)
        out[:-1] -= w
        out[1:]  += w
        return out

    L   = 2.0
    tau = 0.99 / L
    sig = 0.99 / L
    x   = y.copy()
    x_  = y.copy()
    p   = np.zeros(n - 1)

    for _ in range(n_iter):
        p_new = np.clip(p + sig * D(x_), -lam, lam)
        x_new = (x - tau * Dt(p_new) + tau * y) / (1 + tau)
        x_    = 2 * x_new - x
        x     = x_new
        p     = p_new

    return x


# ------------------------------------------------------------
#  ALGORITMO CONDAT  (Fused Lasso 1D, salida piecewise-constant)
# ------------------------------------------------------------
def condat_denoise(signal: np.ndarray, lam: float) -> np.ndarray:
    """
    Fused Lasso 1D exacto — produce salida PIECEWISE CONSTANT (escalonada).
    Resuelve:  min_x  0.5*||x-y||^2  +  lam*sum|x[i+1]-x[i]|

    Algoritmo FLSA via Pool Adjacent Violators (Hoefling, 2010).
    Equivalente al resultado de Condat (2013) pero con implementacion
    mas estable. Condicion de optimalidad KKT: bloques adyacentes se
    fusionan mientras |media_b2 - media_b1| <= 2*lam.
    Complejidad O(n), solucion exacta garantizada.
    """
    y = signal.astype(float)
    n = len(y)

    # Cada bloque: [indice_inicio, suma_y, count]
    blocks = []

    for i in range(n):
        blocks.append([i, y[i], 1])

        # Fusionar bloques adyacentes mientras violen la condicion KKT
        while len(blocks) >= 2:
            b1, b2 = blocks[-2], blocks[-1]
            if abs(b2[1]/b2[2] - b1[1]/b1[2]) <= 2 * lam:
                blocks.pop()
                blocks.pop()
                blocks.append([b1[0], b1[1] + b2[1], b1[2] + b2[2]])
            else:
                break

    # Reconstruir senal piecewise-constant
    x = np.empty(n)
    blocks.append([n, 0, 1])  # centinela
    for idx in range(len(blocks) - 1):
        x[blocks[idx][0]:blocks[idx + 1][0]] = blocks[idx][1] / blocks[idx][2]

    return x


def denoise(signal: np.ndarray, lam: float, mode: str = "condat") -> np.ndarray:
    """Wrapper: selecciona el algoritmo segun 'mode'."""
    if mode == "condat":
        return condat_denoise(signal, lam)
    elif mode == "chambolle_pock":
        return tvd_denoise(signal, lam)
    else:
        raise ValueError(f"Modo desconocido: '{mode}'. Usa 'condat' o 'chambolle_pock'.")


# ------------------------------------------------------------
#  ERROR RELATIVO MEDIO
# ------------------------------------------------------------
def relative_error(denoised: np.ndarray, clean: np.ndarray) -> float:
    """
    Error relativo medio normalizado por el rango de la senal limpia (%):
        sum(|denoised - clean|) / (N * (max(clean) - min(clean))) * 100
    """
    N            = len(clean)
    signal_range = np.max(clean) - np.min(clean)
    if signal_range < 1e-12:
        return 0.0
    return float(np.sum(np.abs(denoised - clean)) / (N * signal_range)) * 100


# ------------------------------------------------------------
#  LECTURA DE DATOS
# ------------------------------------------------------------
def find_snr_folders(root: str) -> dict:
    """
    Devuelve {snr_level_str: [lista de CSVs]} para cada carpeta SNR_* en root.
    """
    root_path = Path(root)
    snr_map   = {}
    for entry in sorted(root_path.iterdir()):
        if entry.is_dir() and re.match(r"SNR_", entry.name, re.IGNORECASE):
            csvs = sorted(entry.glob("*.csv"))
            if csvs:
                snr_map[entry.name.split("_", 1)[1]] = csvs
    return snr_map


def load_traces(csv_path: Path):
    """Carga mag_trace y mag_trace_clean. Devuelve None si faltan columnas."""
    df = pd.read_csv(csv_path)
    if not {"mag_trace", "mag_trace_clean"}.issubset(df.columns):
        print(f"  [WARN] Columnas faltantes en {csv_path.name} — se omite.")
        return None
    return df["mag_trace"].values, df["mag_trace_clean"].values


# ------------------------------------------------------------
#  ANALISIS PRINCIPAL
# ------------------------------------------------------------
def run_analysis(root_folder: str, lambdas: np.ndarray, output_folder: str,
                 mode: str = "condat") -> pd.DataFrame:

    os.makedirs(output_folder, exist_ok=True)

    snr_map = find_snr_folders(root_folder)
    if not snr_map:
        raise FileNotFoundError(
            f"No se encontraron carpetas 'SNR_*' dentro de '{root_folder}'."
        )

    print(f"SNR levels encontrados: {list(snr_map.keys())}")
    print(f"Modo de denoising: {mode}")

    results = []

    for snr_level, csv_list in snr_map.items():
        print(f"\n-- SNR = {snr_level} dB  ({len(csv_list)} archivos) --")

        noisy_traces = []
        clean_traces = []

        for csv_path in csv_list:
            out = load_traces(csv_path)
            if out is None:
                continue
            noisy_traces.append(out[0])
            clean_traces.append(out[1])

        if not noisy_traces:
            print("  Sin datos validos, se omite.")
            continue

        # error baseline y unique values de la senal ruidosa
        noisy_errors      = [relative_error(n, c) for n, c in zip(noisy_traces, clean_traces)]
        noisy_mean_err    = float(np.mean(noisy_errors))
        noisy_std_err     = float(np.std(noisy_errors))
        clean_unique_mean = float(np.mean([len(np.unique(np.round(c, 8))) for c in clean_traces]))

        # estimacion robusta de sigma^2 del ruido via MAD sobre diferencias de primer orden
        # mad_sigma = median(|diff(y)|) / 1.4826  ->  robusto a escalones RTN
        sigma_est_list = [float((np.median(np.abs(np.diff(n))) / 1.4826) ** 2)
                          for n in noisy_traces]
        sigma2_est = float(np.mean(sigma_est_list))   # media sobre trazas del nivel SNR

        print(f"  [baseline ruidoso]  error = {noisy_mean_err:.4f}% +/- {noisy_std_err:.4f}%")
        print(f"  [clean unique vals] media = {clean_unique_mean:.1f}")
        print(f"  [sigma^2 estimada]        = {sigma2_est:.6f}")

        for lam in lambdas:
            denoised_list   = [denoise(n, lam, mode) for n in noisy_traces]
            errors          = [relative_error(d, c) for d, c in zip(denoised_list, clean_traces)]
            unique_denoised = [len(np.unique(np.round(d, 8))) for d in denoised_list]
            unique_clean    = [len(np.unique(np.round(c, 8))) for c in clean_traces]
            unique_diff     = [abs(ud - uc) for ud, uc in zip(unique_denoised, unique_clean)]

            # varianza residual: Var(noisy - denoised) / sigma2_est
            # ratio ~ 1 -> lambda optimo; >> 1 -> sobresuavizado; << 1 -> infrasuavizado
            residual_vars  = [float(np.var(n - d)) for n, d in zip(noisy_traces, denoised_list)]
            resvar_ratios  = [rv / sigma2_est if sigma2_est > 0 else np.nan
                              for rv in residual_vars]

            mean_err           = float(np.mean(errors))
            std_err            = float(np.std(errors))
            mean_unique_den    = float(np.mean(unique_denoised))
            mean_unique_cln    = float(np.mean(unique_clean))
            mean_unique_dif    = float(np.mean(unique_diff))
            mean_resvar_ratio  = float(np.mean(resvar_ratios))

            results.append({
                "snr_level":             snr_level,
                "lambda":                lam,
                "mean_rel_error":        mean_err,
                "std_rel_error":         std_err,
                "noisy_mean_rel_error":  noisy_mean_err,
                "noisy_std_rel_error":   noisy_std_err,
                "mean_unique_denoised":  mean_unique_den,
                "mean_unique_clean":     mean_unique_cln,
                "mean_unique_diff":      mean_unique_dif,
                "mean_resvar_ratio":     mean_resvar_ratio,
                "sigma2_est":            sigma2_est,
                "n_traces":              len(errors),
            })
            print(f"  lambda={lam:.4e}  ->  error={mean_err:.4f}%  "
                  f"resvar_ratio={mean_resvar_ratio:.3f}  unique_den={mean_unique_den:.1f}")

    df = pd.DataFrame(results)
    csv_out = os.path.join(output_folder, "tvd_results.csv")
    df.to_csv(csv_out, index=False)
    print(f"\nResultados guardados en: {csv_out}")

    generate_plots(df, output_folder, mode)
    return df


# ------------------------------------------------------------
#  PLOTS
# ------------------------------------------------------------
def generate_plots(df: pd.DataFrame, output_folder: str, mode: str = "condat"):

    snr_levels = sorted(df["snr_level"].unique(), key=lambda x: float(x))
    colors     = px.colors.qualitative.Bold

    # -- Plot 1: Error vs Lambda --
    fig1 = go.Figure()
    for i, snr in enumerate(snr_levels):
        sub   = df[df["snr_level"] == snr].sort_values("lambda")
        color = colors[i % len(colors)]

        fig1.add_trace(go.Scatter(
            x=sub["lambda"], y=sub["mean_rel_error"],
            mode="lines+markers",
            name=f"SNR = {snr} dB ({mode})",
            line=dict(color=color, width=2),
            marker=dict(size=5),
        ))

        noisy_err = sub["noisy_mean_rel_error"].iloc[0]
        fig1.add_trace(go.Scatter(
            x=[sub["lambda"].min(), sub["lambda"].max()],
            y=[noisy_err, noisy_err],
            mode="lines",
            name=f"SNR = {snr} dB (ruidoso)",
            line=dict(color=color, width=1.5, dash="dot"),
        ))

    fig1.update_layout(
        title=f"Denoising ({mode}): Error relativo medio vs lambda",
        xaxis=dict(title="lambda (log)", type="log"),
        yaxis=dict(title="Error relativo medio (%)", type="log"),
        legend=dict(title="SNR"),
        template="plotly_white",
        hovermode="x unified",
    )
    path1 = os.path.join(output_folder, "tvd_error_vs_lambda.html")
    fig1.write_html(path1)
    print(f"Plot guardado: {path1}")

    # -- Plot 2: Heatmap error --
    pivot = df.pivot_table(index="snr_level", columns="lambda",
                           values="mean_rel_error", aggfunc="mean")
    pivot.index = pd.to_numeric(pivot.index, errors="coerce")
    pivot = pivot.sort_index()

    fig2 = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{v:.2e}" for v in pivot.columns],
        y=[str(v) for v in pivot.index],
        colorscale="Viridis_r",
        colorbar=dict(title="Error (%)"),
        hoverongaps=False,
    ))
    fig2.update_layout(
        title=f"Heatmap ({mode}): Error relativo medio (%) — SNR x lambda",
        xaxis=dict(title="lambda", tickangle=-45),
        yaxis=dict(title="SNR (dB)"),
        template="plotly_white",
    )
    path2 = os.path.join(output_folder, "tvd_heatmap.html")
    fig2.write_html(path2)
    print(f"Plot guardado: {path2}")

    # -- Plot 3: Lambda optimo por SNR --
    best_rows = df.loc[df.groupby("snr_level")["mean_rel_error"].idxmin()].copy()
    best_rows["snr_numeric"] = pd.to_numeric(best_rows["snr_level"], errors="coerce")
    best_rows = best_rows.sort_values("snr_numeric")

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=best_rows["snr_level"].astype(str),
        y=best_rows["lambda"],
        text=[f"err={e:.2f}%" for e in best_rows["mean_rel_error"]],
        textposition="outside",
        marker_color=colors[:len(best_rows)],
    ))
    fig3.update_layout(
        title=f"Lambda optimo por SNR ({mode})",
        xaxis=dict(title="SNR (dB)"),
        yaxis=dict(title="lambda optimo", type="log"),
        template="plotly_white",
    )
    path3 = os.path.join(output_folder, "tvd_optimal_lambda.html")
    fig3.write_html(path3)
    print(f"Plot guardado: {path3}")

    # -- Plot 4: Heatmap mejora --
    df = df.copy()
    df["mejora_%"] = ((df["noisy_mean_rel_error"] - df["mean_rel_error"])
                      / df["noisy_mean_rel_error"] * 100)

    pivot_imp = df.pivot_table(index="snr_level", columns="lambda",
                               values="mejora_%", aggfunc="mean")
    pivot_imp.index = pd.to_numeric(pivot_imp.index, errors="coerce")
    pivot_imp = pivot_imp.sort_index()

    zmin_imp = float(pivot_imp.values[~np.isnan(pivot_imp.values)].min())

    fig4 = go.Figure(data=go.Heatmap(
        z=pivot_imp.values,
        x=[f"{v:.2e}" for v in pivot_imp.columns],
        y=[str(v) for v in pivot_imp.index],
        colorscale="RdYlGn",
        zmin=zmin_imp,
        zmax=100,
        colorbar=dict(title="Mejora (%)"),
        hoverongaps=False,
    ))
    fig4.update_layout(
        title=f"Heatmap ({mode}): Mejora sobre senal ruidosa (%) — SNR x lambda",
        xaxis=dict(title="lambda", tickangle=-45),
        yaxis=dict(title="SNR (dB)"),
        template="plotly_white",
    )
    path4 = os.path.join(output_folder, "tvd_heatmap_mejora.html")
    fig4.write_html(path4)
    print(f"Plot guardado: {path4}")

    # -- Plot 5: Valores unicos (denoised vs clean) vs lambda --
    if "mean_unique_denoised" in df.columns:
        fig5 = go.Figure()
        for i, snr in enumerate(snr_levels):
            sub   = df[df["snr_level"] == snr].sort_values("lambda")
            color = colors[i % len(colors)]

            # valores unicos del denoised
            fig5.add_trace(go.Scatter(
                x=sub["lambda"], y=sub["mean_unique_denoised"],
                mode="lines+markers",
                name=f"SNR = {snr} dB (denoised)",
                line=dict(color=color, width=2),
                marker=dict(size=5),
            ))

            # linea horizontal: valores unicos de la senal clean
            clean_u = sub["mean_unique_clean"].iloc[0]
            fig5.add_trace(go.Scatter(
                x=[sub["lambda"].min(), sub["lambda"].max()],
                y=[clean_u, clean_u],
                mode="lines",
                name=f"SNR = {snr} dB (clean)",
                line=dict(color=color, width=1.5, dash="dot"),
            ))

        fig5.update_layout(
            title=f"Denoising ({mode}): Valores unicos (denoised vs clean) vs lambda",
            xaxis=dict(title="lambda (log)", type="log"),
            yaxis=dict(title="Numero de valores unicos (media)"),
            legend=dict(title="SNR"),
            template="plotly_white",
            hovermode="x unified",
        )
        path5 = os.path.join(output_folder, "tvd_unique_values_vs_lambda.html")
        fig5.write_html(path5)
        print(f"Plot guardado: {path5}")

    # -- Plot 6: Ratio varianza residual vs lambda --
    if "mean_resvar_ratio" in df.columns:
        fig6 = go.Figure()
        for i, snr in enumerate(snr_levels):
            sub   = df[df["snr_level"] == snr].sort_values("lambda")
            color = colors[i % len(colors)]

            fig6.add_trace(go.Scatter(
                x=sub["lambda"], y=sub["mean_resvar_ratio"],
                mode="lines+markers",
                name=f"SNR = {snr} dB",
                line=dict(color=color, width=2),
                marker=dict(size=5),
            ))

        # linea de referencia en ratio = 1 (lambda optimo teorico)
        lam_min = df["lambda"].min()
        lam_max = df["lambda"].max()
        fig6.add_trace(go.Scatter(
            x=[lam_min, lam_max], y=[1.0, 1.0],
            mode="lines",
            name="ratio = 1 (optimo)",
            line=dict(color="black", width=1.5, dash="dash"),
            showlegend=True,
        ))

        fig6.update_layout(
            title=f"Denoising ({mode}): Ratio varianza residual vs lambda<br>"
                  "<sup>ratio = Var(noisy - denoised) / sigma^2_est  |  "
                  "ratio~1: lambda optimo  |  ratio>>1: sobresuavizado  |  ratio<<1: infrasuavizado</sup>",
            xaxis=dict(title="lambda (log)", type="log"),
            yaxis=dict(title="Var(residuo) / sigma^2_est", type="log"),
            legend=dict(title="SNR"),
            template="plotly_white",
            hovermode="x unified",
        )
        path6 = os.path.join(output_folder, "tvd_resvar_ratio_vs_lambda.html")
        fig6.write_html(path6)
        print(f"Plot guardado: {path6}")


# ------------------------------------------------------------
#  BUSQUEDA DE LAMBDA OPTIMO  (biseccion sobre ratio varianza residual)
# ------------------------------------------------------------
def estimate_sigma2(noisy: np.ndarray) -> float:
    """
    Estimacion robusta de sigma^2 del ruido via MAD sobre diferencias:
        sigma = median(|diff(y)|) / 1.4826
    Robusto a los escalones RTN porque usa la mediana.
    """
    return float((np.median(np.abs(np.diff(noisy))) / 1.4826) ** 2)


def resvar_ratio(noisy: np.ndarray, lam: float, mode: str) -> float:
    """Calcula Var(noisy - denoised) / sigma^2_est para un lambda dado."""
    sigma2 = estimate_sigma2(noisy)
    if sigma2 < 1e-12:
        return 1.0
    d = denoise(noisy, lam, mode)
    return float(np.var(noisy - d) / sigma2)


def find_optimal_lambda(noisy: np.ndarray, mode: str = "condat",
                        lam_lo: float = 1e-6, lam_hi: float = 1e6,
                        tol: float = 1e-4, max_iter: int = 100) -> tuple:
    """
    Busca por biseccion el lambda tal que Var(residuo)/sigma^2_est = 1.

    El ratio es monotonamente creciente con lambda:
      - lambda pequeno -> denoised ~ noisy -> residuo ~ 0 -> ratio < 1
      - lambda grande  -> denoised ~ cte   -> residuo ~ noisy -> ratio > 1

    Returns
    -------
    lam_opt  : lambda optimo encontrado
    ratio    : ratio final (deberia ser ~1.0)
    n_iter   : iteraciones usadas
    """
    # verificar que el ratio cruza 1 en el rango dado
    r_lo = resvar_ratio(noisy, lam_lo, mode)
    r_hi = resvar_ratio(noisy, lam_hi, mode)

    if r_lo > 1.0:
        print(f"  [WARN] ratio en lam_lo={lam_lo:.2e} ya es {r_lo:.3f} > 1. "
              f"Reducir lam_lo.")
        return lam_lo, r_lo, 0
    if r_hi < 1.0:
        print(f"  [WARN] ratio en lam_hi={lam_hi:.2e} es {r_hi:.3f} < 1. "
              f"Aumentar lam_hi.")
        return lam_hi, r_hi, 0

    # biseccion en escala logaritmica (mas eficiente para lambdas)
    log_lo = np.log10(lam_lo)
    log_hi = np.log10(lam_hi)

    for it in range(max_iter):
        log_mid = 0.5 * (log_lo + log_hi)
        lam_mid = 10 ** log_mid
        r_mid   = resvar_ratio(noisy, lam_mid, mode)

        if abs(r_mid - 1.0) < tol:
            return lam_mid, r_mid, it + 1

        if r_mid < 1.0:
            log_lo = log_mid
        else:
            log_hi = log_mid

    lam_mid = 10 ** (0.5 * (log_lo + log_hi))
    return lam_mid, resvar_ratio(noisy, lam_mid, mode), max_iter


# ------------------------------------------------------------
#  MODO TRAZA INDIVIDUAL
# ------------------------------------------------------------
def plot_single_trace(csv_path: str, lam: float, output_folder: str,
                      mode: str = "condat", find_lambda: bool = False):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")

    out = load_traces(csv_path)
    if out is None:
        raise ValueError("El CSV no tiene 'mag_trace' y 'mag_trace_clean'.")

    noisy, clean = out

    if find_lambda:
        print(f"  Buscando lambda optimo (biseccion)...")
        lam, final_ratio, n_iter = find_optimal_lambda(noisy, mode=mode)
        print(f"  Lambda optimo: {lam:.6e}  (ratio={final_ratio:.4f}, iter={n_iter})")
    else:
        final_ratio = resvar_ratio(noisy, lam, mode)

    denoised     = denoise(noisy, lam, mode)
    err_noisy    = relative_error(noisy,    clean)
    err_denoised = relative_error(denoised, clean)
    t = np.arange(len(noisy))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=noisy, mode="lines", name="Ruidosa",
        line=dict(color="lightcoral", width=1), opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=t, y=clean, mode="lines", name="Limpia (referencia)",
        line=dict(color="steelblue", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=t, y=denoised, mode="lines", name=f"{mode} (lambda={lam:.3g})",
        line=dict(color="seagreen", width=2),
    ))
    label_lam = f"lambda={lam:.4e} (auto)" if find_lambda else f"lambda={lam:.3g}"
    fig.update_layout(
        title=(f"Traza individual — {csv_path.name} | {mode} | {label_lam}<br>"
               f"<sup>Error ruidosa: {err_noisy:.2f}%  |  "
               f"Error denoised: {err_denoised:.2f}%  |  "
               f"Mejora: {err_noisy - err_denoised:.2f}%  |  "
               f"resvar_ratio: {final_ratio:.3f}</sup>"),
        xaxis=dict(title="Muestra"),
        yaxis=dict(title="Amplitud"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        hovermode="x unified",
    )
    os.makedirs(output_folder, exist_ok=True)
    lam_tag  = f"auto{lam:.4e}" if find_lambda else f"lam{lam:.3g}"
    out_path = os.path.join(output_folder,
                            f"single_trace_{csv_path.stem}_{mode}_{lam_tag}.html")
    fig.write_html(out_path)
    print(f"Error senal ruidosa   : {err_noisy:.4f}%")
    print(f"Error denoised (lambda={lam:.3g}) : {err_denoised:.4f}%")
    print(f"Mejora                : {err_noisy - err_denoised:.4f}%")
    print(f"Plot guardado: {out_path}")


# ------------------------------------------------------------
#  SUMMARY
# ------------------------------------------------------------
def print_summary(df: pd.DataFrame):
    print("\n-- Resumen: lambda optimo por SNR --")
    best = df.loc[df.groupby("snr_level")["mean_rel_error"].idxmin(),
                  ["snr_level", "lambda", "mean_rel_error", "noisy_mean_rel_error"]].copy()
    best["mejora_%"] = ((best["noisy_mean_rel_error"] - best["mean_rel_error"])
                        / best["noisy_mean_rel_error"] * 100)
    print(best.to_string(index=False))


# ------------------------------------------------------------
#  ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":

    if SINGLE_TRACE:
        plot_single_trace(
            csv_path      = SINGLE_TRACE_CSV,
            lam           = SINGLE_TRACE_LAMBDA,
            output_folder = OUTPUT_FOLDER,
            mode          = DENOISE_MODE,
            find_lambda   = SINGLE_TRACE_FIND_LAMBDA,
        )
    else:
        csv_cached = os.path.join(OUTPUT_FOLDER, "tvd_results.csv")

        if SKIP_TO_PLOT:
            if not os.path.exists(csv_cached):
                raise FileNotFoundError(
                    f"SKIP_TO_PLOT=True pero no se encontro '{csv_cached}'.\n"
                    f"Ejecuta primero con SKIP_TO_PLOT=False."
                )
            print(f"[SKIP_TO_PLOT] Cargando desde: {csv_cached}")
            df = pd.read_csv(csv_cached)
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            generate_plots(df, OUTPUT_FOLDER, DENOISE_MODE)
        else:
            df = run_analysis(
                root_folder   = ROOT_FOLDER,
                lambdas       = LAMBDA_VALUES,
                output_folder = OUTPUT_FOLDER,
                mode          = DENOISE_MODE,
            )

        print_summary(df)