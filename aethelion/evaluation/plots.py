"""
Gerador de Gráficos Analíticos (plots.py)

Lê as tabelas consolidadas (`resumo.csv` e `varredura_sv_*.csv`) e produz
até 8 gráficos em PNG via Matplotlib, cobrindo custo por estágio do pipeline,
concordância com ray casting (Precisão, Revocação, F1), carga geométrica vs.
preenchimento, saturação vs. wrap de stencil, memória e varreduras paramétricas.
"""
from __future__ import annotations

import csv
import os


def read_csv_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num_val(row: dict, key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key, "")
        return float(v) if v != "" else default
    except (TypeError, ValueError):
        return default


def plots(outdir: str) -> list[str]:
    """Gera até 8 gráficos analíticos em PNG a partir dos CSVs de `outdir`."""
    rows = read_csv_rows(os.path.join(outdir, "resumo.csv"))
    if not rows:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return []

    made = []
    backend_id = rows[0].get("backend", "cpu-sw")
    labels = [f"{r['fase']}\n{r['algoritmo']}" for r in rows]
    x = np.arange(len(rows))

    # 1. Custo por etapa
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    geo = [num_val(r, "mediana_ms_geometry") for r in rows]
    sil = [num_val(r, "mediana_ms_silhouette") for r in rows]
    stn = [num_val(r, "mediana_ms_stencil") for r in rows]
    ax.bar(x, geo, label="geometria", color="#4c78a8")
    ax.bar(x, sil, bottom=geo, label="silhueta", color="#f58518")
    ax.bar(x, stn, bottom=[a + b for a, b in zip(geo, sil)],
           label="stencil", color="#e45756")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("tempo mediano por quadro (ms)")
    ax.set_title(f"Custo por etapa e por fase (backend {backend_id})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "grafico_custo_etapas.png")
    fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    # 2. Concordância com a referência
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for key, color in (("media_precision", "#4c78a8"),
                       ("media_recall", "#54a24b"),
                       ("media_f1", "#e45756")):
        ax.plot(x, [num_val(r, key) for r in rows], "o-", color=color,
                label=key.replace("media_", ""))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("concordância com o ray casting")
    ax.set_title("Precisão, revocação e F1 por fase")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "grafico_concordancia.png")
    fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    # 3. Carga geométrica e de preenchimento
    fig, ax1 = plt.subplots(figsize=(8.5, 4.2))
    ax1.bar(x - 0.2, [num_val(r, "mediana_tris_volume") for r in rows], 0.4,
            label="triângulos de volume", color="#9ecae9")
    ax1.set_ylabel("triângulos de volume"); ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, [num_val(r, "mediana_frag_stencil") for r in rows], 0.4,
            label="fragmentos de stencil", color="#ffbf79")
    ax2.set_ylabel("fragmentos de stencil")
    ax1.set_title("Carga geométrica vs. carga de preenchimento")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.tight_layout()
    p = os.path.join(outdir, "grafico_carga.png")
    fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    # 4. Saturação de stencil
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.bar(x - 0.2, [num_val(r, "mediana_stencil_saturation") for r in rows], 0.4,
           label="saturações (INCR/DECR)", color="#e45756")
    ax.bar(x + 0.2, [num_val(r, "mediana_stencil_wrap_events") for r in rows], 0.4,
           label="transbordos (WRAP)", color="#54a24b")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("eventos por quadro (mediana)")
    ax.set_title("Saturação versus transbordo circular do contador de stencil")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "grafico_stencil.png")
    fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    # 5. Perfilamento de memória por fase
    if any(num_val(r, "mediana_mem_total_kb") > 0 for r in rows):
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        mbuf = [num_val(r, "mediana_mem_buffers_kb") for r in rows]
        mgeo = [num_val(r, "mediana_mem_geometry_kb") for r in rows]
        ax.bar(x, mbuf, label="buffers por pixel (cor + depth + stencil)", color="#4c78a8")
        ax.bar(x, mgeo, bottom=mbuf, label="malhas da cena + volumes extrudados", color="#72b7b2")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("memória alocada (KB)")
        ax.set_title("Perfilamento de consumo de memória por fase")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(outdir, "grafico_memoria.png")
        fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    # 6. Varredura de resolução
    sw_res = read_csv_rows(os.path.join(outdir, "varredura_sv_resolucao.csv"))
    if sw_res:
        xr = np.arange(len(sw_res))
        rlbls = [r["variante"] for r in sw_res]
        fig, ax1 = plt.subplots(figsize=(8.0, 3.8))
        ax1.bar(xr - 0.18, [num_val(r, "mediana_ms_total") for r in sw_res], 0.36,
                label="tempo mediano (ms)", color="#4c78a8")
        ax1.set_xticks(xr); ax1.set_xticklabels(rlbls, fontsize=8)
        ax1.set_ylabel("tempo mediano por quadro (ms)")
        ax2 = ax1.twinx()
        ax2.plot(xr + 0.18, [num_val(r, "mem_total_kb") for r in sw_res], "o-",
                 label="memória total (KB)", color="#e45756")
        ax2.set_ylabel("memória alocada (KB)")
        ax1.set_title("Varredura de Resolução: Latência e Consumo de Memória")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
        fig.tight_layout()
        p = os.path.join(outdir, "grafico_varredura_resolucao.png")
        fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    # 7. Varredura de complexidade geométrica
    sw_det = read_csv_rows(os.path.join(outdir, "varredura_sv_complexidade.csv"))
    if sw_det:
        xd = np.arange(len(sw_det))
        dlbls = [r["variante"] for r in sw_det]
        fig, ax1 = plt.subplots(figsize=(8.0, 3.8))
        ax1.bar(xd - 0.18, [num_val(r, "tris_volume") for r in sw_det], 0.36,
                label="triângulos de volume", color="#9ecae9")
        ax1.set_xticks(xd); ax1.set_xticklabels(dlbls, fontsize=8)
        ax1.set_ylabel("triângulos de volume")
        ax2 = ax1.twinx()
        ax2.plot(xd + 0.18, [num_val(r, "mediana_ms_total") for r in sw_det], "s-",
                 label="tempo mediano (ms)", color="#f58518")
        ax2.set_ylabel("tempo mediano por quadro (ms)")
        ax1.set_title("Varredura de Complexidade Geométrica: Carga vs. Latência")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
        fig.tight_layout()
        p = os.path.join(outdir, "grafico_varredura_complexidade.png")
        fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    # 8. Varredura de extrusão
    sw_ext = read_csv_rows(os.path.join(outdir, "varredura_sv_extrusao.csv"))
    if sw_ext:
        xe = np.arange(len(sw_ext))
        elbls = [r["variante"] for r in sw_ext]
        fig, ax = plt.subplots(figsize=(8.0, 3.8))
        ax.bar(xe - 0.25, [num_val(r, "media_precision") for r in sw_ext], 0.25,
               label="precisão", color="#4c78a8")
        ax.bar(xe, [num_val(r, "media_recall") for r in sw_ext], 0.25,
               label="revocação", color="#54a24b")
        ax.bar(xe + 0.25, [num_val(r, "media_f1") for r in sw_ext], 0.25,
               label="F1", color="#e45756")
        ax.set_xticks(xe); ax.set_xticklabels(elbls, fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("concordância com ray casting")
        ax.set_title("Extrusão Finita (Carmack) vs. Homogênea no Infinito (P_inf, w=0)")
        ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(outdir, "grafico_varredura_extrusao.png")
        fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    return made
