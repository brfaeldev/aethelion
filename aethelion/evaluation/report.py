"""Geração de gráficos e do relatório final a partir dos CSVs do experimento."""
from __future__ import annotations

import csv
import os
from datetime import datetime

from .scenario import HIPOTESES


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(row, key, default=0.0):
    try:
        v = row.get(key, "")
        return float(v) if v != "" else default
    except (TypeError, ValueError):
        return default


def _by_phase(rows):
    return {r["fase"]: r for r in rows}


# --------------------------------------------------------------------------- #
def plots(outdir: str) -> list[str]:
    rows = _read(os.path.join(outdir, "resumo.csv"))
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
    labels = [f"{r['fase']}\n{r['algoritmo']}" for r in rows]
    x = np.arange(len(rows))

    # 1. Custo por etapa
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    geo = [_num(r, "mediana_ms_geometry") for r in rows]
    sil = [_num(r, "mediana_ms_silhouette") for r in rows]
    stn = [_num(r, "mediana_ms_stencil") for r in rows]
    ax.bar(x, geo, label="geometria", color="#4c78a8")
    ax.bar(x, sil, bottom=geo, label="silhueta", color="#f58518")
    ax.bar(x, stn, bottom=[a + b for a, b in zip(geo, sil)],
           label="stencil", color="#e45756")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("tempo mediano por quadro (ms)")
    ax.set_title("Custo por etapa e por fase (backend cpu-sw)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "grafico_custo_etapas.png")
    fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    # 2. Concordância com a referência
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for key, color in (("media_precision", "#4c78a8"),
                       ("media_recall", "#54a24b"),
                       ("media_f1", "#e45756")):
        ax.plot(x, [_num(r, key) for r in rows], "o-", color=color,
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
    ax1.bar(x - 0.2, [_num(r, "mediana_tris_volume") for r in rows], 0.4,
            label="triângulos de volume", color="#9ecae9")
    ax1.set_ylabel("triângulos de volume"); ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, [_num(r, "mediana_frag_stencil") for r in rows], 0.4,
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
    ax.bar(x - 0.2, [_num(r, "mediana_stencil_saturation") for r in rows], 0.4,
           label="saturações (INCR/DECR)", color="#e45756")
    ax.bar(x + 0.2, [_num(r, "mediana_stencil_wrap_events") for r in rows], 0.4,
           label="transbordos (WRAP)", color="#54a24b")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("eventos por quadro (mediana)")
    ax.set_title("Saturação versus transbordo circular do contador de stencil")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "grafico_stencil.png")
    fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    # 5. Perfilamento de memória por fase (buffers arquiteturais + geometria de volume)
    if any(_num(r, "mediana_mem_total_kb") > 0 for r in rows):
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        mbuf = [_num(r, "mediana_mem_buffers_kb") for r in rows]
        mgeo = [_num(r, "mediana_mem_geometry_kb") for r in rows]
        ax.bar(x, mbuf, label="buffers por pixel (cor + depth + stencil)", color="#4c78a8")
        ax.bar(x, mgeo, bottom=mbuf, label="malhas da cena + volumes extrudados", color="#72b7b2")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("memória alocada (KB)")
        ax.set_title("Perfilamento de consumo de memória por fase")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(outdir, "grafico_memoria.png")
        fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    # 6. Varredura de resolução (latência/FPS e memória)
    sw_res = _read(os.path.join(outdir, "varredura_sv_resolucao.csv"))
    if sw_res:
        xr = np.arange(len(sw_res))
        rlbls = [r["variante"] for r in sw_res]
        fig, ax1 = plt.subplots(figsize=(8.0, 3.8))
        ax1.bar(xr - 0.18, [_num(r, "mediana_ms_total") for r in sw_res], 0.36,
                label="tempo mediano (ms)", color="#4c78a8")
        ax1.set_xticks(xr); ax1.set_xticklabels(rlbls, fontsize=8)
        ax1.set_ylabel("tempo mediano por quadro (ms)")
        ax2 = ax1.twinx()
        ax2.plot(xr + 0.18, [_num(r, "mem_total_kb") for r in sw_res], "o-",
                 label="memória total (KB)", color="#e45756")
        ax2.set_ylabel("memória alocada (KB)")
        ax1.set_title("Varredura de Resolução: Latência e Consumo de Memória")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
        fig.tight_layout()
        p = os.path.join(outdir, "grafico_varredura_resolucao.png")
        fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    # 7. Varredura de complexidade geométrica (tris_volume vs ms)
    sw_det = _read(os.path.join(outdir, "varredura_sv_complexidade.csv"))
    if sw_det:
        xd = np.arange(len(sw_det))
        dlbls = [r["variante"] for r in sw_det]
        fig, ax1 = plt.subplots(figsize=(8.0, 3.8))
        ax1.bar(xd - 0.18, [_num(r, "tris_volume") for r in sw_det], 0.36,
                label="triângulos de volume", color="#9ecae9")
        ax1.set_xticks(xd); ax1.set_xticklabels(dlbls, fontsize=8)
        ax1.set_ylabel("triângulos de volume")
        ax2 = ax1.twinx()
        ax2.plot(xd + 0.18, [_num(r, "mediana_ms_total") for r in sw_det], "s-",
                 label="tempo mediano (ms)", color="#f58518")
        ax2.set_ylabel("tempo mediano por quadro (ms)")
        ax1.set_title("Varredura de Complexidade Geométrica: Carga vs. Latência")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
        fig.tight_layout()
        p = os.path.join(outdir, "grafico_varredura_complexidade.png")
        fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    # 8. Varredura de extrusão (finita vs P_inf w=0)
    sw_ext = _read(os.path.join(outdir, "varredura_sv_extrusao.csv"))
    if sw_ext:
        xe = np.arange(len(sw_ext))
        elbls = [r["variante"] for r in sw_ext]
        fig, ax = plt.subplots(figsize=(8.0, 3.8))
        ax.bar(xe - 0.25, [_num(r, "media_precision") for r in sw_ext], 0.25,
               label="precisão", color="#4c78a8")
        ax.bar(xe, [_num(r, "media_recall") for r in sw_ext], 0.25,
               label="revocação", color="#54a24b")
        ax.bar(xe + 0.25, [_num(r, "media_f1") for r in sw_ext], 0.25,
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


# --------------------------------------------------------------------------- #
def _verdict(cond: bool | None) -> str:
    if cond is None:
        return "INCONCLUSIVO (dados insuficientes)"
    return "SUSTENTADA pelos dados" if cond else "NÃO SUSTENTADA pelos dados"


def write_report(outdir: str) -> str:
    resumo = _read(os.path.join(outdir, "resumo.csv"))
    conf = _read(os.path.join(outdir, "conformidade.csv"))
    sw_res = _read(os.path.join(outdir, "varredura_sv_resolucao.csv"))
    sw_det = _read(os.path.join(outdir, "varredura_sv_complexidade.csv"))
    sw_ext = _read(os.path.join(outdir, "varredura_sv_extrusao.csv"))
    ph = _by_phase(resumo)

    def f1(k):
        return _num(ph[k], "media_f1", float("nan")) if k in ph else float("nan")

    def ms(k):
        return _num(ph[k], "mediana_ms_total") if k in ph else 0.0

    def tv(k):
        return _num(ph[k], "mediana_tris_volume") if k in ph else 0.0

    # Veredictos
    h1 = None
    if "C" in ph and "D" in ph:
        h1 = f1("D") > f1("C")
    h2 = None
    if "A" in ph and "B" in ph:
        h2 = (ms("B") > ms("A")) and (tv("B") > tv("A"))
    h3 = None
    if "E" in ph and "F" in ph:
        h3 = f1("F") > f1("E")
    h4 = None
    if "G" in ph and "B" in ph:
        h4 = f1("G") < f1("B")
    h5 = None
    if "H" in ph and "B" in ph:
        h5 = _num(ph["H"], "media_recall") < _num(ph["B"], "media_recall")

    passed = sum(1 for r in conf if r.get("resultado") == "PASS")
    total = len(conf)

    lines = []
    a = lines.append
    a("# Relatório Experimental — Sombras Volumétricas por Stencil")
    a("")
    a(f"Gerado automaticamente em {datetime.now():%Y-%m-%d %H:%M}. "
      "Todos os números abaixo vêm de `experimento.csv` e `resumo.csv`.")
    a("")
    a("> **Backend.** Todas as medições de tempo foram obtidas no rasterizador "
      "por software (`cpu-sw`). Não descrevem o desempenho de uma GPU e não "
      "devem ser comparadas com hardware real. O backend é registrado em cada "
      "linha dos CSVs.")
    a("")
    a("## 1. Pergunta de pesquisa")
    a("")
    a("Como a arquitetura do buffer (largura do stencil, disponibilidade de "
      "operações de transbordo circular `INCR_WRAP`/`DECR_WRAP`, largura do "
      "buffer de profundidade) e a variante do algoritmo (`z-pass` versus "
      "`z-fail` com extrusão finita ou homogênea no infinito $P_{\\text{inf}}$) "
      "afetam a **robustez**, a **precisão geométrica**, o **consumo de memória** "
      "e o **custo computacional** da geração de sombras volumétricas?")
    a("")
    a("## 2. Validação do instrumento")
    a("")
    if total:
        a(f"A suíte de conformidade do estágio de stencil (OpenGL 1.4, Seção 4.1.5) "
          f"e da regra de preenchimento top-left registrou **{passed}/{total} testes aprovados** "
          f"(`conformidade.csv`). O rasterizador só é usado como instrumento de "
          f"medida depois dessa verificação.")
        falhas = [r["teste"] for r in conf if r.get("resultado") != "PASS"]
        if falhas:
            a("")
            a(f"Testes reprovados: {', '.join(falhas)}. Os resultados das fases "
              "afetadas devem ser lidos com essa ressalva.")
    else:
        a("Suíte de conformidade não executada nesta sessão "
          "(`python main.py --conformance`).")
    a("")
    a("## 3. Resultados por fase (Desempenho, Memória e Concordância)")
    a("")
    if resumo:
        a("| Fase | Hipótese | Configuração | ms (mediana ± IQR) | Tris volume | "
          "Frag. stencil | Memória (KB) | Precisão | Revocação | F1 (± DP) |")
        a("|---|---|---|---|---|---|---|---|---|---|")
        for r in resumo:
            cfg = (f"{r['algoritmo']}, {r['stencil_bits']}b, "
                   f"wrap={'sim' if r['stencil_wrap'] == '1' else 'não'}, "
                   f"z={r['depth_bits'] or 'f32'}, L={r['extrude_len']}")
            iqr_v = _num(r, "iqr_ms_total")
            dp_f1 = _num(r, "dp_f1")
            mem_kb = _num(r, "mediana_mem_total_kb")
            a(f"| {r['fase']} — {r['titulo']} | {r['hipotese']} | {cfg} | "
              f"{_num(r, 'mediana_ms_total'):.1f} ± {iqr_v:.1f} | "
              f"{_num(r, 'mediana_tris_volume'):.0f} | "
              f"{_num(r, 'mediana_frag_stencil'):.0f} | "
              f"{mem_kb:.1f} | "
              f"{_num(r, 'media_precision'):.3f} | "
              f"{_num(r, 'media_recall'):.3f} | "
              f"{_num(r, 'media_f1'):.3f} ± {dp_f1:.3f} |")
    else:
        a("_Sem dados: execute o experimento antes de gerar o relatório._")
    a("")
    a("## 4. Veredicto por hipótese")
    a("")
    for key, cond in (("H1", h1), ("H2", h2), ("H3", h3), ("H4", h4), ("H5", h5)):
        a(f"**{key}** — {HIPOTESES[key]}")
        a("")
        a(f"> {_verdict(cond)}")
        a("")
    if "C" in ph and "D" in ph:
        a(f"Detalhe de H1: F1 de z-pass com a câmera interna = {f1('C'):.3f}; "
          f"z-fail na mesma trajetória = {f1('D'):.3f}; volumes contendo a "
          f"câmera (mediana) = "
          f"{_num(ph['C'], 'mediana_volumes_with_camera'):.0f}.")
        a("")
    if "E" in ph and "F" in ph:
        a(f"Detalhe de H3: saturações por quadro sem wrap = "
          f"{_num(ph['E'], 'mediana_stencil_saturation'):.0f} "
          f"(F1 {f1('E'):.3f}); com wrap = "
          f"{_num(ph['F'], 'mediana_stencil_wrap_events'):.0f} transbordos "
          f"(F1 {f1('F'):.3f}).")
        a("")

    if sw_res or sw_det or sw_ext:
        a("## 5. Análise sob Diferentes Resoluções, Complexidades e Extrusão ($P_{\\text{inf}}$)")
        a("")
        if sw_res:
            a("### 5.1 Varredura de Resolução")
            a("")
            a("| Resolução | ms (mediana ± IQR) | FPS | Frag. stencil | Memória Total (KB) | F1 |")
            a("|---|---|---|---|---|---|")
            for r in sw_res:
                a(f"| {r['variante']} | {_num(r, 'mediana_ms_total'):.1f} ± {_num(r, 'iqr_ms_total'):.1f} | "
                  f"{_num(r, 'mediana_fps'):.2f} | {_num(r, 'frag_stencil'):.0f} | "
                  f"{_num(r, 'mem_total_kb'):.1f} | {_num(r, 'media_f1'):.4f} |")
            a("")
        if sw_det:
            a("### 5.2 Varredura de Complexidade Geométrica (`detail`)")
            a("")
            a("| Complexidade | Tris Cena | Tris Volume | ms (mediana ± IQR) | Memória Total (KB) | F1 |")
            a("|---|---|---|---|---|---|")
            for r in sw_det:
                a(f"| {r['variante']} | {_num(r, 'tris_scene'):.0f} | {_num(r, 'tris_volume'):.0f} | "
                  f"{_num(r, 'mediana_ms_total'):.1f} ± {_num(r, 'iqr_ms_total'):.1f} | "
                  f"{_num(r, 'mem_total_kb'):.1f} | {_num(r, 'media_f1'):.4f} |")
            a("")
        if sw_ext:
            a("### 5.3 Extrusão Finita (Carmack) vs. Homogênea no Infinito ($P_{\\text{inf}}, w=0$ — Everitt & Kilgard)")
            a("")
            a("| Regime de Extrusão | Precisão | Revocação | F1 | ms (mediana) |")
            a("|---|---|---|---|---|")
            for r in sw_ext:
                a(f"| {r['variante']} | {_num(r, 'media_precision'):.4f} | "
                  f"{_num(r, 'media_recall'):.4f} | {_num(r, 'media_f1'):.4f} | "
                  f"{_num(r, 'mediana_ms_total'):.1f} |")
            a("")

    a("## 6. Figuras")
    a("")
    for name, cap in (("grafico_custo_etapas.png", "custo por etapa e por fase"),
                      ("grafico_concordancia.png", "concordância com a referência"),
                      ("grafico_carga.png", "carga geométrica vs. preenchimento"),
                      ("grafico_stencil.png", "saturação vs. transbordo"),
                      ("grafico_memoria.png", "perfilamento de memória por fase"),
                      ("grafico_varredura_resolucao.png", "varredura de resolução"),
                      ("grafico_varredura_complexidade.png", "varredura de complexidade geométrica"),
                      ("grafico_varredura_extrusao.png", "varredura de regime de extrusão")):
        if os.path.exists(os.path.join(outdir, name)):
            a(f"![{cap}]({name})")
            a("")
    a("## 7. Limitações declaradas e Referências")
    a("")
    a("1. Rasterizador em Python/NumPy (`cpu-sw`): os tempos absolutos não são comparáveis "
      "a hardware dedicado. As comparações válidas são as tendências relativas entre fases.")
    a("2. Suporta tanto a extrusão finita parametrizada (`extrude_len`) quanto a extrusão "
      "homogênea no infinito ($w = 0$) com matriz $P_{\\text{inf}}$ e *depth clamping* "
      "(Everitt & Kilgard, 2002). Na ausência de rasterização homogênea 4D sem divisão "
      "perspectiva, quando algum vértice extrudado aponta para trás da câmera ($w_{\\text{clip}} \\le 10^{-3}$), "
      "o rasterizador ancora o volume em distância finita longa ($L = 500$) sob $P_{\\text{inf}}$ "
      "para evitar singularidade de $w_{\\text{clip}} = 0$ na tampa traseira.")
    a("3. A verdade de referência por ray casting assume fonte pontual e oclusão binária; "
      "seu tempo de processamento é descontado do relógio experimental das fases.")
    a("4. Referências fundamentais: Everitt & Kilgard (2002); Segal & Akeley (OpenGL 1.4, 2002); "
      "Li, Magnor & Seidel (WSCG 2003).")
    a("")
    path = os.path.join(outdir, "RELATORIO.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def build(outdir: str = "results") -> str:
    made = plots(outdir)
    for p in made:
        print(f"  -> {p}")
    path = write_report(outdir)
    print(f"  -> {path}")
    return path
