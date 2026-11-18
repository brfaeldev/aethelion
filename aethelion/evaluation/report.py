"""Geração do relatório consolidado (`RELATORIO.md`) e orquestração com `plots`."""
from __future__ import annotations

import os
from datetime import datetime

from .phases import HIPOTESES
from .plots import num_val as _num, plots, read_csv_rows as _read

__all__ = ["plots", "write_report", "build"]


def _by_phase(rows: list[dict]) -> dict[str, dict]:
    return {r["fase"]: r for r in rows}


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

    h1 = (f1("D") > f1("C")) if ("C" in ph and "D" in ph) else None
    h2 = ((ms("B") > ms("A")) and (tv("B") > tv("A"))) if ("A" in ph and "B" in ph) else None
    h3 = (f1("F") > f1("E")) if ("E" in ph and "F" in ph) else None
    h4 = (f1("G") < f1("B")) if ("G" in ph and "B" in ph) else None
    h5 = (_num(ph["H"], "media_recall") < _num(ph["B"], "media_recall")) if ("H" in ph and "B" in ph) else None

    passed = sum(1 for r in conf if r.get("resultado") == "PASS")
    total = len(conf)
    backend_id = resumo[0].get("backend", "cpu-sw") if resumo else "cpu-sw"

    lines = []
    a = lines.append
    a("# Relatório Experimental — Sombras Volumétricas por Stencil")
    a("")
    a(f"Gerado automaticamente em {datetime.now():%Y-%m-%d %H:%M}. "
      "Todos os números abaixo vêm de `experimento.csv` e `resumo.csv`.")
    a("")
    a(f"> **Backend.** As medições desta sessão foram obtidas no backend "
      f"`{backend_id}`. O identificador do backend é registrado em cada "
      "linha dos CSVs para rastreabilidade.")
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
    a(f"1. Backend executado (`{backend_id}`): no modo por software (`cpu-sw`/`cpu-fast`), os tempos absolutos "
      "não são comparáveis a hardware dedicado; as comparações válidas são as tendências relativas entre fases.")
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
