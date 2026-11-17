# Aethelion — Sombras Volumétricas por Stencil

Estudo experimental sobre **como os recursos e os limites da arquitetura de
rasterização condicionam a geração de sombras volumétricas**, construído sobre
um rasterizador próprio, inteiramente por software.

Eixo 7 — Arquitetura de Consoles Antigos.

A pergunta de pesquisa, a modelagem matemática ($P_{\text{inf}}$, extrusão
homogênea $w=0$, regra *top-left*), as hipóteses H1–H5 e as referências estão em
[`METODOLOGIA.md`](METODOLOGIA.md).

---

## Instalação

```bash
pip install -r requirements.txt
```

## Execução

```bash
python -m aethelion                             # tela única: cena, volumes 3D, HUD e barra de estado
python main.py                                  # atalho equivalente na raiz
python main.py --fullscreen                     # inicia diretamente em tela cheia (F11 alterna)
python main.py --experiment                     # conformidade + 8 fases + varreduras + relatório
python main.py --experiment --deterministic     # execução determinística por quadro (reprodutível)
python main.py --conformance                    # apenas a suíte de conformidade (24 testes)
python main.py --sv-sweeps                      # varreduras de resolução, complexidade e P_inf
python main.py --report                         # gráficos e RELATORIO.md a partir dos CSVs
pytest                                          # suíte completa de testes unitários e conformidade
```

Na janela interativa, **ESPAÇO** inicia o roteiro de 96 s. Durante a execução é
possível alterar qualquer parâmetro: cada alteração vira um evento em
`eventos.log` e marca as linhas seguintes como `origem_config=manual`.

### Teclas no Visualizador Interativo

| Tecla | Ação |
|---|---|
| `ESPAÇO` | iniciar / parar o experimento |
| `1` / `2` / `0` | `z-pass` / `z-fail` / sem sombra |
| `W` | alterna `INCR_WRAP` / `INCR` |
| `B` | largura do stencil (8, 5, 4, 3, 2, 1 bits) |
| `D` | largura do z-buffer por fragmento (float32, 24, 16, 12, 10) |
| `P` | precisão numérica (`float32`, `float16`, `fixed16_16`, `fixed12_4`) |
| `E` | comprimento de extrusão finita |
| `I` | alterna extrusão homogênea no infinito ($P_{\text{inf}}, w=0$) |
| `R` | resolução interna |
| `V` | alterna vista: **cena** / **volumes 3D + silhuetas** / **stencil** / **diferença (ray cast)** |
| `G` | medir agora contra o ray casting |
| `C` | rodar a suíte de conformidade |
| `S` | captura PNG |
| setas | orbitar câmera (fora do roteiro) |
| `ESC` | sair |

* Na vista **volumes 3D + silhuetas**: faces frontais dos volumes aparecem em
  âmbar semitransparente, faces traseiras em azul-ciano e arestas de silhueta em
  ciano brilhante (inspirado nas Figuras 2, 3 e 5 de Everitt & Kilgard, 2002).
* Na vista **diferença**: **verde** = acerto, **vermelho** = falso positivo,
  **azul** = falso negativo.

---

## Saídas em `results/`

| Arquivo | Conteúdo |
|---|---|
| `conformidade.csv` | PASS/FAIL por teste de conformidade (24/24) |
| `experimento.csv` | uma linha por quadro, com tempos, memória (KB) e métricas |
| `resumo.csv` | agregação por fase (mediana, IQR, média e desvio padrão) |
| `varredura_sv_*.csv` | varreduras de resolução, complexidade geométrica e extrusão $P_{\text{inf}}$ |
| `eventos.log` | início, transições de fase, overrides manuais, encerramento |
| `fase_*.png` | captura do meio de cada fase |
| `grafico_*.png` | custo por etapa, concordância, carga, saturação e memória |
| `RELATORIO.md` | relatório consolidado com tabelas, dispersão e veredictos |

---

## As oito fases

| Fase | Hipótese | O que isola |
|---|---|---|
| A | H2 | `z-pass`, câmera externa — linha de base |
| B | H2 | `z-fail`, mesma trajetória — custo das tampas |
| C | H1 | `z-pass` com a câmera atravessando volumes — artefato esperado |
| D | H1 | `z-fail` na mesma trajetória — robustez esperada |
| E | H3 | stencil de 3 bits **sem** wrap, zona de alta sobreposição |
| F | H3 | stencil de 3 bits **com** wrap, mesma cena |
| G | H4 | buffer de profundidade de 12 bits (por fragmento), vista rasante |
| H | H5 | extrusão finita curta — volume truncado vs. extrusão robusta |

---

## Arquitetura do Pacote

```text
aethelion/
├── main.py                      # Atalho de entrada na raiz
├── pyproject.toml               # Metadados do pacote e configuração pytest
├── METODOLOGIA.md               # Modelagem matemática e desenho experimental
├── docs/                        # Diagnóstico, alinhamento acadêmico e artigos PDF
├── tests/                       # Testes automatizados (pytest)
└── aethelion/
    ├── __init__.py              # API pública do pacote
    ├── __main__.py              # Suporte a `python -m aethelion`
    ├── cli.py                   # Interface de linha de comando (CLI)
    ├── core/                    # Motor gráfico e estágios do pipeline
    │   ├── mathutil.py          # Álgebra 4x4, perspectiva finita e P_inf
    │   ├── precision.py         # Emulação de ponto fixo/flutuante e Z-buffer N bits
    │   ├── clipping.py          # Recorte Sutherland-Hodgman no plano near
    │   ├── raster.py            # Varredura com regra top-left orientada
    │   ├── stencil.py           # Estágio de stencil OpenGL 1.4 (1..8 bits, wrap)
    │   ├── shadow.py            # Silhueta, extrusão finita e homogênea (w=0)
    │   ├── texture.py           # Texturas procedurais
    │   └── sv_renderer.py       # Renderizador vetorizado + inspeção 3D + memória
    ├── scenes/
    │   └── scene_crypt.py       # Cripta procedural two-manifold e topologia
    ├── evaluation/
    │   ├── conformance.py       # Suíte de conformidade (24 testes)
    │   ├── truth.py             # Referência por ray casting e métricas F1
    │   ├── scenario.py          # Roteiro de 8 fases e varreduras paramétricas
    │   └── report.py            # Gráficos e gerador de RELATORIO.md
    └── ui/
        └── viewer.py            # Aplicação interativa Pygame
```
