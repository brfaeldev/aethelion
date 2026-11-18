# Aethelion — Sombras Volumétricas por Stencil

Estudo experimental sobre **como os recursos e os limites da arquitetura de
rasterização condicionam a geração de sombras volumétricas**, construído sobre
um rasterizador próprio por software (`cpu-sw` / `cpu-fast`) e backend GPU (`gpu-gl`).

Eixo 7 — Arquitetura de Consoles Antigos.

A pergunta de pesquisa, a modelagem matemática ($P_{\text{inf}}$, extrusão
homogênea $w=0$, regra *top-left*), as hipóteses H1–H5 e as referências estão em
[`METODOLOGIA.md`](METODOLOGIA.md).

---

## Instalação

```bash
pip install -r requirements.txt
# Opcional: aceleração JIT em CPU (Numba) e backend OpenGL (ModernGL)
pip install -e ".[dev,fast,gpu]"
```

## Execução

```bash
python -m aethelion                             # tela única: cena, volumes 3D, HUD e barra de estado
python main.py                                  # atalho equivalente na raiz
python main.py --fullscreen                     # inicia diretamente em tela cheia (F11 alterna)
python main.py --backend gpu-gl --preset high   # executa no backend GPU com preset de alta qualidade
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

### Teclas e Mouse no Visualizador Interativo

| Controle | Ação |
|---|---|
| `ESPAÇO` | iniciar / parar o experimento |
| `TAB` | alternar backend em tempo real (`cpu-sw` / `cpu-fast` / `gpu-gl`) |
| `M` | pausar / retomar movimento da luz (renderização sob demanda) |
| `F11` / `F` | alternar modo Tela Cheia |
| `1` / `2` / `0` | `z-pass` / `z-fail` / sem sombra |
| `W` | alterna `INCR_WRAP` / `INCR` |
| `B` | largura do stencil (8, 5, 4, 3, 2, 1 bits) |
| `D` | largura do z-buffer por fragmento (float32, 24, 16, 12, 10) |
| `P` | precisão numérica (`float32`, `float16`, `fixed16_16`, `fixed12_4`) |
| `E` | comprimento de extrusão finita |
| `I` | alterna extrusão homogênea no infinito ($P_{\text{inf}}, w=0$) |
| `R` / `]` / `[` | resolução interna (+/-) |
| `V` | alterna vista: **cena** / **volumes 3D + silhuetas** / **stencil** / **diferença (ray cast)** |
| `G` | medir agora contra o ray casting |
| `C` | rodar a suíte de conformidade |
| `S` | captura PNG |
| Mouse / Setas | arrastar na viewport = orbitar câmera; roda do mouse = zoom / rolar painel |
| `ESC` | sair |

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
| `grafico_*.png` | custo por etapa, concordância, carga, saturação, memória e varreduras |
| `RELATORIO.md` | relatório consolidado com tabelas, dispersão e veredictos |

---

## Arquitetura do Pacote (Componentizada)

```text
aethelion/
├── main.py                      # Atalho de entrada na raiz
├── pyproject.toml               # Metadados do pacote, dependências e configuração pytest
├── METODOLOGIA.md               # Modelagem matemática e desenho experimental
├── docs/                        # Artigo científico IEEE em LaTeX e referências base
├── tests/                       # Testes automatizados e fixtures (pytest)
└── aethelion/
    ├── __init__.py              # API pública do pacote
    ├── __main__.py              # Suporte a `python -m aethelion`
    ├── cli.py                   # Interface de linha de comando (CLI)
    ├── core/                    # Motor gráfico, geometria base e backends
    │   ├── settings.py          # SVSettings, SVStats e PRESETS
    │   ├── mathutil.py          # Álgebra 4x4, perspectiva finita e P_inf
    │   ├── precision.py         # Emulação de ponto fixo/flutuante e Z-buffer N bits
    │   ├── mesh.py              # Solid, Prop, soldagem de vértices e topologia 2-manifold
    │   ├── clipping.py          # Recorte Sutherland-Hodgman no plano near
    │   ├── raster.py            # Varredura com regra top-left orientada (NumPy / Numba)
    │   ├── stencil.py           # Estágio de stencil OpenGL 1.4 (1..8 bits, wrap)
    │   ├── shadow.py            # Silhueta, extrusão finita e homogênea (w=0)
    │   ├── texture.py           # Texturas procedurais e amostragem nearest/bilinear
    │   ├── overlay.py           # Renderizador didático 3D de volumes e silhuetas
    │   ├── backend.py           # Protocolo e fábrica de backends (cpu-sw, cpu-fast, gpu-gl)
    │   ├── base_renderer.py     # Base compartilhada (cache de mundo, volumes e memória)
    │   ├── sv_renderer.py       # Renderizadores CPU (SVRenderer e FastCPURenderer)
    │   ├── gpu_renderer.py      # Renderizador GPU (GLRenderer via ModernGL)
    │   └── shaders.py           # Shaders GLSL desacoplados do código de controle
    ├── scenes/
    │   ├── primitives.py        # Geradores de sólidos 2-manifold (box, cylinder, sphere, torus)
    │   └── crypt.py             # Montagem da cripta procedural e trajetória da luz
    ├── evaluation/
    │   ├── conformance.py       # Suíte de conformidade OpenGL 1.4 (24 testes)
    │   ├── truth.py             # Referência por ray casting e métricas F1
    │   ├── phases.py            # Definição das 8 fases, hipóteses H1–H5 e câmeras
    │   ├── runner.py            # ExperimentRunner e execução headless
    │   ├── sweeps.py            # Varreduras de resolução, complexidade e P_inf
    │   ├── plots.py             # Gerador de gráficos analíticos (Matplotlib)
    │   └── report.py            # Gerador de RELATORIO.md e veredictos
    └── ui/
        ├── theme.py             # Paleta de cores, dimensões e ciclos de opções
        ├── camera.py            # Controlador de câmera orbital (OrbitCamera)
        ├── widgets.py           # Componentes de UI imediata, scroll e hitboxes (UIPanelCanvas)
        ├── panel.py             # Renderização do HUD lateral e barra de status
        └── viewer.py            # Controlador da janela Pygame e loop de eventos
```
