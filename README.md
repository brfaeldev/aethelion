# Aethelion — Sombras Volumétricas por Stencil

Estudo experimental sobre **como os recursos e os limites da arquitetura de
rasterização condicionam a geração de sombras volumétricas**, construído sobre
um rasterizador próprio por software (`cpu-sw` / `cpu-fast`) e um backend em GPU
via OpenGL 3.3 (`gpu-gl`).

**Enquadramento:** Eixo 7 — Arquitetura de Consoles Antigos.

## 👥 Identificação

* **Bruno Rafael Barbosa** — RA: 112944
* **João Pedro Magrin** — RA: 113164

A formulação matemática detalhada (matriz de perspectiva no infinito $P_{\text{inf}}$,
extrusão homogênea com $w=0$, regra de preenchimento *top-left* orientada,
quantização de profundidade por fragmento e aritmética saturada vs. modular de
stencil), o desenho experimental das hipóteses **H1–H5** e as referências
bibliográficas encontram-se em [`METODOLOGIA.md`](METODOLOGIA.md). Os resultados
consolidados da bateria experimental estão em [`results/RELATORIO.md`](results/RELATORIO.md).

---

## 1. Visão Geral e Hipóteses Experimentais

O projeto investiga empiricamente o impacto de restrições arquiteturais típicas de
pipelines gráficos clássicos e consoles de 5ª/6ª geração sobre o algoritmo de
*stenciled shadow volumes* (Heidmann, 1991; Carmack, 2000; Everitt & Kilgard, 2002)
em uma cena procedural fechada (*two-manifold*) de pior caso (cripta com colunas,
arcos, estátua e grade de alta sobreposição):

| Hipótese | Fator Arquitetural / Algorítmico | Fases Comparadas | Comportamento Investigado |
|---|---|---|---|
| **H1** | `z-pass` vs. `z-fail` (Carmack's Reverse) | **C** vs. **D** | `z-pass` falha quando a câmera entra em volumes de sombra (`volumes_with_camera > 0`); `z-fail` preserva a máscara com tampas fechadas. |
| **H2** | Custo geométrico e de *fill rate* das tampas | **A** vs. **B** | `z-fail` processa e rasteriza as tampas frontal e traseira de cada volume, elevando `tris_volume`, `frag_stencil` e o tempo por quadro. |
| **H3** | Largura do stencil ($1\dots 8$ bits) e `INCR_WRAP`/`DECR_WRAP` | **E** vs. **F** | Contadores estreitos (3 bits) saturam sob alta sobreposição (`INCR`/`DECR`); o transbordo circular (OpenGL 1.4) elimina o erro na mesma cena. |
| **H4** | Quantização do Z-buffer por fragmento | **G** vs. **B** | Reduzir a precisão de profundidade (12 bits) introduz artefatos de autoteste na inversão de profundidade em vistas rasantes. |
| **H5** | Extrusão finita curta vs. longa ou homogênea ($P_{\text{inf}}, w=0$) | **H** vs. **B** / Varredura | Extrusão finita curta ($L=3{,}5$) trunca o volume e derruba a revocação; extrusão longa ($L=40$) ou no infinito ($w=0$) preserva o volume. |

Antes de qualquer coleta experimental, o motor é validado por uma **suíte de
conformidade de 24 testes** (`conformance.py`) baseada na especificação **OpenGL 1.4**
(Seção 4.1.5 — *Stencil Test*, Seção 3.5 — *Polygon Rasterization* e Seção 2.11 — *Clipping*),
e cada quadro medido é comparado contra uma verdade de referência independente
obtida por **ray casting** (Möller-Trumbore) sobre o G-buffer.

---

## 2. Backends e Presets

### Backends de Renderização (`--backend`)

Todos os backends compartilham a mesma infraestrutura de cena, transformação,
extração de silhuetas, cache de volumes e métricas (`BaseRenderer`), cumprindo o
contrato `RendererProtocol`:

* `cpu-sw` *(padrão)* — Rasterizador por software em Python/NumPy puro, com avaliação
  separável 1D+1D de funções de aresta e regra *top-left* orientada.
* `cpu-fast` — Variante CPU acelerada via compilação JIT (`numba.njit`) no laço
  interno de varredura baricêntrica (opera com *fallback* transparente para NumPy se
  Numba não estiver instalado).
* `gpu-gl` — Pipeline OpenGL 3.3 Core (`moderngl`) em 3 passagens *offscreen* (FBO
  de geometria MRT, FBO de acumulação aditiva `RG32F` de incrementos/decrementos
  de stencil para instrumentação exata de $1\dots 8$ bits, e FBO de composição).

### Perfis Prontos (`--preset`)

| Preset | Resolução | `detail` | Precisão | `vertex_snap` | Cor / Dither | Sombreamento / Filtro |
|---|---|---|---|---|---|---|
| `fast` | `160x120` | `0.6` | `float32` | `0` (OFF) | `8 bits` / OFF | `flat` / `nearest` |
| `balanced` | `256x192` | `1.0` | `float32` | `0` (OFF) | `8 bits` / OFF | `gouraud` / `nearest` |
| `retro-ps1` | `320x240` | `1.0` | `fixed12_4` | `1 px` | `5 bits` (RGB555) / Bayer 4x4 | `gouraud` / `nearest` |
| `high` | `512x384` | `1.5` | `float32` | `0` (OFF) | `8 bits` / OFF | `gouraud` / `bilinear` |

---

## 3. Instalação

Requer **Python 3.10+**.

```bash
# Instalação padrão (inclui NumPy, Pillow, Matplotlib e Pygame-CE)
pip install -r requirements.txt

# Opcional: suíte de testes/lint (pytest, ruff), JIT em CPU (Numba) e backend OpenGL (ModernGL)
pip install -e ".[dev,fast,gpu]"
```

---

## 4. Execução e Linha de Comando (CLI)

```bash
# Visualizador interativo (cena, volumes 3D, stencil, ray cast diff e HUD clicável)
python -m aethelion
python main.py                                  # atalho equivalente na raiz
python main.py --fullscreen                     # inicia diretamente em tela cheia (F11 alterna)
python main.py --backend gpu-gl --preset high   # executa no backend GPU com preset de alta qualidade
python main.py --preset retro-ps1               # emula restrições de 5ª geração (Q12.4, snap, 5b + dither)

# Execução experimental e geração de artefatos (headless)
python main.py --experiment                     # conformidade (24) + 8 fases (A–H) + varreduras + relatório
python main.py --experiment --deterministic     # passo temporal fixo por quadro (100% reprodutível)
python main.py --conformance                    # executa apenas a suíte de conformidade OpenGL 1.4
python main.py --sv-sweeps                      # varreduras de resolução, complexidade (detail) e P_inf
python main.py --report                         # regenera gráficos PNG e RELATORIO.md a partir dos CSVs

# Testes automatizados (unitários, topologia 2-manifold, conformidade, CLI e integração)
pytest
```

### Principais Opções de Linha de Comando

* **Modos e Controle Experimental:** `--experiment`, `--conformance`, `--sv-sweeps`, `--report`, `-o / --outdir <dir>`, `--measure-every <seg>`, `--deterministic`, `--fixed-dt <seg>`, `--config <json>`, `--preset {fast,balanced,retro-ps1,high}`, `--backend {cpu-sw,cpu-fast,gpu-gl}`.
* **Janela:** `--fullscreen`, `--win-width <px>`, `--win-height <px>`, `--display-filter {nearest,bilinear}`.
* **Algoritmo de Sombra e Buffers:** `--algorithm {none,zpass,zfail}`, `--stencil-bits {1,2,3,4,5,8}`, `--no-wrap` (força `INCR`/`DECR` saturados), `--depth-bits <N>` (`0` = `float32`), `--extrude-len <L>`, `--infinite-extrude` ($P_{\text{inf}}, w=0$), `--no-depth-clamp`.
* **Geometria, Precisão e Sombreamento:** `--width <px>`, `--height <px>`, `--detail <0.25..4.0>`, `--precision {float32,float16,fixed16_16,fixed12_4}`, `--vertex-snap <px>`, `--shading {gouraud,flat,unlit}`, `--ambient <0..1>`, `--color-bits {1..8}`, `--dither`, `--no-texture`, `--texture-res <px>`, `--texture-filter {nearest,bilinear}`, `--fov <graus>`, `--near <z>`, `--far <z>`.

---

## 5. Visualizador Interativo

Na janela interativa, **ESPAÇO** inicia ou interrompe o roteiro automatizado de
96 s (8 fases × 12 s). Durante a execução é possível alterar qualquer parâmetro pelo
teclado ou clicando nos controles do painel lateral: cada intervenção é registrada em
`eventos.log` e marca os quadros seguintes no CSV como `origem_config=manual`.

O visualizador oferece **4 modos de vista** (tecla `V`):
1. **Cena**: imagem final com iluminação difusa/ambiente, texturas, quantização de cor e dithering.
2. **Volumes 3D + Silhueta**: sobreposição didática inspirada em Everitt & Kilgard (2002), destacando faces frontais do volume em âmbar, faces traseiras em azul-ciano e arestas de silhueta em ciano brilhante.
3. **Buffer de Stencil**: mapa de intensidade das contagens por pixel no buffer de stencil.
4. **Diferença (Ray Cast)**: comparação pixel a pixel contra o *ray casting* de referência (verde = concordância, vermelho = falso positivo de sombra, azul = falso negativo).

### Atalhos de Teclado e Mouse

| Controle | Ação |
|---|---|
| `ESPAÇO` | Iniciar / parar o roteiro experimental de 8 fases |
| `TAB` | Alternar backend em tempo real (`cpu-sw` → `cpu-fast` → `gpu-gl`) |
| `M` | Pausar / retomar movimento da luz (ativa renderização sob demanda) |
| `F11` / `F` / `Alt+Enter` | Alternar modo Tela Cheia |
| `1` / `2` / `0` | Selecionar algoritmo: `z-pass` / `z-fail` / sem sombra (`none`) |
| `W` | Alternar aritmética do stencil: `INCR_WRAP` (modular) / `INCR` (saturado) |
| `B` | Ciclar largura do stencil (`8`, `5`, `4`, `3`, `2`, `1` bits) |
| `D` | Ciclar largura do Z-buffer por fragmento (`float32`, `24`, `16`, `12`, `10` bits) |
| `P` | Ciclar precisão numérica (`float32`, `float16`, `fixed16_16`, `fixed12_4`) |
| `E` | Ciclar comprimento de extrusão finita (`40`, `15`, `7`, `3.5`, `80`) |
| `I` | Alternar extrusão homogênea no infinito ($P_{\text{inf}}, w=0$) |
| `R` / `Shift+R` / `]` / `[` | Aumentar / reduzir resolução interna do framebuffer |
| `V` | Alternar vista (**cena** / **volumes 3D + silhueta** / **stencil** / **diferença ray cast**) |
| `G` | Executar medição imediata contra a referência por *ray casting* |
| `C` | Executar a suíte de conformidade (24 testes) |
| `S` | Salvar captura PNG do quadro atual em `results/` |
| Mouse / Setas | Arrastar na viewport = orbitar câmera; roda do mouse = zoom na cena ou rolagem do painel |
| `ESC` | Sair da tela cheia ou encerrar a aplicação |

---

## 6. Saídas em `results/`

| Arquivo | Conteúdo |
|---|---|
| `conformidade.csv` | Resultado individual (`PASS`/`FAIL`) dos 24 testes de conformidade OpenGL 1.4 |
| `experimento.csv` | Telemetria quadro a quadro: tempos por estágio, contagens geométricas, memória (KB) e métricas F1 |
| `resumo.csv` | Agregação estatística por fase (mediana e IQR para tempos; média e desvio padrão para precisão/revocação/F1) |
| `varredura_sv_resolucao.csv` | Varredura paramétrica de resolução (`160x120` a `400x300`) |
| `varredura_sv_complexidade.csv` | Varredura de complexidade geométrica da cripta (`detail` de `0.5` a `2.0`) |
| `varredura_sv_extrusao.csv` | Comparação entre extrusão finita curta ($L=3{,}5$), longa ($L=40$) e infinita ($P_{\text{inf}}, w=0$) |
| `eventos.log` | Linha do tempo auditável (início, transições de fase, *overrides* manuais e encerramento) |
| `fase_<A..H>_<H1..H5>.png` | Captura representativa no instante médio ($u=0{,}5$) de cada fase experimental |
| `grafico_*.png` | 8 gráficos analíticos gerados via Matplotlib (custo, concordância, carga, stencil, memória e varreduras) |
| `RELATORIO.md` | Relatório consolidado gerado automaticamente com tabelas, dispersão, figuras e veredictos das hipóteses |

---

## 7. Arquitetura do Pacote

```text
aethelion/
├── main.py                          # Atalho de entrada na raiz (`python main.py`)
├── pyproject.toml                   # Metadados PEP 621, dependências opcionais, pytest e ruff
├── requirements.txt                 # Instalação editável (`-e .`)
├── METODOLOGIA.md                   # Modelagem matemática e desenho experimental detalhado
├── docs/                            # Artigo científico IEEE em LaTeX e artigos de referência (PDF)
├── results/                         # CSVs, logs, capturas de fases, gráficos e RELATORIO.md
├── tests/                           # Suíte de testes automatizados (pytest)
│   ├── conftest.py                  # Fixtures compartilhadas (configuração leve e cubo 2-manifold)
│   ├── test_conformance.py          # Verifica os 24 casos da suíte de conformidade OpenGL 1.4
│   ├── test_topology.py             # Verifica fechamento 2-manifold, orientação de normais e trajetória da luz
│   ├── test_shadow.py               # Verifica volumes finitos/infinitos (w=0), P_inf e concordância F1
│   └── test_backend_and_cli.py      # Verifica equivalência de backends, cache, CLI, presets e execução E2E
└── aethelion/
    ├── __init__.py                  # API pública do pacote
    ├── __main__.py                  # Suporte a `python -m aethelion`
    ├── cli.py                       # Parser de argumentos e orquestração dos modos de execução
    ├── core/                        # Motor gráfico, estágios do pipeline e backends CPU/GPU
    │   ├── settings.py              # Dataclasses SVSettings e SVStats, validação, matriz Bayer e PRESETS
    │   ├── mathutil.py              # Transformações homogêneas 4x4, look_at, perspectiva finita e P_inf
    │   ├── precision.py             # Emulação de precisão (float16, fixed16_16, fixed12_4) e Z-buffer N bits
    │   ├── mesh.py                  # Estruturas Solid/Prop, soldagem espacial (weld) e topologia half-edge
    │   ├── clipping.py              # Recorte homogêneo Sutherland-Hodgman contra o plano near
    │   ├── raster.py                # Varredura de triângulos com regra top-left orientada (NumPy e Numba JIT)
    │   ├── stencil.py               # Emulador do estágio de stencil OpenGL 1.4 (1..8 bits, saturado e wrap)
    │   ├── shadow.py                # Detecção de silhueta, extrusão finita/homogênea (w=0) e point_in_volume
    │   ├── texture.py               # Texturas procedurais (checker, bricks, gradient, noise) nearest/bilinear
    │   ├── overlay.py               # Visualização didática 3D dos volumes de sombra e arestas de silhueta
    │   ├── backend.py               # RendererProtocol, detecção de backends e fábrica create_renderer
    │   ├── base_renderer.py         # Infraestrutura comum de cena, transformação, caches e memória
    │   ├── sv_renderer.py           # Backends CPU por software (SVRenderer e FastCPURenderer)
    │   ├── gpu_renderer.py          # Backend GPU via ModernGL (GLRenderer em 3 FBOs offscreen)
    │   └── shaders.py               # Shaders GLSL 330 (geometria, volumes de sombra e composição)
    ├── scenes/
    │   ├── primitives.py            # Geradores de sólidos 2-manifold fechados (box, cylinder, sphere, torus)
    │   └── crypt.py                 # Montagem procedural da cripta e trajetória determinística da luz
    ├── evaluation/
    │   ├── conformance.py           # Suíte de conformidade de 24 testes (OpenGL 1.4 Seções 4.1.5, 3.5 e 2.11)
    │   ├── truth.py                 # Verdade de referência por ray casting (Möller-Trumbore) e métricas F1
    │   ├── phases.py                # Definição das 8 fases (A–H), hipóteses H1–H5, câmeras e esquema CSV
    │   ├── runner.py                # Executor experimental (ExperimentRunner) com desconto de overhead
    │   ├── sweeps.py                # Varreduras paramétricas de resolução, complexidade e extrusão P_inf
    │   ├── plots.py                 # Gerador dos 8 gráficos analíticos em Matplotlib
    │   └── report.py                # Avaliador automático de veredictos e gerador de RELATORIO.md
    └── ui/
        ├── theme.py                 # Paleta de cores, dimensões, ciclos de parâmetros e resumo de atalhos
        ├── camera.py                # Controlador de câmera orbital esférica (OrbitCamera)
        ├── widgets.py               # Primitivos de UI imediata, rolagem vertical e hitboxes (UIPanelCanvas)
        ├── panel.py                 # Desenho do painel lateral interativo (HUD) e barra de status
        └── viewer.py                # Controlador principal da janela Pygame, render sob demanda e eventos
```
