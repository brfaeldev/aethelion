# Aethelion — Sombras Volumétricas por Stencil

Estudo experimental sobre **como os recursos e os limites da arquitetura de
rasterização condicionam a geração de sombras volumétricas**, construído sobre
um rasterizador próprio, inteiramente por software.

Eixo 7 — Arquitetura de Consoles Antigos.

A pergunta, as hipóteses, o enquadramento no eixo e as limitações declaradas
estão em [`METODOLOGIA.md`](METODOLOGIA.md).

---

## Instalação

```bash
pip install -r requirements.txt
```

## Execução

```bash
python main.py                  # tela única: cena, HUD e barra de estado
python main.py --experiment     # roteiro completo sem janela + relatório
python main.py --conformance    # apenas a suíte de conformidade
python main.py --report         # gráficos e RELATORIO.md a partir dos CSVs
python main.py --legacy-sweeps  # varreduras retrô do estudo anterior
```

Na janela, **ESPAÇO** inicia o roteiro de 96 s. Durante a execução é possível
alterar qualquer parâmetro: cada alteração vira um evento em `eventos.log` e
marca as linhas seguintes como `origem_config=manual`.

### Teclas

| Tecla | Ação |
|---|---|
| `ESPAÇO` | iniciar / parar o experimento |
| `1` / `2` / `0` | z-pass / z-fail / sem sombra |
| `W` | alterna `INCR_WRAP` / `INCR` |
| `B` | largura do stencil (8, 5, 4, 3, 2, 1 bits) |
| `D` | largura do z-buffer (float32, 24, 16, 12, 10) |
| `P` | precisão numérica |
| `E` | comprimento de extrusão |
| `R` | resolução interna |
| `V` | vista: cena / stencil / diferença contra a referência |
| `G` | medir agora contra o ray casting |
| `C` | rodar a suíte de conformidade |
| `S` | captura PNG |
| setas | orbitar (fora do roteiro) |
| `ESC` | sair |

Na vista de diferença: **verde** = acerto, **vermelho** = falso positivo,
**azul** = falso negativo.

---

## Saídas em `results/`

| Arquivo | Conteúdo |
|---|---|
| `conformidade.csv` | PASS/FAIL por teste de conformidade |
| `experimento.csv` | uma linha por quadro, com configuração completa |
| `resumo.csv` | agregação por fase (mediana de tempos, média de métricas) |
| `eventos.log` | início, transições de fase, overrides manuais, encerramento |
| `fase_*.png` | captura do meio de cada fase |
| `grafico_*.png` | custo por etapa, concordância, carga, saturação |
| `RELATORIO.md` | relatório preenchido com os números medidos |

---

## As oito fases

| Fase | Hipótese | O que isola |
|---|---|---|
| A | H2 | z-pass, câmera externa — linha de base |
| B | H2 | z-fail, mesma trajetória — custo das tampas |
| C | H1 | z-pass com a câmera atravessando volumes — artefato esperado |
| D | H1 | z-fail na mesma trajetória — robustez esperada |
| E | H3 | stencil de 3 bits **sem** wrap, zona de alta sobreposição |
| F | H3 | stencil de 3 bits **com** wrap, mesma cena |
| G | H4 | buffer de profundidade de 12 bits, vista rasante |
| H | H5 | extrusão curta — volume truncado |

---

## Desempenho esperado

O rasterizador é por software, por decisão de projeto: o objetivo é
**instrumentar** cada estágio, não atingir taxas altas. A 256×192 com a cripta
em detalhe 1.0, espere da ordem de **1 a 3 quadros por segundo**. Isso não
afeta o experimento, que é conduzido por **tempo de parede**: as fases duram o
que foi especificado, com mais ou menos quadros amostrados.

Para uma demonstração mais fluida:

```bash
python main.py --width 160 --height 120 --detail 0.5
```

Um achado do estudo anterior deste projeto era que o custo do pipeline era
dominado pelo **custo por primitiva**, não pelo preenchimento. Shadow volumes
invertem essa relação: compare `frag_stencil` com `tris_volume` nos CSVs.

---

## Estrutura

```
main.py                   CLI
aethelion/
  raster.py               varredura com regra de preenchimento exclusiva
  stencil.py              unidade de stencil (subconjunto OpenGL 1.4)
  scene_crypt.py          cena de ensaio: malhas fechadas + topologia
  shadow.py               silhueta, extrusão, tampas, ponto-em-volume
  sv_renderer.py          renderizador: geometria, stencil, composição
  truth.py                referência por ray casting e métricas
  conformance.py          suíte de conformidade (24 testes)
  scenario.py             roteiro de fases, logs e CSVs
  report.py               gráficos e RELATORIO.md
  viewer.py               aplicação de tela única (pygame)
  mathutil.py             álgebra 4x4 e perspectiva
  pipeline.py             pipeline retrô do estudo anterior
  framebuffer.py          quantização, dithering, paleta
  texture.py              texturas procedurais
  geometry.py             primitivas do estudo anterior
  gpu_pipeline.py         backend ModernGL do estudo anterior
  experiments.py          varreduras retrô
```

## Trabalho futuro declarado

O backend ModernGL existente (`gpu_pipeline.py`) ainda não implementa o caminho
de stencil em hardware. Portá-lo permitiria contrastar o comportamento do
algoritmo em uma implementação real de OpenGL com a emulação controlada feita
aqui — inclusive verificando se as operações de transbordo circular se comportam
conforme a especificação de 1.4 no driver em uso.
