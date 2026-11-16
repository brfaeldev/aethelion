# Metodologia

## 1. Objetivo

Medir como **recursos e limites da arquitetura de rasterização** afetam a
geração de sombras volumétricas por stencil, em termos de **robustez**,
**precisão geométrica** e **custo computacional**.

O objeto de estudo não é a cena. A cena é o banco de ensaio.

## 2. Pergunta de pesquisa

> Como a arquitetura do buffer (largura do stencil, disponibilidade de
> operações de transbordo circular, largura do buffer de profundidade) e a
> variante do algoritmo (z-pass versus z-fail) afetam a robustez, a precisão
> geométrica e o custo computacional da geração de sombras volumétricas?

## 3. Hipóteses

| # | Hipótese | Fase |
|---|---|---|
| H1 | z-pass e z-fail concordam enquanto a câmera está fora de todos os volumes; a divergência acompanha o número de volumes que contêm a câmera | C vs D |
| H2 | z-fail custa sistematicamente mais que z-pass, por processar e preencher as tampas frontal e traseira | A vs B |
| H3 | Sem operações de transbordo circular, a saturação de um contador estreito produz sombra incorreta; com `INCR_WRAP`/`DECR_WRAP`, o erro desaparece na mesma cena | E vs F |
| H4 | Reduzir a largura do buffer de profundidade degrada a máscara de sombra de forma distinta da degradação da visibilidade | G vs B |
| H5 | Extrusão finita insuficiente trunca o volume e reduz a revocação sem afetar a precisão na mesma proporção | H vs B |

## 4. Enquadramento no Eixo 7 — Arquitetura de Consoles Antigos

O eixo trata de como **recursos e limitações do pipeline gráfico** condicionam a
renderização. Sombras volumétricas por stencil são o caso didático exato disso,
porque o algoritmo **não funciona** sem um conjunto específico de recursos de
arquitetura:

* **Buffer de stencil, e sua largura.** O algoritmo é um contador por pixel. Um
  contador estreito satura e a sombra sai errada. Essa é uma limitação de
  **memória por pixel**, não de aritmética — exatamente o tipo de restrição que
  caracteriza hardware de gerações anteriores.
* **Operações de transbordo circular.** `INCR_WRAP` e `DECR_WRAP` foram
  promovidas ao núcleo do OpenGL em 1.4; antes disso eram extensão. Um programa
  escrito contra uma implementação sem elas precisa de outra estratégia. A
  disponibilidade de uma operação é um fato de arquitetura, e aqui ela é medida.
* **Buffer de profundidade.** Tanto a existência quanto a largura. Z-fail
  depende de testar profundidade na direção contrária, e a resolução do buffer
  determina onde a sombra começa a falhar.
* **Recorte no plano próximo.** É a razão de z-fail existir: z-pass quebra
  quando o volume é recortado pelo plano próximo da câmera.
* **Taxa de preenchimento.** Volumes de sombra são enormes em área de tela.

A comparação entre as duas variantes, instrumentada, é portanto uma investigação
sobre **arquitetura**, e não sobre estética.

## 5. Instrumento e sua validação

O rasterizador é inteiramente por software. Antes de ser usado como instrumento
de medida, ele é submetido a uma **suíte de conformidade** que verifica o
subconjunto de comportamento especificado em OpenGL 1.4 para o estágio de
stencil, mais a regra de preenchimento de triângulos e o recorte no plano
próximo. O resultado é gravado em `conformidade.csv`.

A regra de preenchimento é verificada explicitamente porque dupla cobertura de
uma aresta compartilhada incrementaria o contador de stencil duas vezes, criando
sombra espúria: é um defeito do instrumento que se disfarçaria de resultado.

## 6. Verdade de referência

A referência é obtida por **ray casting**: para cada pixel visível, um raio é
lançado do ponto de superfície até a fonte de luz e testado contra a geometria
oclusora. É independente do algoritmo em avaliação, o que é a condição para que
precisão, revocação e F1 signifiquem alguma coisa.

Por custo, a referência é amostrada periodicamente (padrão: a cada 2,5 s), não
em todos os quadros. Isso está declarado e os quadros medidos são marcados na
coluna `medido` do CSV.

## 7. Indicadores

| Indicador | O que revela |
|---|---|
| `ms_geometry`, `ms_silhouette`, `ms_stencil` | onde o custo está, por etapa |
| `tris_volume`, `tris_volume_drawn` | carga geométrica do algoritmo |
| `frag_stencil` | carga de preenchimento (o gargalo clássico) |
| `pixels_shadow` / `pixels_visible` | extensão da região sombreada |
| `precision`, `recall`, `f1`, `error_rate` | concordância com a referência |
| `volumes_with_camera` | variável explicativa de H1 |
| `stencil_saturation`, `stencil_wrap_events`, `stencil_max` | H3 |
| `backend`, e a configuração completa | rastreabilidade de cada medição |

## 8. Procedimento

1. Suíte de conformidade.
2. Roteiro de 8 fases, 12 s cada (96 s), com trajetória de câmera determinística
   por fase. Cada fase isola uma hipótese.
3. Registro de todos os quadros em `experimento.csv`; agregação por fase em
   `resumo.csv`; eventos em `eventos.log`.
4. Alterações manuais durante a execução são permitidas e registradas como
   eventos, com as linhas afetadas marcadas como `origem_config=manual`.
5. Geração automática de gráficos e de `RELATORIO.md`.

## 9. Ameaças à validade, declaradas

1. **Backend.** Todos os tempos vêm do rasterizador em software (`cpu-sw`).
   Não descrevem hardware real. Apenas tendências relativas entre fases são
   válidas. O backend é registrado em cada linha dos CSVs.
2. **Extrusão finita.** Os volumes são extrudados por um comprimento finito e
   configurável, não infinito com plano distante no infinito. É um desvio em
   relação à formulação robusta, e por isso é tratado como **fator
   experimental** (fase H), não como aproximação silenciosa.
3. **Ambiente de medição.** Concorrência de CPU introduz ruído. Fases curtas
   reduzem, mas não eliminam, o efeito.
4. **Modelo de luz.** Fonte pontual, oclusão binária. Não há penumbra.

## 10. Referências

1. C. Everitt, M. J. Kilgard. "Practical and Robust Stenciled Shadow Volumes for
   Hardware-Accelerated Rendering." 2002. https://arxiv.org/abs/cs/0301002
   — origem das variantes z-pass e z-fail, do tratamento de tampas, do problema
   do plano próximo e da necessidade de operações de transbordo circular.
2. WSCG 2003, "An online system which approved OpenGL 1.4 standard."
   http://wscg.zcu.cz/WSCG2003/Papers_2003/D19.pdf
   — prática de validação por conjunto de testes de conformidade, adotada aqui
   para validar o rasterizador antes de usá-lo como instrumento de medida.
3. T. Akenine-Möller, E. Haines, N. Hoffman. *Real-Time Rendering*, 4ª ed.,
   CRC Press, 2018. — organização do pipeline e rasterização.
4. J. F. Blinn. "A Trip Down the Graphics Pipeline: The Homogeneous Perspective
   Transform." *IEEE CG&A*, 13(3):75–80, 1993. — transformação de perspectiva.
