# Metodologia Experimental e Modelagem Matemática

## 1. Objetivo

Medir como **recursos e limites da arquitetura de rasterização** afetam a
geração de sombras volumétricas por stencil, em termos de **robustez**,
**precisão geométrica**, **consumo de memória** e **custo computacional**.

O objeto de estudo não é a cena. A cena procedural (cripta *two-manifold*) é o
banco de ensaio de pior caso.

## 2. Pergunta de pesquisa

> Como a arquitetura do buffer (largura do stencil, disponibilidade de
> operações de transbordo circular `INCR_WRAP`/`DECR_WRAP`, largura do buffer
> de profundidade quantizada por fragmento) e a variante do algoritmo (`z-pass`
> versus `z-fail` com extrusão finita ou homogênea no infinito $P_{\text{inf}}$)
> afetam a robustez, a precisão geométrica, o consumo de memória e o custo
> computacional da geração de sombras volumétricas?

## 3. Hipóteses

| # | Hipótese | Fase |
|---|---|---|
| H1 | `z-pass` e `z-fail` concordam enquanto a câmera está fora de todos os volumes; a divergência acompanha o número de volumes que contêm a câmera | C vs D |
| H2 | `z-fail` custa sistematicamente mais que `z-pass`, por construir e rasterizar as tampas frontal e traseira do volume fechado | A vs B |
| H3 | Sem operações de transbordo circular, a saturação de um contador estreito produz sombra incorreta; com `INCR_WRAP`/`DECR_WRAP`, o erro desaparece na mesma cena | E vs F |
| H4 | Reduzir a largura do buffer de profundidade (quantizado por fragmento) degrada a máscara de sombra em vistas rasantes | G vs B |
| H5 | Extrusão finita insuficiente trunca o volume e reduz a revocação, enquanto a extrusão suficiente ou homogênea no infinito ($P_{\text{inf}}, w=0$) preserva o volume | H vs B |

## 4. Modelagem Matemática do Pipeline e dos Volumes de Sombra

### 4.1 Detecção de Silhueta e Extrusão Homogênea (Everitt & Kilgard, 2002)

Seja uma malha fechada (*2-manifold*) e uma fonte de luz pontual em coordenadas
homogêneas $L = \langle L_x, L_y, L_z, 1 \rangle$. Uma face triangular com normal
externa $\mathbf{n}_f$ e baricentro $\mathbf{c}_f$ está voltada para a luz quando:

$$\mathbf{n}_f \cdot (\mathbf{L} - \mathbf{c}_f) > 0$$

Uma aresta orientada $(v_a, v_b)$ compartilhada pelas faces adjacentes $f_1$ e
$f_2$ pertence ao contorno de silhueta se e somente se $f_1$ está voltada para a
luz e $f_2$ não está.

Para fechar o volume de sombra de forma robusta no infinito (Everitt & Kilgard,
2002, Seção 4.2), cada vértice $V = \langle V_x, V_y, V_z, 1 \rangle$ das faces
voltadas para a luz é extrudado na direção oposta à luz como um ponto direcional
no infinito com coordenada homogênea $w = 0$:

$$V_\infty = \langle V_x - L_x, \; V_y - L_y, \; V_z - L_z, \; 0 \rangle$$

Para comparação experimental com a aproximação finita de Carmack (Fase H), o
sistema também modela a extrusão finita por distância $L_{\text{ext}}$:

$$V_{\text{fin}} = \left\langle \mathbf{V} + L_{\text{ext}} \frac{\mathbf{V} - \mathbf{L}}{\|\mathbf{V} - \mathbf{L}\|}, \; 1 \right\rangle$$

### 4.2 Matriz de Perspectiva Finita vs. Infinita ($P_{\text{inf}}$)

Quando vértices no infinito ($w = 0$) são projetados, a matriz de perspectiva
clássica com plano distante finito `far` descartaria a tampa traseira. Tomando o
limite $\text{far} \to \infty$ com folga numérica $\epsilon$ (Everitt & Kilgard,
2002, Seção 3.2), obtém-se:

$$P_{\text{inf}} = \begin{bmatrix} \frac{\cot(\text{fovy}/2)}{\text{aspect}} & 0 & 0 & 0 \\ 0 & \cot(\text{fovy}/2) & 0 & 0 \\ 0 & 0 & \epsilon - 1 & (\epsilon - 2)\,\text{near} \\ 0 & 0 & -1 & 0 \end{bmatrix}$$

Complementarmente, o rasterizador implementa *depth clamping* (`NV_depth_clamp`),
saturando $z_{\text{ndc}} \leftarrow \min(z_{\text{ndc}}, 1)$ nos fragmentos de
volumes de sombra em `z-fail`.

### 4.3 Regra de Preenchimento Exclusiva (Top-Left Orientada)

Para evitar dupla contagem ou furos em arestas compartilhadas entre triângulos do
volume de sombra, as três funções de aresta orientadas pelo sinal da área
$s = \text{sign}(A)$ são avaliadas diretamente:

$$e_i(p_x, p_y) = s \cdot \det\begin{bmatrix} x_{j} - p_x & x_{k} - p_x \\ y_{j} - p_y & y_{k} - p_y \end{bmatrix}$$

Um centro de pixel $(p_x, p_y)$ pertence ao triângulo se, para $i \in \{0, 1, 2\}$:

$$(e_i > 0) \;\lor\; \bigl(e_i = 0 \;\land\; \text{TopLeft}(s\,\Delta x_i, s\,\Delta y_i)\bigr)$$

onde $\text{TopLeft}(dx, dy) \iff (dy < 0) \lor (dy = 0 \land dx > 0)$.

### 4.4 Operações de Stencil de $b$ Bits e Quantização de Profundidade por Fragmento

Para um buffer de stencil de $b \in [1, 8]$ bits com valor máximo $M = 2^b - 1$:

$$\text{INCR}(s) = \min(s + 1, M), \qquad \text{DECR}(s) = \max(s - 1, 0)$$

$$\text{INCR\_WRAP}(s) = (s + 1) \bmod (M + 1), \qquad \text{DECR\_WRAP}(s) = (s - 1) \bmod (M + 1)$$

Para um Z-buffer de $D$ bits, cada fragmento interpolado $z_{\text{ndc}} \in [-1, 1]$
é quantizado antes do teste de profundidade:

$$z_q = \frac{2}{2^D - 1}\,\text{round}\!\left(\frac{z_{\text{ndc}} + 1}{2}\,(2^D - 1)\right) - 1$$

## 5. Enquadramento no Eixo 7 — Arquitetura de Consoles Antigos

O eixo investiga como **recursos e limitações do pipeline gráfico** condicionam a
renderização. Sombras volumétricas por stencil exigem um conjunto específico de
recursos arquiteturais:

* **Buffer de stencil e sua largura ($1\dots 8$ bits)**: contadores estreitos
  saturam sob alta sobreposição de volumes (limitação de memória por pixel).
* **Operações de transbordo circular (`INCR_WRAP`/`DECR_WRAP`)**: promovidas de
  `EXT_stencil_wrap` ao núcleo na especificação **OpenGL 1.4** (Segal & Akeley,
  2002).
* **Buffer de profundidade por fragmento**: determina a precisão da passagem de
  geometria e da inversão do teste de profundidade em `z-fail`.
* **Recorte no plano próximo e distante**: razão da transição de `z-pass` para
  `z-fail` com $P_{\text{inf}}$ / *depth clamping*.
* **Taxa de preenchimento (*fill rate*)**: os quadriláteros extrudados dominam o
  número de fragmentos processados (`frag_stencil`).

## 6. Instrumento, Validação e Verdade de Referência

1. **Validação de Conformidade (`conformance.py`)**: 24 testes automatizados
   verificam todas as funções, operações, máscaras e ordens de teste da Seção
   4.1.5 da especificação OpenGL 1.4 (Segal & Akeley, 2002), a regra top-left sem
   furos ou dupla cobertura e o recorte homogêneo no plano próximo.
2. **Verdade de Referência por Ray Casting (`truth.py`)**: lança raios de sombra
   (Möller-Trumbore) de cada pixel visível até a luz pontual e computa Precisão,
   Revocação, F1-Score e Taxa de Erro. O tempo de CPU gasto no ray casting é
   isolado e descontado dos cronômetros do experimento.

## 7. Indicadores e Perfilamento

| Indicador | O que revela |
|---|---|
| `ms_geometry`, `ms_silhouette`, `ms_stencil` | custo por estágio do pipeline (mediana e IQR) |
| `tris_volume`, `tris_volume_drawn` | carga geométrica gerada pelas silhuetas e tampas |
| `frag_stencil` | carga de preenchimento (*fill rate*) no estágio de stencil |
| `mem_buffers_kb`, `mem_geometry_kb`, `mem_total_kb` | perfilamento de memória dos buffers por pixel e malhas |
| `precision`, `recall`, `f1`, `error_rate` | concordância objetiva contra o ray casting (média e DP) |
| `volumes_with_camera` | variável explicativa de H1 (medida fora do cronômetro) |
| `stencil_saturation`, `stencil_wrap_events`, `stencil_max` | contadores de saturação vs. aritmética modular (H3) |

## 8. Referências

1. C. Everitt, M. J. Kilgard. *"Practical and Robust Stenciled Shadow Volumes for
   Hardware-Accelerated Rendering."* NVIDIA Corporation, 2002.
   https://arxiv.org/abs/cs/0301002 — formulação das variantes `z-pass` e
   `z-fail`, necessidade das tampas frontal/traseira, matriz $P_{\text{inf}}$ com
   extrusão homogênea ($w=0$), *depth clamping* e uso de `INCR_WRAP`/`DECR_WRAP`.
2. M. Segal, K. Akeley. *"The OpenGL Graphics System: A Specification (Version
   1.4)."* Silicon Graphics, Inc., 2002 — especificação normativa do estágio de
   stencil (Seção 4.1.5), promoção das operações `INCR_WRAP` e `DECR_WRAP` ao
   núcleo do OpenGL 1.4 e regras de rasterização de polígonos (Seção 3.5).
3. M. Li, M. Magnor, H.-P. Seidel. *"Online Accelerated Rendering of Visual Hulls
   in Real Scenes."* WSCG 2003 (`D19.pdf`), pp. 290–297, 2003 — extração de
   cones de silhueta, acumulação multi-passagem no buffer de stencil, uso de
   recursos recém-aprovados do padrão OpenGL 1.4 para redução de passagens e
   aplicação de *polygon offset* contra artefatos de precisão em bordas.
4. T. Akenine-Möller, E. Haines, N. Hoffman. *Real-Time Rendering*, 4ª ed.,
   CRC Press, 2018 — organização do pipeline e rasterização por funções de aresta.
5. J. F. Blinn. *"A Trip Down the Graphics Pipeline: The Homogeneous Perspective
   Transform."* *IEEE CG&A*, 13(3):75–80, 1993 — transformação homogênea e recorte.
