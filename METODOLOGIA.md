# Metodologia Experimental e Modelagem Matemática

## 1. Objetivo

Medir como **recursos e limites da arquitetura de rasterização** — largura do
buffer de stencil, disponibilidade de operações de transbordo circular, largura
do buffer de profundidade e precisão numérica — condicionam a geração de sombras
volumétricas por stencil, no contexto do **Eixo 7 (arquitetura de consoles
antigos)**, em termos de **robustez**, **precisão geométrica**, **consumo de
memória** e **custo computacional**.

O método de referência do trabalho é o de Everitt e Kilgard [1], que combina o
plano distante no infinito ($P_\infty$), a extrusão de polígonos em coordenadas
homogêneas ($w = 0$) e o esquema `z-fail` com contadores de transbordo circular.
A cena procedural (cripta *two-manifold* fechada) funciona como banco de ensaio
de pior caso, e cada quadro é avaliado contra uma verdade de referência
independente por *ray casting* (Möller–Trumbore [10]).

## 2. Pergunta de Pesquisa

> Como a largura do stencil, a disponibilidade de `INCR_WRAP`/`DECR_WRAP`, a
> largura do buffer de profundidade, a precisão numérica e a variante do
> algoritmo (`z-pass`, `z-fail` finito ou `z-fail` no infinito $P_\infty, w=0$)
> afetam a robustez, a precisão geométrica, a memória e o custo por quadro?

## 3. Hipóteses Experimentais

| # | Hipótese | Fases |
|---|---|---|
| **H1** | `z-pass` e `z-fail` concordam com a câmera fora dos volumes ($V_c = 0$), e divergem quando ela está dentro ($V_c > 0$) | **C** vs. **D** (e **A** vs. **B**) |
| **H2** | `z-fail` custa sistematicamente mais que `z-pass`, por construir e rasterizar as tampas frontal e traseira do volume fechado | **A** vs. **B** (e **C** vs. **D**) |
| **H3** | Sem transbordo circular, um stencil estreito (3 bits) satura sob alta sobreposição e produz sombra errada; com `INCR_WRAP`/`DECR_WRAP`, o erro desaparece na mesma cena | **E** vs. **F** |
| **H4** | Reduzir a largura do buffer de profundidade por fragmento (12 bits) degrada a máscara de sombra em vistas rasantes | **G** vs. **B** |
| **H5** | Extrusão finita curta ($L = 3{,}5$) trunca o volume e reduz a revocação, enquanto a extrusão suficiente ($L = 40$) ou homogênea no infinito ($P_\infty, w=0$) preserva o volume | **H** vs. **B** / Varredura |

## 4. Modelagem Matemática do Pipeline e dos Volumes de Sombra

### 4.1 Transformação, Projeção ($P$ e $P_\infty$) e Restrição de Precisão ($Q_m$)

Um vértice homogêneo $\mathbf{p} = (x, y, z, 1)^\top$ é levado ao espaço de
recorte por:

$$\mathbf{c} = \mathbf{P}\,\mathbf{V}\,\mathbf{M}\,\mathbf{p}$$

onde $\mathbf{V}$ é a matriz *look-at* (olhando para $-z$) com
$\mathbf{f} = \widehat{\mathbf{t} - \mathbf{e}}$,
$\mathbf{s} = \widehat{\mathbf{f} \times \mathbf{u}_0}$ e
$\mathbf{u} = \mathbf{s} \times \mathbf{f}$:

$$\mathbf{V} = \begin{bmatrix} \mathbf{s}^\top & -\mathbf{s}\cdot\mathbf{e} \\ \mathbf{u}^\top & -\mathbf{u}\cdot\mathbf{e} \\ -\mathbf{f}^\top & \mathbf{f}\cdot\mathbf{e} \\ \mathbf{0}^\top & 1 \end{bmatrix}$$

Com $k = \cot(\theta/2)$, aspecto $a$ e planos próximo $n$ e distante $f$, a
matriz de perspectiva clássica (Blinn [6]) e o limite $f \to \infty$ com folga
numérica $\epsilon = 2{,}4 \times 10^{-7}$ (Everitt e Kilgard [1]) são:

$$\mathbf{P} = \begin{bmatrix} \frac{k}{a} & 0 & 0 & 0 \\ 0 & k & 0 & 0 \\ 0 & 0 & \frac{f+n}{n-f} & \frac{2fn}{n-f} \\ 0 & 0 & -1 & 0 \end{bmatrix}, \qquad \mathbf{P}_\infty = \begin{bmatrix} \frac{k}{a} & 0 & 0 & 0 \\ 0 & k & 0 & 0 \\ 0 & 0 & \epsilon - 1 & (\epsilon - 2)n \\ 0 & 0 & -1 & 0 \end{bmatrix}$$

Para um ponto direcional no infinito $(\mathbf{d}, 0)$ com $z_v < 0$,
$\mathbf{P}_\infty$ produz $z_c = (\epsilon - 1)z_v$ e $w_c = -z_v$, logo
$z_{\text{ndc}} = 1 - \epsilon \le 1$: o infinito é mapeado dentro do volume de
visão e a tampa traseira não é descartada pelo plano distante.

A divisão perspectiva e o mapeamento para coordenadas de tela (com $y_s$
crescendo para baixo) são:

$$(x_n, y_n, z_n) = \frac{(c_x, c_y, c_z)}{c_w}, \qquad x_s = \left(\frac{x_n}{2} + \frac{1}{2}\right)W, \qquad y_s = \left(\frac{1}{2} - \frac{y_n}{2}\right)H$$

A restrição de precisão aritmética é um operador $Q_m$ aplicado a $\mathbf{c}$ e
a $(x_n, y_n, z_n)$: para ponto fixo com $q$ bits fracionários e faixa $R$,

$$Q(x) = \frac{\text{round}\bigl(\text{clip}(x, \pm R)\,2^q\bigr)}{2^q}$$

com $(q, R) = (16, 32767)$ em `fixed16_16` e $(4, 2047)$ em `fixed12_4` (estilo
PS1/GTE); `float16` converte para meia precisão IEEE 754. O *snap* opcional de
vértices em grade de passo $g$ é $x_s \leftarrow g\,\text{round}(x_s / g)$.

### 4.2 Recorte Homogêneo no Plano Próximo (Sutherland–Hodgman)

Triângulos que cruzam o plano próximo são recortados em espaço de recorte 4D por
Sutherland–Hodgman [8] contra os semiespaços:

$$d_1 = c_z + c_w \ge 0, \qquad d_2 = c_w - \varepsilon \ge 0 \quad (\varepsilon = 10^{-5})$$

Para uma aresta $(i, j)$ com sinais distintos, o vértice de interseção e seus
atributos são interpolados linearmente por $\mathbf{v} = \mathbf{v}_i + t(\mathbf{v}_j - \mathbf{v}_i)$
com $t = d_i / (d_i - d_j)$, e o polígono convexo resultante é triangulado em
leque.

### 4.3 Rasterização por Funções de Aresta e Regra Top-Left (Pineda)

Sejam $(x_i, y_i)$, $i \in \{0, 1, 2\}$, os vértices em tela e
$A = (x_1 - x_0)(y_2 - y_0) - (x_2 - x_0)(y_1 - y_0)$ a área com sinal, com
$s = \text{sgn}(A)$. Para o centro de pixel $\mathbf{p} = (p_x, p_y)$ e
permutações cíclicas $(i, j, k)$ de $(0, 1, 2)$, as funções de aresta orientadas
(Pineda [9]; Akenine-Möller et al. [5]) são:

$$e_i(\mathbf{p}) = s \begin{vmatrix} x_j - p_x & x_k - p_x \\ y_j - p_y & y_k - p_y \end{vmatrix} = a_i p_x + (b_i p_y + c_i)$$

avaliadas de forma separável 1D+1D por *broadcast* ($O(w + h)$ multiplicações)
com coordenadas baricêntricas $\lambda_i = e_i / |A|$. Para garantir cobertura
única em arestas compartilhadas (Seção 3.5 do OpenGL 1.4 [3]), o pixel pertence
ao triângulo se:

$$\bigwedge_{i=0}^{2} \Bigl[\, e_i > 0 \;\lor\; \bigl(e_i = 0 \;\land\; \text{TL}(s\,\Delta x_i, s\,\Delta y_i)\bigr) \Bigr]$$

com $\text{TL}(dx, dy) \iff dy < 0 \lor (dy = 0 \land dx > 0)$. Como $y_s$
cresce para baixo, faces frontais (anti-horárias em NDC) têm $A < 0$ em tela. A
interpolação de atributos é corrigida pela perspectiva e a profundidade $z_n$ é
interpolada linearmente em tela:

$$\hat{\lambda}_i = \frac{\lambda_i / w_i}{\sum_j \lambda_j / w_j}, \qquad a = \sum_i \hat{\lambda}_i a_i, \qquad z = \sum_i \lambda_i z_{n,i}$$

O sombreamento por vértice é Lambert com atenuação quadrática
$\ell = \max(0, \mathbf{n}\cdot\hat{\mathbf{l}}) / (1 + 0{,}035\|\mathbf{L}-\mathbf{v}\|^2)$;
a cor composta de um pixel é $\mathbf{b}(\alpha + (1 - \alpha)\ell)$ fora da
sombra e $\mathbf{b}\alpha$ onde o stencil indica sombra ($s \neq 0$), com
albedo $\mathbf{b}$ e luz ambiente $\alpha = 0{,}17$.

### 4.4 Construção dos Volumes de Sombra (Crow, Heidmann, Carmack, Everitt & Kilgard)

Seja uma malha 2-manifold fechada e uma fonte de luz pontual $\mathbf{L}$. Uma
face $f$ com normal externa $\mathbf{n}_f = (\mathbf{v}_1 - \mathbf{v}_0) \times (\mathbf{v}_2 - \mathbf{v}_0)$
e baricentro $\mathbf{c}_f$ é iluminada quando:

$$\mathbf{n}_f \cdot (\mathbf{L} - \mathbf{c}_f) > 0$$

Uma aresta orientada $(a, b)$ da face $f_1$, com face oposta $f_2$, pertence à
silhueta se $f_1$ é iluminada e $f_2$ não é (Crow [7]). Cada vértice $\mathbf{V}$
é extrudado na direção oposta à luz, de forma finita por distância $\ell$
(Carmack) ou como ponto direcional no infinito com $w = 0$ (Everitt e Kilgard [1]):

$$\mathbf{V}_{\text{fin}} = \left\langle \mathbf{V} + \ell\,\frac{\mathbf{V} - \mathbf{L}}{\|\mathbf{V} - \mathbf{L}\|}, \; 1 \right\rangle, \qquad \mathbf{V}_\infty = \left\langle \frac{\mathbf{V} - \mathbf{L}}{\|\mathbf{V} - \mathbf{L}\|}, \; 0 \right\rangle$$

Normalizar a direção é projetivamente invariante em $\mathbb{P}^3$ e evita
estouro em ponto fixo e `float16`. Cada aresta de silhueta $(a, b)$ gera dois
triângulos laterais $(b, a, a')$ e $(b, a', b')$ (sentido oposto ao da tampa
frontal para manter normais externas consistentes). O `z-fail` acrescenta a
tampa frontal $(v_0, v_1, v_2)$ para cada face iluminada e a tampa traseira
extrudada $(v'_0, v'_2, v'_1)$ com enrolamento invertido. Com $S$ o conjunto de
arestas de silhueta e $F_L$ o de faces iluminadas:

$$T_{\text{vol}}^{z\text{-pass}} = 2|S|, \qquad T_{\text{vol}}^{z\text{-fail}} = 2|S| + 2|F_L|$$

Quando algum vértice extrudado de um oclusor tem $w_c \le 10^{-3}$ sob
$\mathbf{P}_\infty$ (aponta para trás da câmera), o rasterizador de software
ancora esse oclusor em extrusão finita longa ($\ell = 500$).

### 4.5 Passagem de Stencil ($b$ Bits) e Quantização de Profundidade ($D$ Bits)

Seja $z$ a profundidade (NDC) de um fragmento de volume e $d$ a do Z-buffer. A
orientação da face e o resultado do teste $z < d$ determinam a atualização do
contador de stencil $s$ (Heidmann [2]; Everitt e Kilgard [1]):

$$\text{z-pass}: \; z < d \implies s \leftarrow \begin{cases} \text{INCR}(s) & \text{face frontal} \\ \text{DECR}(s) & \text{face traseira} \end{cases}$$

$$\text{z-fail}: \; z \ge d \implies s \leftarrow \begin{cases} \text{INCR}(s) & \text{face traseira} \\ \text{DECR}(s) & \text{face frontal} \end{cases}$$

Para um contador de $b \in [1, 8]$ bits com limite máximo $M = 2^b - 1$ (Seção
4.1.5 do OpenGL 1.4 [3]):

$$\text{INCR}(s) = \min(s + 1, M), \qquad \text{DECR}(s) = \max(s - 1, 0)$$

$$\text{INCR\_WRAP}(s) = (s + 1) \bmod 2^b, \qquad \text{DECR\_WRAP}(s) = (s - 1) \bmod 2^b$$

Reproduzindo as duas sub-passagens do OpenGL 1.4 (*cull* traseiro e frontal), a
fila de incrementos é processada antes da de decrementos. Em `z-fail`,
aplica-se *depth clamping* ($z \leftarrow \min(z, 1)$), e para um Z-buffer de $D$
bits a profundidade é quantizada por fragmento antes do teste:

$$z_q = \frac{2}{2^D - 1}\,\text{round}\!\left(\frac{z + 1}{2}\,(2^D - 1)\right) - 1$$

### 4.6 Verdade de Referência (Ray Casting), Métricas e Modelo Analítico de Memória

* **Ray Casting (Möller–Trumbore [10]):** Para cada pixel visível com posição de
  mundo $\mathbf{q}$ (G-buffer), lança-se o raio
  $\mathbf{o} = \mathbf{q} + \beta\hat{\mathbf{l}}$,
  $\hat{\mathbf{l}} = \widehat{\mathbf{L} - \mathbf{q}}$,
  $\beta = 2{,}5 \times 10^{-3}$, contra todos os triângulos oclusores. Há
  sombra se existe interseção com $u, v \ge 0$, $u + v \le 1$ e
  $0 < t < \|\mathbf{L} - \mathbf{q}\| - 2\beta$.
* **Métricas de Concordância:** Sobre os pixels visíveis (sombra = classe
  positiva):

$$P = \frac{TP}{TP + FP}, \qquad R = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2PR}{P + R}, \qquad E = \frac{FP + FN}{TP + FP + FN + TN}$$

* **Modelo Analítico de Memória:** Com $P_x = WH$ pixels, $c$ bits de cor por
  canal, $d = 32$ ou $D$ bits de profundidade, $s_b$ bits de stencil ($0$ sem
  sombra), $N_{\text{tex}} = 4$ texturas de lado $r$, $T_s$ triângulos de cena e
  $T_v$ triângulos de volume ($k = 3$ ou $4$ coordenadas por vértice de volume):

$$M_{\text{buf}} = P_x\left(\frac{3c}{8} + \frac{d}{8} + \frac{s_b}{8}\right) + N_{\text{tex}}\,r^2\,\frac{3c}{8}, \qquad M_{\text{geo}} = 4\,(18\,T_s + 3k\,T_v)\;\text{bytes}$$

## 5. Validação do Instrumento e Indicadores

1. **Validação de Conformidade (`conformance.py`)**: 24 testes automatizados
   verificam as 8 funções de comparação, máscaras, saturação e transbordo de
   `INCR`/`DECR`, `INVERT`, `REPLACE`, ordem stencil $\to$ profundidade e
   seleção `sfail`/`dpfail`/`dppass` (Seção 4.1.5 do OpenGL 1.4 [3]), cobertura
   *top-left* sem furos nem dupla contagem (Seção 3.5 [3]; Pineda [9]) e recorte
   no plano próximo (Seção 2.11 [3]; Sutherland–Hodgman [8]).
2. **Indicadores Coletados (`SVStats`)**:
   * `ms_geometry`, `ms_silhouette`, `ms_stencil`, `ms_composite`, `ms_total`, `fps` (mediana e IQR; custo da verificação de câmera interna e do ray casting descontados);
   * `tris_scene`, `tris_scene_drawn`, `tris_volume`, `tris_volume_drawn`, `frag_scene`, `frag_stencil`;
   * `mem_buffers_kb`, `mem_geometry_kb`, `mem_total_kb`;
   * `volumes_with_camera` ($V_c$), `stencil_saturation`, `stencil_wrap_events`, `stencil_max`;
   * `precision`, `recall`, `f1`, `error_rate` (média e desvio-padrão).

## 6. Referências

1. **[1]** C. Everitt and M. J. Kilgard, *"Practical and Robust Stenciled Shadow Volumes for Hardware-Accelerated Rendering,"* NVIDIA Corporation, 2002, arXiv:cs/0301002. (`docs/0301002v1.pdf`)
2. **[2]** T. Heidmann, *"Real Shadows, Real Time,"* *Iris Universe*, no. 18, pp. 28–31, 1991.
3. **[3]** M. Segal and K. Akeley, *The OpenGL Graphics System: A Specification (Version 1.4)*. Silicon Graphics, Inc., 2002.
4. **[4]** M. Li, M. Magnor, and H.-P. Seidel, *"Online Accelerated Rendering of Visual Hulls in Real Scenes,"* *Journal of WSCG*, vol. 11, no. 1, pp. 290–297, 2003. (`docs/D19.pdf`)
5. **[5]** T. Akenine-Möller, E. Haines, and N. Hoffman, *Real-Time Rendering*, 4th ed. Boca Raton, FL, USA: CRC Press, 2018.
6. **[6]** J. F. Blinn, *"A Trip Down the Graphics Pipeline: The Homogeneous Perspective Transform,"* *IEEE Computer Graphics and Applications*, vol. 13, no. 3, pp. 75–80, 1993.
7. **[7]** F. C. Crow, *"Shadow Algorithms for Computer Graphics,"* in *Proc. SIGGRAPH*, 1977, pp. 242–248.
8. **[8]** I. E. Sutherland and G. W. Hodgman, *"Reentrant Polygon Clipping,"* *Communications of the ACM*, vol. 17, no. 1, pp. 32–42, 1974.
9. **[9]** J. Pineda, *"A Parallel Algorithm for Polygon Rasterization,"* in *Proc. SIGGRAPH*, 1988, pp. 17–20.
10. **[10]** T. Möller and B. Trumbore, *"Fast, Minimum Storage Ray-Triangle Intersection,"* *Journal of Graphics Tools*, vol. 2, no. 1, pp. 21–28, 1997.
11. **[11]** B. R. Barbosa and J. P. Magrin, *"Aethelion: Sombras Volumétricas por Stencil,"* repositório GitHub, 2026. Disponível em: https://github.com/brfaeldev/aethelion
