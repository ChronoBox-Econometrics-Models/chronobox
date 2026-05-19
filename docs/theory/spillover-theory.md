---
title: "Spillover: Fundamentos Teoricos"
description: "Teoria completa do framework de spillover de Diebold-Yilmaz — FEVD generalizada, spillover table, indices direcionais e net pairwise, com extensoes time-varying e frequency domain."
---

# Spillover: Fundamentos Teoricos

!!! abstract "Resumo"
    Esta pagina apresenta a fundamentacao matematica do framework de spillover
    proposto por Diebold & Yilmaz (2009, 2012, 2014). Partindo da decomposicao
    da variancia do erro de previsao generalizada (GFEVD) de Pesaran & Shin
    (1998), derivamos a construcao da spillover table, os indices direcionais
    e net pairwise, e discutimos as extensoes para janelas moveis (time-varying)
    e decomposicao no dominio da frequencia.

---

## 1. FEVD Generalizada de Pesaran-Shin

### 1.1 Motivacao

A decomposicao da variancia do erro de previsao (FEVD) ortogonalizada via
Cholesky depende da **ordenacao das variaveis** (ver [VAR](var-theory.md),
Secao 8.2). A FEVD **generalizada** de Pesaran & Shin (1998) remove essa
dependencia ao permitir correlacao contemporanea entre os choques.

### 1.2 VAR e Representacao MA

Considere um VAR(p) estavel com $K$ variaveis:

$$
\mathbf{y}_t = \boldsymbol{\nu} + \sum_{i=1}^{p} \mathbf{A}_i\, \mathbf{y}_{t-i} + \mathbf{u}_t, \quad \mathbf{u}_t \sim N(\mathbf{0}, \boldsymbol{\Sigma})
$$

com representacao MA($\infty$):

$$
\mathbf{y}_t = \boldsymbol{\mu} + \sum_{h=0}^{\infty} \boldsymbol{\Phi}_h\, \mathbf{u}_{t-h}
$$

onde $\boldsymbol{\Phi}_0 = \mathbf{I}_K$ e os coeficientes sao obtidos recursivamente
(ver [VAR](var-theory.md), Secao 4).

### 1.3 GFEVD: Definicao

A proporcao da variancia do erro de previsao $H$-passos a frente da variavel
$i$ atribuida a choques na variavel $j$ e:

$$
\tilde{d}_{ij}^H = \frac{\sigma_{jj}^{-1} \sum_{h=0}^{H-1} (\mathbf{e}_i^\top \boldsymbol{\Phi}_h\, \boldsymbol{\Sigma}\, \mathbf{e}_j)^2}{\sum_{h=0}^{H-1} \mathbf{e}_i^\top \boldsymbol{\Phi}_h\, \boldsymbol{\Sigma}\, \boldsymbol{\Phi}_h^\top\, \mathbf{e}_i}
$$

onde:

- $\mathbf{e}_i$ e o $i$-esimo vetor da base canonica de $\mathbb{R}^K$
- $\sigma_{jj} = [\boldsymbol{\Sigma}]_{jj}$ e a variancia do choque $j$
- $H$ e o horizonte de previsao

### 1.4 Normalizacao

Como os choques generalizados nao sao ortogonais, as linhas da matriz de GFEVD
**nao somam 1** em geral. A normalizacao e:

$$
\tilde{d}_{ij}^{H,*} = \frac{\tilde{d}_{ij}^H}{\sum_{j=1}^{K} \tilde{d}_{ij}^H}
$$

Apos a normalizacao, $\sum_{j=1}^{K} \tilde{d}_{ij}^{H,*} = 1$ e
$\sum_{i=1}^{K} \sum_{j=1}^{K} \tilde{d}_{ij}^{H,*} = K$.

!!! note "Invariancia a Ordenacao"
    A GFEVD normalizada e **invariante a ordenacao** das variaveis,
    eliminando a arbitrariedade da decomposicao de Cholesky. Isso a torna
    particularmente adequada para medir conectividade em sistemas onde
    nao ha justificativa teorica para uma ordenacao especifica.

---

## 2. Construcao da Spillover Table

### 2.1 Matriz de Decomposicao

A **spillover table** organiza os $\tilde{d}_{ij}^{H,*}$ em uma matriz $K \times K$:

$$
\mathbf{D}^H = \begin{pmatrix}
\tilde{d}_{11}^{H,*} & \tilde{d}_{12}^{H,*} & \cdots & \tilde{d}_{1K}^{H,*} \\
\tilde{d}_{21}^{H,*} & \tilde{d}_{22}^{H,*} & \cdots & \tilde{d}_{2K}^{H,*} \\
\vdots & \vdots & \ddots & \vdots \\
\tilde{d}_{K1}^{H,*} & \tilde{d}_{K2}^{H,*} & \cdots & \tilde{d}_{KK}^{H,*}
\end{pmatrix}
$$

### 2.2 Interpretacao

- **Diagonal** ($\tilde{d}_{ii}^{H,*}$): proporcao da variancia de $y_i$ devida a **choques proprios**
- **Fora da diagonal** ($\tilde{d}_{ij}^{H,*}$, $i \neq j$): proporcao da variancia de $y_i$ devida a choques em $y_j$ — **spillover recebido**
- **Soma da linha** (excl. diagonal): total de spillover **recebido** pela variavel $i$
- **Soma da coluna** (excl. diagonal): total de spillover **transmitido** pela variavel $j$

### 2.3 Formato da Tabela

|  | $y_1$ | $y_2$ | $\cdots$ | $y_K$ | **From others** |
|---|---|---|---|---|---|
| $y_1$ | $\tilde{d}_{11}$ | $\tilde{d}_{12}$ | $\cdots$ | $\tilde{d}_{1K}$ | $\sum_{j \neq 1} \tilde{d}_{1j}$ |
| $y_2$ | $\tilde{d}_{21}$ | $\tilde{d}_{22}$ | $\cdots$ | $\tilde{d}_{2K}$ | $\sum_{j \neq 2} \tilde{d}_{2j}$ |
| $\vdots$ | $\vdots$ | $\vdots$ | $\ddots$ | $\vdots$ | $\vdots$ |
| $y_K$ | $\tilde{d}_{K1}$ | $\tilde{d}_{K2}$ | $\cdots$ | $\tilde{d}_{KK}$ | $\sum_{j \neq K} \tilde{d}_{Kj}$ |
| **To others** | $\sum_{i \neq 1} \tilde{d}_{i1}$ | $\sum_{i \neq 2} \tilde{d}_{i2}$ | $\cdots$ | $\sum_{i \neq K} \tilde{d}_{iK}$ | $S$ |

---

## 3. Total Spillover Index

### 3.1 Definicao

O **Total Spillover Index** mede a contribuicao agregada dos spillovers
cruzados para a variancia total do sistema:

$$
S = \frac{\sum_{i=1}^{K} \sum_{\substack{j=1 \\ j \neq i}}^{K} \tilde{d}_{ij}^{H,*}}{\sum_{i=1}^{K} \sum_{j=1}^{K} \tilde{d}_{ij}^{H,*}} \times 100 = \frac{\sum_{i=1}^{K} \sum_{\substack{j=1 \\ j \neq i}}^{K} \tilde{d}_{ij}^{H,*}}{K} \times 100
$$

### 3.2 Interpretacao

- $S = 0\%$: nenhum spillover — cada variavel e determinada apenas por seus proprios choques
- $S = 100\%$: spillover maximo — toda a variancia e devida a choques cruzados
- Na pratica, $S$ tipicamente varia entre $20\%$ e $80\%$ em aplicacoes financeiras

### 3.3 Propriedades

| Propriedade | Descricao |
|---|---|
| Limitado | $S \in [0, 100]$ |
| Invariante a ordenacao | Baseado na GFEVD de Pesaran-Shin |
| Dependente do horizonte | Varia com $H$ (tipicamente estabiliza para $H$ grande) |
| Agregacao | Resume a conectividade do sistema em um unico numero |

---

## 4. Directional Spillover

### 4.1 Spillover Recebido (From)

O spillover **recebido** pela variavel $i$ de todas as demais variaveis:

$$
S_{i \leftarrow \bullet}^H = \frac{\sum_{\substack{j=1 \\ j \neq i}}^{K} \tilde{d}_{ij}^{H,*}}{K} \times 100
$$

Mede a **vulnerabilidade** ou **exposicao** da variavel $i$ a choques externos.

### 4.2 Spillover Transmitido (To)

O spillover **transmitido** pela variavel $j$ para todas as demais:

$$
S_{\bullet \leftarrow j}^H = \frac{\sum_{\substack{i=1 \\ i \neq j}}^{K} \tilde{d}_{ij}^{H,*}}{K} \times 100
$$

Mede a **influencia** ou **impacto sistemico** da variavel $j$ sobre o sistema.

### 4.3 Relacao com o Total Spillover

$$
S = \frac{1}{K} \sum_{i=1}^{K} S_{i \leftarrow \bullet}^H = \frac{1}{K} \sum_{j=1}^{K} S_{\bullet \leftarrow j}^H
$$

O total spillover e a media dos spillovers direcionais.

---

## 5. Net Pairwise Spillover

### 5.1 Net Spillover

O spillover **liquido** da variavel $i$ e:

$$
S_i^H = S_{\bullet \leftarrow i}^H - S_{i \leftarrow \bullet}^H
$$

- $S_i^H > 0$: variavel $i$ e um **transmissor liquido** de choques
- $S_i^H < 0$: variavel $i$ e um **receptor liquido** de choques

### 5.2 Net Pairwise

O spillover liquido **bilateral** entre as variaveis $i$ e $j$:

$$
S_{ij}^H = \frac{\tilde{d}_{ji}^{H,*} - \tilde{d}_{ij}^{H,*}}{K} \times 100
$$

- $S_{ij}^H > 0$: $i$ transmite mais para $j$ do que recebe de $j$
- $S_{ij}^H < 0$: $i$ recebe mais de $j$ do que transmite para $j$
- Antissimetria: $S_{ij}^H = -S_{ji}^H$

!!! note "Redes de Spillover"
    Os net pairwise spillovers definem uma **rede direcionada ponderada**
    entre as variaveis, onde os nos sao as variaveis e os pesos das
    arestas sao $S_{ij}^H$. Essa representacao e util para visualizar
    a estrutura de transmissao de choques em sistemas financeiros.

---

## 6. Propriedades do Indice

### 6.1 Propriedades Formais

O framework de Diebold-Yilmaz satisfaz as seguintes propriedades:

1. **Nao-negatividade**: $\tilde{d}_{ij}^{H,*} \geq 0$ para todo $i, j$
2. **Normalizacao por linha**: $\sum_{j=1}^{K} \tilde{d}_{ij}^{H,*} = 1$
3. **Invariancia a ordenacao**: nao depende da sequencia das variaveis
4. **Decomponibilidade**: $S$ pode ser decomposto em contribuicoes direcionais e pairwise
5. **Monotonicidade em $H$**: para $H \to \infty$, $\tilde{d}_{ij}^{H,*}$ converge

### 6.2 Escolha do Horizonte $H$

| Horizonte $H$ | Interpretacao |
|---|---|
| $H = 1$ | Spillover contemporaneo e de curtissimo prazo |
| $H = 4$ (trimestral) | Spillover de curto prazo (1 ano) |
| $H = 10$-$12$ | Spillover de medio prazo (padrao na literatura) |
| $H \to \infty$ | Spillover incondicional (independente do horizonte) |

!!! tip "Recomendacao"
    Diebold & Yilmaz (2012) recomendam $H = 10$ para dados diarios de
    mercados financeiros. Para dados macroeconomicos trimestrais, $H = 8$ a
    $12$ e usual. Os resultados devem ser verificados para robustez a $H$.

---

## 7. Extensoes

### 7.1 Time-Varying Spillover (Rolling Window)

Para capturar a **variacao temporal** da conectividade, o framework e
aplicado a janelas moveis (rolling windows) de tamanho $w$:

$$
S_t = S(\mathbf{Y}_{t-w+1:t}), \quad t = w, w+1, \ldots, T
$$

onde $\mathbf{Y}_{t-w+1:t}$ denota a subamostra de $w$ observacoes terminando em $t$.

A serie $\{S_t\}$ revela como a conectividade do sistema evolui ao longo
do tempo, tipicamente aumentando durante crises financeiras.

!!! warning "Escolha do Tamanho da Janela"
    A escolha de $w$ envolve um trade-off:

    - $w$ grande: estimativas mais precisas, menor variabilidade, mas
      menor capacidade de detectar mudancas abruptas
    - $w$ pequeno: maior sensibilidade a mudancas, mas estimativas
      mais ruidosas e possivelmente instaveis

    Valores tipicos: $w = 100$-$200$ para dados diarios, $w = 40$-$60$
    para dados trimestrais.

### 7.2 TVP-VAR Spillover

Uma alternativa ao rolling window e utilizar um VAR com **parametros
variantes no tempo** (TVP-VAR) estimado via filtro de Kalman (Antonakakis
et al., 2020), que evita a escolha arbitraria de $w$:

$$
\mathbf{y}_t = \mathbf{B}_t\, \mathbf{x}_t + \mathbf{u}_t, \quad \mathbf{u}_t \sim N(\mathbf{0}, \boldsymbol{\Sigma}_t)
$$

$$
\text{vec}(\mathbf{B}_t) = \text{vec}(\mathbf{B}_{t-1}) + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim N(\mathbf{0}, \mathbf{Q})
$$

### 7.3 Frequency Domain Decomposition

Barunik & Krehlik (2018) decompoem o spillover no **dominio da frequencia**,
permitindo separar a conectividade por faixas de frequencia:

$$
\tilde{d}_{ij}(\omega) = \frac{\sigma_{jj}^{-1} |\mathbf{e}_i^\top \boldsymbol{\Phi}(\omega)\, \boldsymbol{\Sigma}\, \mathbf{e}_j|^2}{\mathbf{e}_i^\top \left[\sum_h \boldsymbol{\Phi}_h\, \boldsymbol{\Sigma}\, \boldsymbol{\Phi}_h^\top\right]\, \mathbf{e}_i}
$$

onde $\boldsymbol{\Phi}(\omega) = \sum_{h=0}^{\infty} \boldsymbol{\Phi}_h\, e^{-i\omega h}$ e a
funcao de transferencia do VAR.

Os spillovers podem entao ser decompostos em contribuicoes de:

- **Curto prazo** ($\omega$ alto): ciclos de menos de 1 mes
- **Medio prazo** ($\omega$ intermediario): ciclos de 1-12 meses
- **Longo prazo** ($\omega$ baixo): ciclos de mais de 12 meses

---

## 8. Implementacao Computacional

### 8.1 Algoritmo

O procedimento completo para computar o spillover table e:

1. Estimar o VAR(p) (via OLS ou BVAR)
2. Computar os coeficientes MA: $\boldsymbol{\Phi}_0, \boldsymbol{\Phi}_1, \ldots, \boldsymbol{\Phi}_{H-1}$
3. Computar a GFEVD: $\tilde{d}_{ij}^H$ para todo $i, j$
4. Normalizar: $\tilde{d}_{ij}^{H,*} = \tilde{d}_{ij}^H / \sum_j \tilde{d}_{ij}^H$
5. Calcular indices: $S$, $S_{i \leftarrow \bullet}$, $S_{\bullet \leftarrow j}$, $S_i$, $S_{ij}$

### 8.2 Parametros de Configuracao

| Parametro | Descricao | Valor padrao |
|---|---|---|
| $p$ | Ordem do VAR | Selecionado via AIC/BIC |
| $H$ | Horizonte de previsao | 10 |
| $w$ | Tamanho da janela (rolling) | 200 (diario), 60 (trimestral) |

---

## Referencias

- Antonakakis, N., Chatziantoniou, I. & Gabauer, D. (2020). "Refined Measures of Dynamic Connectedness Based on Time-Varying Parameter Vector Autoregressions." *Journal of Risk and Financial Management*, 13(4), 84.
- Barunik, J. & Krehlik, T. (2018). "Measuring the Frequency Dynamics of Financial Connectedness and Systemic Risk." *Journal of Financial Econometrics*, 16(2), 271-296.
- Diebold, F. X. & Yilmaz, K. (2009). "Measuring Financial Asset Return and Volatility Spillovers, with Application to Global Equity Markets." *Economic Journal*, 119(534), 158-171.
- Diebold, F. X. & Yilmaz, K. (2012). "Better to Give than to Receive: Predictive Directional Measurement of Volatility Spillovers." *International Journal of Forecasting*, 28(1), 57-66.
- Diebold, F. X. & Yilmaz, K. (2014). "On the Network Topology of Variance Decompositions: Measuring the Connectedness of Financial Firms." *Journal of Econometrics*, 182(1), 119-134.
- Koop, G., Pesaran, M. H. & Potter, S. M. (1996). "Impulse Response Analysis in Nonlinear Multivariate Models." *Journal of Econometrics*, 74(1), 119-147.
- Pesaran, M. H. & Shin, Y. (1998). "Generalized Impulse Response Analysis in Linear Multivariate Models." *Economics Letters*, 58(1), 17-29.

---

## Veja Tambem

- [VAR](var-theory.md) — Fundamentos do VAR e FEVD ortogonalizada
- [SVAR](svar-theory.md) — Identificacao estrutural
- [BVAR](bvar-theory.md) — VAR Bayesiano (alternativa na estimacao)
