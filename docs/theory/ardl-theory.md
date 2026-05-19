---
title: "ARDL: Fundamentos Teoricos"
description: "Teoria completa do modelo ARDL — bounds test de Pesaran, Shin & Smith, derivacao do ECM, multiplicadores de curto e longo prazo, distribuicao assintotica e valores criticos."
---

# ARDL: Fundamentos Teoricos

!!! abstract "Resumo"
    Esta pagina apresenta a fundamentacao matematica do modelo Autoregressive
    Distributed Lag (ARDL) e do procedimento de bounds testing para cointegracao
    proposto por Pesaran, Shin & Smith (2001). Derivamos a representacao de
    correcao de erros, os multiplicadores de curto e longo prazo, a distribuicao
    assintotica do teste F e os valores criticos em bandas. O tratamento segue
    Pesaran et al. (2001) e Pesaran & Shin (1999).

---

## 1. Modelo ARDL(p, q_1, ..., q_k)

### 1.1 Especificacao

O modelo ARDL de ordens $(p, q_1, \ldots, q_k)$ para a variavel dependente
$y_t$ e $k$ regressores $x_{1t}, \ldots, x_{kt}$:

$$
\phi(B)\, y_t = c_0 + c_1 t + \sum_{i=1}^{k} \boldsymbol{\beta}_i(B)\, x_{it} + u_t
$$

onde:

- $\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$ e o polinomio autorregressivo
- $\boldsymbol{\beta}_i(B) = \beta_{i,0} + \beta_{i,1} B + \cdots + \beta_{i,q_i} B^{q_i}$ e o polinomio de defasagens distribuidas para $x_i$
- $c_0$ e o intercepto, $c_1$ e o coeficiente de tendencia (opcional)

Explicitamente:

$$
y_t = c_0 + c_1 t + \sum_{j=1}^{p} \phi_j y_{t-j} + \sum_{i=1}^{k} \sum_{j=0}^{q_i} \beta_{i,j}\, x_{i,t-j} + u_t
$$

### 1.2 Flexibilidade de Ordens

!!! tip "Vantagem Chave"
    Diferentemente dos modelos VAR/VECM que requerem a mesma ordem de defasagem
    para todas as variaveis, o ARDL permite **ordens diferentes** para cada
    variavel ($p, q_1, \ldots, q_k$), oferecendo uma especificacao mais parcimoniosa.

---

## 2. ARDL e Cointegracao com Ordens Mistas

### 2.1 Contexto

O procedimento de bounds testing de Pesaran, Shin & Smith (2001) permite
testar a existencia de relacao de longo prazo entre variaveis que podem ser:

- $I(0)$ (estacionarias)
- $I(1)$ (integradas de ordem 1)
- Uma mistura de $I(0)$ e $I(1)$

!!! note "Vantagem sobre Johansen"
    A metodologia de Johansen (1991) requer que **todas** as variaveis sejam
    $I(1)$. O bounds test do ARDL nao exige pre-teste de raiz unitaria,
    sendo valido para regressores $I(0)$, $I(1)$ ou fracionariamente integrados,
    desde que nenhum seja $I(2)$.

### 2.2 Condicao de Existencia

Uma relacao de longo prazo existe entre $y_t$ e $\mathbf{x}_t = (x_{1t}, \ldots, x_{kt})^\top$
se a equacao de nivel:

$$
y_t = \theta_0 + \theta_1 x_{1t} + \cdots + \theta_k x_{kt} + \nu_t
$$

produz residuos $\nu_t$ estacionarios, i.e., $\nu_t \sim I(0)$, mesmo que
$y_t$ e/ou algum $x_{it}$ sejam $I(1)$.

---

## 3. Representacao de Correcao de Erros (ECM)

### 3.1 Reparametrizacao

O ARDL pode ser reparametrizado na forma de **correcao de erros condicional**
(conditional ECM):

$$
\Delta y_t = c_0 + c_1 t + \underbrace{\pi_{yy}\, y_{t-1} + \boldsymbol{\pi}_{yx}^\top \mathbf{x}_{t-1}}_{\text{parte de longo prazo}} + \sum_{j=1}^{p-1} \gamma_j \Delta y_{t-j} + \sum_{i=1}^{k} \sum_{j=0}^{q_i-1} \delta_{i,j} \Delta x_{i,t-j} + u_t
$$

onde:

$$
\pi_{yy} = -(1 - \phi_1 - \cdots - \phi_p) = -\phi(1)
$$

$$
\pi_{yx,i} = \beta_{i,0} + \beta_{i,1} + \cdots + \beta_{i,q_i} = \boldsymbol{\beta}_i(1)
$$

### 3.2 Interpretacao dos Coeficientes

- $\pi_{yy}$: velocidade de ajustamento ao equilibrio de longo prazo. Deve ser **negativo** para estabilidade.
- $\boldsymbol{\pi}_{yx}$: efeitos de longo prazo dos regressores
- $\gamma_j$, $\delta_{i,j}$: dinamicas de **curto prazo** (ajustamento)

### 3.3 Relacao de Longo Prazo Implicita

Os **coeficientes de longo prazo** sao obtidos normalizando:

$$
\theta_i = -\frac{\pi_{yx,i}}{\pi_{yy}} = \frac{\boldsymbol{\beta}_i(1)}{\phi(1)}, \quad i = 1, \ldots, k
$$

A relacao de longo prazo implicita e:

$$
y_t = -\frac{c_0}{\pi_{yy}} - \frac{\pi_{yx,1}}{\pi_{yy}} x_{1t} - \cdots - \frac{\pi_{yx,k}}{\pi_{yy}} x_{kt} + \nu_t
$$

---

## 4. Bounds Test de Pesaran, Shin & Smith (2001)

### 4.1 Hipoteses

O teste avalia a existencia de relacao de longo prazo testando a significancia
conjunta dos coeficientes de nivel na representacao ECM:

$$
H_0: \pi_{yy} = 0 \text{ e } \pi_{yx,1} = \cdots = \pi_{yx,k} = 0 \quad (\text{sem relacao de longo prazo})
$$

$$
H_1: \pi_{yy} \neq 0 \text{ ou } \pi_{yx,i} \neq 0 \text{ para algum } i \quad (\text{relacao de longo prazo existe})
$$

### 4.2 Estatistica do Teste F

A estatistica de Wald para $H_0$ e:

$$
F = \frac{(\text{RSS}_r - \text{RSS}_u) / (k+1)}{\text{RSS}_u / (T - n)}
$$

onde:

- $\text{RSS}_r$ e a soma dos quadrados dos residuos sob $H_0$ (modelo restrito, sem termos de nivel)
- $\text{RSS}_u$ e a soma dos quadrados dos residuos do modelo irrestrito (ECM completo)
- $k+1$ e o numero de restricoes ($\pi_{yy}$ e $k$ coeficientes $\pi_{yx,i}$)
- $n$ e o numero de parametros no modelo irrestrito

### 4.3 Teste t Complementar

Alem do teste F conjunto, Pesaran et al. (2001) propoem um teste t sobre
o coeficiente de ajustamento:

$$
t_{yy} = \frac{\hat{\pi}_{yy}}{\text{se}(\hat{\pi}_{yy})}
$$

sob $H_0: \pi_{yy} = 0$. Este teste e util como confirmacao do teste F.

---

## 5. Distribuicao Assintotica

### 5.1 Problema da Distribuicao Nao-Padrao

A distribuicao assintotica do teste F **depende** da ordem de integracao
dos regressores, que e desconhecida em geral. Pesaran et al. (2001) resolvem
este problema derivando **duas distribuicoes limites**:

- **Caso I(0)**: todos os regressores $x_{it}$ sao $I(0)$ — fornece o **limite inferior** (lower bound)
- **Caso I(1)**: todos os regressores $x_{it}$ sao $I(1)$ — fornece o **limite superior** (upper bound)

### 5.2 Derivacao

Sob $H_0$ e para $T \to \infty$, Pesaran et al. derivam a distribuicao
funcional da estatistica F usando a teoria de processos de Wiener:

=== "Caso I(0)"

    Se todos os regressores sao $I(0)$:

    $$
    F \xrightarrow{d} F_{\text{I(0)}} \sim \text{funcional de } W(r)
    $$

    onde $W(r)$ e um processo de Wiener padrao. Esta distribuicao fornece os
    **valores criticos inferiores** $c_L(\alpha)$.

=== "Caso I(1)"

    Se todos os regressores sao $I(1)$:

    $$
    F \xrightarrow{d} F_{\text{I(1)}} \sim \text{funcional de } W(r), W_i(r)
    $$

    onde $W_i(r)$ sao processos de Wiener independentes. Esta distribuicao
    fornece os **valores criticos superiores** $c_U(\alpha)$.

### 5.3 Nota sobre Valores Criticos Tabulados

Os valores criticos de Pesaran et al. (2001) sao obtidos por simulacao de
Monte Carlo e tabulados para diferentes combinacoes de:

- Numero de regressores $k$
- Especificacao deterministica (com/sem intercepto, com/sem tendencia)
- Nivel de significancia ($1\%$, $2.5\%$, $5\%$, $10\%$)

Narayan (2005) fornece valores criticos para amostras pequenas ($T = 30$ a $80$).

---

## 6. Valores Criticos: Bandas I(0) e I(1)

### 6.1 Regra de Decisao

A decisao do bounds test segue tres regioes:

$$
\text{Decisao} = \begin{cases}
\text{Rejeitar } H_0 & \text{se } F > c_U(\alpha) \\
\text{Inconclusivo} & \text{se } c_L(\alpha) \leq F \leq c_U(\alpha) \\
\text{Nao rejeitar } H_0 & \text{se } F < c_L(\alpha)
\end{cases}
$$

onde $c_L(\alpha)$ e $c_U(\alpha)$ sao os valores criticos inferior e superior
ao nivel de significancia $\alpha$.

!!! warning "Zona Inconclusiva"
    Quando $F$ cai na zona inconclusiva, a decisao depende da ordem de
    integracao dos regressores, e procedimentos adicionais (como pre-testes
    de raiz unitaria) sao necessarios.

### 6.2 Valores Criticos Assintoticos (Pesaran et al., 2001)

Para o **Case III** (intercepto irrestrito, sem tendencia):

| $k$ | $c_L(10\%)$ | $c_U(10\%)$ | $c_L(5\%)$ | $c_U(5\%)$ | $c_L(1\%)$ | $c_U(1\%)$ |
|---|---|---|---|---|---|---|
| 1 | 4.04 | 4.78 | 4.94 | 5.73 | 6.84 | 7.84 |
| 2 | 3.17 | 4.14 | 3.79 | 4.85 | 5.15 | 6.36 |
| 3 | 2.72 | 3.77 | 3.23 | 4.35 | 4.29 | 5.61 |
| 4 | 2.45 | 3.52 | 2.86 | 4.01 | 3.74 | 5.06 |
| 5 | 2.26 | 3.35 | 2.62 | 3.79 | 3.41 | 4.68 |

---

## 7. Multiplicadores de Curto e Longo Prazo

### 7.1 Multiplicador de Impacto

O efeito contemporaneo de uma variacao unitaria em $x_{it}$ sobre $y_t$:

$$
\frac{\partial y_t}{\partial x_{it}} = \beta_{i,0}
$$

### 7.2 Multiplicadores Dinamicos

O multiplicador dinamico no lag $s$:

$$
\frac{\partial y_{t+s}}{\partial x_{it}} = m_{i,s}
$$

obtido recursivamente a partir dos coeficientes do ARDL. A funcao de
resposta dinamica acumula os efeitos ao longo do tempo.

### 7.3 Multiplicador de Longo Prazo

O efeito total de longo prazo de uma variacao permanente unitaria em $x_i$:

$$
\theta_i = \frac{\sum_{j=0}^{q_i} \beta_{i,j}}{1 - \sum_{j=1}^{p} \phi_j} = \frac{\boldsymbol{\beta}_i(1)}{\phi(1)}
$$

### 7.4 Multiplicador Cumulativo

O **multiplicador cumulativo** ate o horizonte $h$ e:

$$
\Psi_i(h) = \sum_{s=0}^{h} m_{i,s}
$$

que converge para o multiplicador de longo prazo: $\lim_{h \to \infty} \Psi_i(h) = \theta_i$.

!!! note "Interpretacao Economica"
    Em aplicacoes macroeconomicas, os multiplicadores cumulativos permitem
    avaliar a **velocidade de transmissao** de um choque. Por exemplo, se
    $\Psi_i(4) / \theta_i = 0.8$, entao $80\%$ do efeito de longo prazo
    e realizado nos primeiros 4 periodos.

---

## 8. Vantagens do ARDL sobre Johansen

### 8.1 Comparacao

| Aspecto | ARDL / Bounds Test | Johansen (1991, 1995) |
|---|---|---|
| Ordem de integracao | $I(0)$, $I(1)$ ou mista | Requer todas $I(1)$ |
| Pre-teste de raiz unitaria | Nao necessario | Necessario |
| Amostras pequenas | Melhor performance | Vies de tamanho e potencia |
| Numero de equacoes | Equacao unica (condicional) | Sistema completo |
| Multiplas relacoes de LP | Uma relacao (normalizada) | Multiplas ($r$ relacoes) |
| Ordens de lag | Diferentes por variavel | Mesma para todas |
| Variavel endogena | Exige escolha a priori | Todas endogenas |

### 8.2 Performance em Amostras Pequenas

!!! tip "Resultado de Monte Carlo"
    Pesaran & Shin (1999) demonstram via simulacao que o estimador ARDL dos
    coeficientes de longo prazo e **super-consistente** e tem vies de amostras
    finitas menor que os estimadores alternativos (FMOLS, DOLS, Johansen)
    quando $T$ e pequeno ($T = 30$ a $80$).

### 8.3 Limitacoes do ARDL

- Assume uma **unica** relacao de cointegracao (normalizada em $y_t$)
- Requer que a variavel dependente seja $I(1)$ quando os regressores sao $I(1)$
- Nao permite regressores $I(2)$
- A endogeneidade dos regressores deve ser tratada via lags suficientes ou variaveis instrumentais

---

## 9. Selecao de Ordens

### 9.1 Criterios de Informacao

As ordens otimas $(p, q_1, \ldots, q_k)$ sao tipicamente selecionadas
minimizando AIC ou BIC sobre todas as combinacoes:

$$
(\hat{p}, \hat{q}_1, \ldots, \hat{q}_k) = \arg\min_{p, q_1, \ldots, q_k} \text{IC}(p, q_1, \ldots, q_k)
$$

### 9.2 Grade de Busca

Com $p_{\max}$ e $q_{\max}$ como ordens maximas, o numero total de modelos
a avaliar e $(p_{\max}) \times (q_{\max} + 1)^k$, que cresce exponencialmente
em $k$. Na pratica, limita-se $p_{\max} = q_{\max} = 4$ para dados trimestrais.

!!! warning "Dimensionalidade"
    Para $k = 5$ regressores e ordens maximas de $4$, sao $4 \times 5^5 = 12\,500$
    modelos. Implementacoes eficientes utilizam otimizacao por etapas ou
    restricoes adicionais para reduzir o espaco de busca.

---

## Referencias

- Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models." *Econometrica*, 59(6), 1551-1580.
- Johansen, S. (1995). *Likelihood-Based Inference in Cointegrated Vector Autoregressive Models*. Oxford University Press.
- Narayan, P. K. (2005). "The Saving and Investment Nexus for China: Evidence from Cointegration Tests." *Applied Economics*, 37(17), 1979-1990.
- Pesaran, M. H. & Shin, Y. (1999). "An Autoregressive Distributed Lag Modelling Approach to Cointegration Analysis." In *Econometrics and Economic Theory in the 20th Century: The Ragnar Frisch Centennial Symposium*, Cambridge University Press.
- Pesaran, M. H., Shin, Y. & Smith, R. J. (2001). "Bounds Testing Approaches to the Analysis of Level Relationships." *Journal of Applied Econometrics*, 16(3), 289-326.

---

## Veja Tambem

- [ARIMA](arima-theory.md) — Modelos univariados AR/MA/ARIMA
- [VAR](var-theory.md) — Vetores autorregressivos
- [VECM & Cointegracao](vecm-theory.md) — Cointegracao via Johansen
