---
title: "VAR: Fundamentos Teoricos"
description: "Teoria completa de modelos VAR — forma companion, estabilidade, representacao MA infinita, Granger causality, estimacao OLS e selecao de lags."
---

# VAR: Fundamentos Teoricos

!!! abstract "Resumo"
    Esta pagina apresenta a teoria matematica rigorosa dos modelos de Vetores
    Autorregressivos (VAR), desde a especificacao e condicoes de estabilidade ate
    a estimacao, selecao de lags e funcoes impulso-resposta. O tratamento segue
    Lutkepohl (2005) e Hamilton (1994).

---

## 1. Especificacao do Modelo VAR(p)

### 1.1 Definicao

Um modelo VAR de ordem $p$ para um vetor $K$-dimensional $\mathbf{y}_t = (y_{1t}, \ldots, y_{Kt})^\top$:

$$
\mathbf{y}_t = \boldsymbol{\nu} + \mathbf{A}_1 \mathbf{y}_{t-1} + \mathbf{A}_2 \mathbf{y}_{t-2} + \cdots + \mathbf{A}_p \mathbf{y}_{t-p} + \mathbf{u}_t
$$

onde:

- $\boldsymbol{\nu}$ e um vetor $K \times 1$ de interceptos
- $\mathbf{A}_i$ sao matrizes $K \times K$ de coeficientes ($i = 1, \ldots, p$)
- $\mathbf{u}_t$ e um processo ruido branco $K$-dimensional: $\E[\mathbf{u}_t] = \mathbf{0}$, $\E[\mathbf{u}_t \mathbf{u}_t^\top] = \boldsymbol{\Sigma}_u$ (definida positiva), $\E[\mathbf{u}_t \mathbf{u}_s^\top] = \mathbf{0}$ para $t \neq s$

### 1.2 Forma Compacta

Usando o operador backshift:

$$
\mathbf{A}(B)\, \mathbf{y}_t = \boldsymbol{\nu} + \mathbf{u}_t
$$

onde $\mathbf{A}(B) = \mathbf{I}_K - \mathbf{A}_1 B - \mathbf{A}_2 B^2 - \cdots - \mathbf{A}_p B^p$.

---

## 2. Forma Companion

### 2.1 Representacao VAR(1)

Todo VAR(p) pode ser reescrito como um VAR(1) em dimensao aumentada. Definindo:

$$
\mathbf{Y}_t = \begin{pmatrix} \mathbf{y}_t \\ \mathbf{y}_{t-1} \\ \vdots \\ \mathbf{y}_{t-p+1} \end{pmatrix}_{Kp \times 1}, \quad
\boldsymbol{\mathcal{A}} = \begin{pmatrix}
\mathbf{A}_1 & \mathbf{A}_2 & \cdots & \mathbf{A}_{p-1} & \mathbf{A}_p \\
\mathbf{I}_K & \mathbf{0} & \cdots & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{I}_K & \cdots & \mathbf{0} & \mathbf{0} \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
\mathbf{0} & \mathbf{0} & \cdots & \mathbf{I}_K & \mathbf{0}
\end{pmatrix}_{Kp \times Kp}
$$

o VAR(p) torna-se:

$$
\mathbf{Y}_t = \boldsymbol{\mathcal{V}} + \boldsymbol{\mathcal{A}}\, \mathbf{Y}_{t-1} + \boldsymbol{\mathcal{U}}_t
$$

onde $\boldsymbol{\mathcal{V}} = (\boldsymbol{\nu}^\top, \mathbf{0}^\top, \ldots, \mathbf{0}^\top)^\top$ e $\boldsymbol{\mathcal{U}}_t = (\mathbf{u}_t^\top, \mathbf{0}^\top, \ldots, \mathbf{0}^\top)^\top$.

### 2.2 Importancia da Forma Companion

A forma companion permite:

- Verificar estabilidade via eigenvalues de $\boldsymbol{\mathcal{A}}$
- Computar potencias $\boldsymbol{\mathcal{A}}^n$ para IRFs e previsoes
- Derivar propriedades assintoticas de forma compacta

---

## 3. Estabilidade

### 3.1 Condicao Necessaria e Suficiente

O processo VAR(p) e **estavel** (estacionario) se e somente se:

$$
\det(\mathbf{I}_K - \mathbf{A}_1 z - \mathbf{A}_2 z^2 - \cdots - \mathbf{A}_p z^p) \neq 0 \quad \text{para } |z| \leq 1
$$

Equivalentemente, todos os eigenvalues da **companion matrix** $\boldsymbol{\mathcal{A}}$ devem
estar estritamente dentro do circulo unitario:

$$
|\lambda_i(\boldsymbol{\mathcal{A}})| < 1, \quad i = 1, \ldots, Kp
$$

### 3.2 Momentos do Processo Estavel

Se o VAR(p) e estavel, a media incondicional existe e e dada por:

$$
\boldsymbol{\mu} = \E[\mathbf{y}_t] = (\mathbf{I}_K - \mathbf{A}_1 - \cdots - \mathbf{A}_p)^{-1}\, \boldsymbol{\nu}
$$

A funcao de autocovariancia $\boldsymbol{\Gamma}(h) = \E[(\mathbf{y}_t - \boldsymbol{\mu})(\mathbf{y}_{t-h} - \boldsymbol{\mu})^\top]$
satisfaz as **equacoes de Yule-Walker**:

$$
\boldsymbol{\Gamma}(h) = \mathbf{A}_1 \boldsymbol{\Gamma}(h-1) + \mathbf{A}_2 \boldsymbol{\Gamma}(h-2) + \cdots + \mathbf{A}_p \boldsymbol{\Gamma}(h-p), \quad h > 0
$$

com condicao inicial obtida via equacao de Lyapunov discreta:

$$
\text{vec}(\boldsymbol{\Gamma}(0)) = (\mathbf{I}_{K^2} - \mathbf{A}_1 \otimes \mathbf{A}_1 - \cdots - \mathbf{A}_p \otimes \mathbf{A}_p)^{-1}\, \text{vec}(\boldsymbol{\Sigma}_u) + \cdots
$$

Para a forma companion, de modo mais compacto:

$$
\text{vec}(\boldsymbol{\Gamma}_\mathcal{A}) = (\mathbf{I}_{(Kp)^2} - \boldsymbol{\mathcal{A}} \otimes \boldsymbol{\mathcal{A}})^{-1}\, \text{vec}(\boldsymbol{\Sigma}_\mathcal{U})
$$

---

## 4. Representacao MA($\infty$)

### 4.1 Derivacao

Se o VAR(p) e estavel, admite a representacao de medias moveis infinitas:

$$
\mathbf{y}_t = \boldsymbol{\mu} + \sum_{i=0}^{\infty} \boldsymbol{\Phi}_i\, \mathbf{u}_{t-i}
$$

onde $\boldsymbol{\Phi}_0 = \mathbf{I}_K$ e os coeficientes satisfazem a recursao:

$$
\boldsymbol{\Phi}_i = \sum_{j=1}^{\min(i,p)} \boldsymbol{\Phi}_{i-j}\, \mathbf{A}_j, \quad i = 1, 2, \ldots
$$

com $\boldsymbol{\Phi}_i = \mathbf{0}$ para $i < 0$.

### 4.2 Interpretacao

O elemento $(k, \ell)$ de $\boldsymbol{\Phi}_i$ mede o efeito de um choque unitario na
variavel $\ell$ sobre a variavel $k$, apos $i$ periodos. Isso constitui a
**funcao impulso-resposta (IRF)** na forma reduzida.

### 4.3 Relacao com a Companion Matrix

$$
\boldsymbol{\Phi}_i = \mathbf{J}\, \boldsymbol{\mathcal{A}}^i\, \mathbf{J}^\top
$$

onde $\mathbf{J} = [\mathbf{I}_K, \mathbf{0}, \ldots, \mathbf{0}]$ e uma matriz $K \times Kp$
de selecao.

---

## 5. Causalidade de Granger

### 5.1 Definicao

A variavel $y_{jt}$ **Granger-causa** $y_{it}$ se os valores passados de $y_{jt}$ contem
informacao util para prever $y_{it}$ alem daquela ja contida nos valores passados de $y_{it}$
e de todas as outras variaveis do sistema.

Formalmente, $y_j$ **nao** Granger-causa $y_i$ se:

$$
\mathbf{A}_{1}^{(ij)} = \mathbf{A}_{2}^{(ij)} = \cdots = \mathbf{A}_{p}^{(ij)} = 0
$$

onde $\mathbf{A}_{\ell}^{(ij)}$ denota o elemento $(i,j)$ da matriz $\mathbf{A}_\ell$.

### 5.2 Teste

A hipotese nula $H_0$: "$y_j$ nao Granger-causa $y_i$" impoe $p$ restricoes lineares
e e testada via estatistica F (Wald):

$$
F = \frac{(\text{RSS}_r - \text{RSS}_u)/p}{\text{RSS}_u / (T - Kp - 1)} \sim F(p, T - Kp - 1)
$$

!!! warning "Limitacoes"
    Causalidade de Granger e um conceito de **precedencia temporal**, nao de
    causalidade estrutural. Um resultado significativo indica poder preditivo,
    nao necessariamente um mecanismo causal.

---

## 6. Estimacao

### 6.1 OLS Equacao por Equacao

Cada equacao do VAR(p) pode ser estimada separadamente por OLS. Escrevendo
o sistema em forma empilhada:

$$
\mathbf{Y} = \mathbf{B}\, \mathbf{X} + \mathbf{U}
$$

onde:

- $\mathbf{Y} = (\mathbf{y}_1, \ldots, \mathbf{y}_T)$ e $K \times T$
- $\mathbf{B} = (\boldsymbol{\nu}, \mathbf{A}_1, \ldots, \mathbf{A}_p)$ e $K \times (Kp+1)$
- $\mathbf{X} = (\mathbf{x}_0, \ldots, \mathbf{x}_{T-1})$ com $\mathbf{x}_t = (1, \mathbf{y}_t^\top, \ldots, \mathbf{y}_{t-p+1}^\top)^\top$ e $(Kp+1) \times T$

O estimador OLS:

$$
\hat{\mathbf{B}} = \mathbf{Y}\, \mathbf{X}^\top (\mathbf{X}\, \mathbf{X}^\top)^{-1}
$$

A estimativa da covariancia dos residuos:

$$
\hat{\boldsymbol{\Sigma}}_u = \frac{1}{T - Kp - 1}\, \hat{\mathbf{U}}\, \hat{\mathbf{U}}^\top
$$

### 6.2 Consistencia e Normalidade Assintotica

**Teorema (Lutkepohl, 2005, Cap. 3).** Se $\mathbf{y}_t$ e um processo VAR(p) estavel
com $\E[\|\mathbf{u}_t\|^4] < \infty$, entao:

$$
\text{vec}(\hat{\mathbf{B}}) \xrightarrow{p} \text{vec}(\mathbf{B})
$$

e

$$
\sqrt{T}\, \text{vec}(\hat{\mathbf{B}} - \mathbf{B}) \xrightarrow{d} N\left(\mathbf{0},\; \boldsymbol{\Gamma}_X^{-1} \otimes \boldsymbol{\Sigma}_u\right)
$$

onde $\boldsymbol{\Gamma}_X = \plim T^{-1} \mathbf{X}\mathbf{X}^\top$.

### 6.3 Eficiencia

!!! note "OLS = GLS"
    Como todas as equacoes do VAR possuem os mesmos regressores, o estimador OLS
    equacao por equacao coincide com o estimador GLS (Generalized Least Squares)
    e com o estimador de maxima verossimilhanca sob normalidade. Portanto, OLS
    e **BLUE** e assintoticamente eficiente neste contexto.

---

## 7. Selecao de Lags

### 7.1 Criterios de Informacao

A selecao da ordem $p$ e realizada minimizando criterios de informacao:

$$
\text{AIC}(p) = \ln|\hat{\boldsymbol{\Sigma}}_u(p)| + \frac{2pK^2}{T}
$$

$$
\text{BIC}(p) = \ln|\hat{\boldsymbol{\Sigma}}_u(p)| + \frac{pK^2 \ln T}{T}
$$

$$
\text{HQ}(p) = \ln|\hat{\boldsymbol{\Sigma}}_u(p)| + \frac{2pK^2 \ln(\ln T)}{T}
$$

onde $\hat{\boldsymbol{\Sigma}}_u(p) = T^{-1} \sum_{t=1}^{T} \hat{\mathbf{u}}_t \hat{\mathbf{u}}_t^\top$
e a estimativa ML da covariancia residual para o VAR($p$).

### 7.2 Propriedades

| Criterio | Consistencia | Eficiencia |
|---|---|---|
| AIC | Nao (tende a sobre-parametrizar) | Sim (minimiza perda de previsao) |
| BIC | Sim (seleciona $p_0$ verdadeiro) | Nao |
| HQ | Sim (sob certas condicoes) | Nao |

!!! tip "Recomendacao Pratica"
    Lutkepohl (2005) recomenda usar BIC ou HQ para identificacao da ordem verdadeira.
    Para previsao, AIC pode ser preferivel. Na pratica, e util comparar os resultados
    dos diferentes criterios e verificar a adequacao dos residuos.

### 7.3 Testes Sequenciais

Uma abordagem alternativa e o **teste de razao de verossimilhanca sequencial**
(LR), testando VAR($p_{\max}$) contra VAR($p_{\max}-1$), etc.:

$$
LR = T\left[\ln|\hat{\boldsymbol{\Sigma}}_u(p-1)| - \ln|\hat{\boldsymbol{\Sigma}}_u(p)|\right] \xrightarrow{d} \chi^2(K^2)
$$

---

## 8. Funcoes Impulso-Resposta

### 8.1 IRF na Forma Reduzida

A resposta da variavel $k$ a um choque unitario na variavel $\ell$, apos $i$ periodos:

$$
\text{IRF}_{k\ell}(i) = [\boldsymbol{\Phi}_i]_{k\ell}
$$

### 8.2 IRF Ortogonalizada

Para isolar o efeito de choques nao correlacionados, utiliza-se a decomposicao
de Cholesky $\boldsymbol{\Sigma}_u = \mathbf{P}\mathbf{P}^\top$ (triangular inferior):

$$
\boldsymbol{\Theta}_i = \boldsymbol{\Phi}_i\, \mathbf{P}
$$

O elemento $(k, \ell)$ de $\boldsymbol{\Theta}_i$ mede o efeito de um choque de um
desvio-padrao na inovacao ortogonalizada $\ell$ sobre a variavel $k$.

!!! warning "Dependencia da Ordenacao"
    A IRF ortogonalizada via Cholesky depende da **ordenacao das variaveis**, pois
    a decomposicao triangular inferior impoe uma estrutura causal recursiva.
    Para identificacao estrutural sem essa restricao, veja [SVAR](svar-theory.md).

### 8.3 Intervalos de Confianca

Intervalos para IRFs sao tipicamente obtidos via:

=== "Assintotico"

    Utilizando o delta method (Lutkepohl, 2005, Cap. 3):

    $$
    \sqrt{T}\, \text{vec}(\hat{\boldsymbol{\Phi}}_i - \boldsymbol{\Phi}_i) \xrightarrow{d} N(\mathbf{0}, \boldsymbol{\Sigma}_{\Phi_i})
    $$

    onde $\boldsymbol{\Sigma}_{\Phi_i}$ e obtida via derivadas dos $\boldsymbol{\Phi}_i$ em relacao aos parametros do VAR.

=== "Bootstrap"

    O bootstrap residual (Efron & Tibshirani, 1993):

    1. Estimar VAR e obter residuos $\hat{\mathbf{u}}_t$
    2. Gerar pseudo-amostras por reamostragem dos residuos
    3. Re-estimar o VAR e computar IRFs em cada replicacao
    4. Construir intervalos via percentis empiricos

---

## 9. Decomposicao da Variancia do Erro de Previsao (FEVD)

A variancia do erro de previsao $h$ passos a frente para a variavel $k$:

$$
\text{MSE}_k(h) = \sum_{i=0}^{h-1} \sum_{\ell=1}^{K} [\boldsymbol{\Theta}_i]_{k\ell}^2
$$

A **proporcao da variancia** de $y_k$ atribuida a choques em $y_\ell$:

$$
\omega_{k\ell}(h) = \frac{\sum_{i=0}^{h-1} [\boldsymbol{\Theta}_i]_{k\ell}^2}{\sum_{i=0}^{h-1} \sum_{j=1}^{K} [\boldsymbol{\Theta}_i]_{kj}^2}
$$

satisfazendo $\sum_{\ell=1}^{K} \omega_{k\ell}(h) = 1$ para todo $k$ e $h$.

---

## 10. Granger Representation Theorem

!!! tip "Conexao com Cointegracao"
    O Granger Representation Theorem fornece a ponte entre VAR, VECM e cointegracao.

**Teorema (Engle & Granger, 1987).** Se $\mathbf{y}_t \sim I(1)$ com $\mathbf{y}_t = (y_{1t}, \ldots, y_{Kt})^\top$,
e existem vetores $\boldsymbol{\beta}$ tais que $\boldsymbol{\beta}^\top \mathbf{y}_t \sim I(0)$
(i.e., as variaveis sao cointegradas), entao existe uma representacao de correcao de erros:

$$
\diff \mathbf{y}_t = \boldsymbol{\alpha}\boldsymbol{\beta}^\top \mathbf{y}_{t-1} + \sum_{i=1}^{p-1} \boldsymbol{\Gamma}_i \diff \mathbf{y}_{t-i} + \mathbf{u}_t
$$

e uma representacao MA:

$$
\mathbf{y}_t = \mathbf{C} \sum_{i=1}^{t} \mathbf{u}_i + C^*(B)\, \mathbf{u}_t + \mathbf{y}_0^*
$$

onde $\mathbf{C} = \boldsymbol{\beta}_\perp (\boldsymbol{\alpha}_\perp^\top \boldsymbol{\Gamma} \boldsymbol{\beta}_\perp)^{-1} \boldsymbol{\alpha}_\perp^\top$,
$\boldsymbol{\Gamma} = \mathbf{I}_K - \sum_{i=1}^{p-1} \boldsymbol{\Gamma}_i$, e $C^*(B)$ e um polinomio matricial
com coeficientes absolutamente somaveis.

A existencia de cointegracao (e portanto de uma representacao VECM) e o tema da
pagina [VECM & Cointegracao](vecm-theory.md).

---

## Referencias

- Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Engle, R. F. & Granger, C. W. J. (1987). "Co-integration and Error Correction: Representation, Estimation, and Testing." *Econometrica*, 55(2), 251-276.
- Granger, C. W. J. (1969). "Investigating Causal Relations by Econometric Models and Cross-Spectral Methods." *Econometrica*, 37(3), 424-438.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Sims, C. A. (1980). "Macroeconomics and Reality." *Econometrica*, 48(1), 1-48.

---

## Veja Tambem

- [ARIMA](arima-theory.md) — Modelos univariados AR/MA/ARIMA
- [VECM & Cointegracao](vecm-theory.md) — Variaveis integradas e cointegradas
- [SVAR](svar-theory.md) — Identificacao estrutural
