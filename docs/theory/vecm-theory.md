---
title: "VECM & Cointegracao: Fundamentos Teoricos"
description: "Teoria completa de VECM — cointegracao, Engle-Granger, procedimento de Johansen, decomposicao alpha-beta, termos deterministicos e estimacao FIML."
---

# VECM & Cointegracao: Fundamentos Teoricos

!!! abstract "Resumo"
    Esta pagina apresenta a teoria matematica dos modelos de Correcao de Erros
    Vetoriais (VECM), abrangendo a definicao formal de cointegracao, o teorema
    de representacao de Engle-Granger, o procedimento completo de Johansen para
    determinacao do rank de cointegracao, a decomposicao $\boldsymbol{\Pi} = \boldsymbol{\alpha}\boldsymbol{\beta}^\top$,
    e os cinco casos deterministicos.

---

## 1. Cointegracao: Definicao e Intuicao

### 1.1 Definicao Formal

**Definicao (Engle & Granger, 1987).** Os componentes do vetor $\mathbf{y}_t = (y_{1t}, \ldots, y_{Kt})^\top$
sao **cointegrados de ordem $(d, b)$**, denotado $\mathbf{y}_t \sim CI(d, b)$, se:

1. Cada componente $y_{it} \sim I(d)$
2. Existe um vetor $\boldsymbol{\beta} \neq \mathbf{0}$ tal que $\boldsymbol{\beta}^\top \mathbf{y}_t \sim I(d-b)$, com $b > 0$

O vetor $\boldsymbol{\beta}$ e chamado **vetor de cointegracao**. O caso mais importante
na pratica e $d = b = 1$: variaveis $I(1)$ cuja combinacao linear e $I(0)$.

### 1.2 Intuicao Economica

Cointegracao captura a ideia de **equilibrio de longo prazo**: variaveis que
individualmente sao nao-estacionarias mas mantem uma relacao estavel ao
longo do tempo. Desvios do equilibrio ($\boldsymbol{\beta}^\top \mathbf{y}_t$) sao
estacionarios — o sistema sempre retorna a vizinhanca do equilibrio.

!!! note "Exemplos Classicos"
    - Consumo e renda disponivel (hipotese da renda permanente)
    - Taxas de juros de curto e longo prazo (estrutura a termo)
    - Precos spot e futures (lei do preco unico)
    - Taxa de cambio e fundamentos macroeconomicos (PPP)

### 1.3 Rank de Cointegracao

Se $\mathbf{y}_t$ e um vetor $K$-dimensional de variaveis $I(1)$, o **rank de cointegracao** $r$
e o numero de vetores de cointegracao linearmente independentes, com $0 \leq r \leq K$:

| Rank | Interpretacao |
|---|---|
| $r = 0$ | Nenhuma cointegracao; $\diff \mathbf{y}_t$ e estacionario (VAR em diferencas) |
| $0 < r < K$ | Cointegracao; usar VECM com $r$ relacoes de equilibrio |
| $r = K$ | Todas as variaveis sao $I(0)$; usar VAR em niveis |

---

## 2. Do VAR ao VECM

### 2.1 Reparametrizacao

Partindo do VAR($p$) em niveis:

$$
\mathbf{y}_t = \boldsymbol{\nu} + \mathbf{A}_1 \mathbf{y}_{t-1} + \cdots + \mathbf{A}_p \mathbf{y}_{t-p} + \mathbf{u}_t
$$

Subtraindo $\mathbf{y}_{t-1}$ de ambos os lados e reagrupando:

$$
\diff \mathbf{y}_t = \boldsymbol{\nu} + \boldsymbol{\Pi}\, \mathbf{y}_{t-1} + \sum_{i=1}^{p-1} \boldsymbol{\Gamma}_i\, \diff \mathbf{y}_{t-i} + \mathbf{u}_t
$$

onde:

$$
\boldsymbol{\Pi} = \sum_{i=1}^{p} \mathbf{A}_i - \mathbf{I}_K = -\mathbf{A}(1)
$$

$$
\boldsymbol{\Gamma}_i = -\sum_{j=i+1}^{p} \mathbf{A}_j, \quad i = 1, \ldots, p-1
$$

### 2.2 Papel da Matriz $\boldsymbol{\Pi}$

A matriz $\boldsymbol{\Pi}$ contem toda a informacao sobre as relacoes de longo prazo:

- Se $\text{rank}(\boldsymbol{\Pi}) = 0$: $\boldsymbol{\Pi} = \mathbf{0}$, nao ha cointegracao, o VECM reduz-se a um VAR em diferencas
- Se $\text{rank}(\boldsymbol{\Pi}) = r$ com $0 < r < K$: existem $r$ relacoes de cointegracao
- Se $\text{rank}(\boldsymbol{\Pi}) = K$: $\boldsymbol{\Pi}$ e invertivel, $\mathbf{y}_t$ e estacionario

---

## 3. Decomposicao $\boldsymbol{\Pi} = \boldsymbol{\alpha}\boldsymbol{\beta}^\top$

### 3.1 Loading e Cointegrating Vectors

Quando $\text{rank}(\boldsymbol{\Pi}) = r < K$, existem matrizes $\boldsymbol{\alpha}$ e $\boldsymbol{\beta}$
de dimensao $K \times r$ com rank $r$ tais que:

$$
\boldsymbol{\Pi} = \boldsymbol{\alpha}\boldsymbol{\beta}^\top
$$

onde:

- $\boldsymbol{\beta}$ e a matriz de **vetores de cointegracao**: as colunas de $\boldsymbol{\beta}$ definem as $r$ relacoes de equilibrio $\boldsymbol{\beta}^\top \mathbf{y}_t \sim I(0)$
- $\boldsymbol{\alpha}$ e a matriz de **loading** (ajustamento): a linha $i$ de $\boldsymbol{\alpha}$ indica a velocidade com que a variavel $i$ se ajusta aos desvios do equilibrio

O VECM com a decomposicao:

$$
\diff \mathbf{y}_t = \boldsymbol{\nu} + \boldsymbol{\alpha}\boldsymbol{\beta}^\top \mathbf{y}_{t-1} + \sum_{i=1}^{p-1} \boldsymbol{\Gamma}_i\, \diff \mathbf{y}_{t-i} + \mathbf{u}_t
$$

### 3.2 Nao-Unicidade e Identificacao

A decomposicao $\boldsymbol{\Pi} = \boldsymbol{\alpha}\boldsymbol{\beta}^\top$ nao e unica: para qualquer
matriz invertivel $\mathbf{Q}$ de dimensao $r \times r$:

$$
\boldsymbol{\alpha}\boldsymbol{\beta}^\top = (\boldsymbol{\alpha}\mathbf{Q})(\boldsymbol{\beta}\mathbf{Q}^{-\top})^\top = \boldsymbol{\alpha}^*\boldsymbol{\beta}^{*\top}
$$

!!! warning "Normalizacao"
    E necessario impor **restricoes de identificacao** para obter $\boldsymbol{\beta}$ unico.
    A normalizacao mais comum e:

    $$
    \boldsymbol{\beta} = \begin{pmatrix} \mathbf{I}_r \\ \boldsymbol{\beta}_2 \end{pmatrix}
    $$

    onde as primeiras $r$ linhas formam a identidade (normalizacao de Phillips).
    Isso requer a escolha adequada das $r$ variaveis "normalizadoras".

### 3.3 Interpretacao Economica

Para um sistema bivariado ($K=2$, $r=1$) com $\boldsymbol{\beta} = (1, -\beta_2)^\top$:

$$
\diff y_{1t} = \alpha_1 (y_{1,t-1} - \beta_2\, y_{2,t-1}) + \cdots + u_{1t}
$$

$$
\diff y_{2t} = \alpha_2 (y_{1,t-1} - \beta_2\, y_{2,t-1}) + \cdots + u_{2t}
$$

O termo $ec_{t-1} = y_{1,t-1} - \beta_2\, y_{2,t-1}$ e o **desvio do equilibrio**.
Se $\alpha_1 < 0$, a variavel $y_1$ se ajusta para restaurar o equilibrio;
se $\alpha_2 > 0$, a variavel $y_2$ tambem contribui para o ajuste.

---

## 4. Procedimento de Johansen

### 4.1 Formulacao

O procedimento de Johansen (1988, 1991) estima simultaneamente $r$, $\boldsymbol{\alpha}$ e
$\boldsymbol{\beta}$ via **maxima verossimilhanca de informacao completa (FIML)**.

O ponto de partida e o VECM:

$$
\diff \mathbf{y}_t = \boldsymbol{\nu} + \boldsymbol{\alpha}\boldsymbol{\beta}^\top \mathbf{y}_{t-1} + \sum_{i=1}^{p-1} \boldsymbol{\Gamma}_i\, \diff \mathbf{y}_{t-i} + \mathbf{u}_t
$$

### 4.2 Reduced Rank Regression

O procedimento envolve dois passos de regressao auxiliar:

**Passo 1.** Regredir $\diff \mathbf{y}_t$ e $\mathbf{y}_{t-1}$ nos regressores de curto prazo
$(\mathbf{1}, \diff\mathbf{y}_{t-1}, \ldots, \diff\mathbf{y}_{t-p+1})$ e obter os residuos $\mathbf{R}_{0t}$ e $\mathbf{R}_{1t}$:

$$
\diff \mathbf{y}_t = \hat{\boldsymbol{\Psi}}_0 \mathbf{Z}_t + \mathbf{R}_{0t}
$$

$$
\mathbf{y}_{t-1} = \hat{\boldsymbol{\Psi}}_1 \mathbf{Z}_t + \mathbf{R}_{1t}
$$

onde $\mathbf{Z}_t = (\mathbf{1}, \diff\mathbf{y}_{t-1}^\top, \ldots, \diff\mathbf{y}_{t-p+1}^\top)^\top$.

**Passo 2.** Computar as matrizes de momentos:

$$
\mathbf{S}_{ij} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{R}_{it}\, \mathbf{R}_{jt}^\top, \quad i, j \in \{0, 1\}
$$

**Passo 3.** Resolver o problema de eigenvalues generalizado:

$$
|\lambda\, \mathbf{S}_{11} - \mathbf{S}_{10}\, \mathbf{S}_{00}^{-1}\, \mathbf{S}_{01}| = 0
$$

Sejam $\hat{\lambda}_1 \geq \hat{\lambda}_2 \geq \cdots \geq \hat{\lambda}_K \geq 0$ os eigenvalues
ordenados e $\hat{\mathbf{v}}_1, \ldots, \hat{\mathbf{v}}_K$ os eigenvectors correspondentes,
normalizados tal que $\hat{\mathbf{v}}_i^\top \mathbf{S}_{11} \hat{\mathbf{v}}_j = \delta_{ij}$.

### 4.3 Estimadores ML

$$
\hat{\boldsymbol{\beta}} = (\hat{\mathbf{v}}_1, \ldots, \hat{\mathbf{v}}_r)
$$

$$
\hat{\boldsymbol{\alpha}} = \mathbf{S}_{01}\, \hat{\boldsymbol{\beta}}
$$

A log-verossimilhanca maximizada:

$$
\ln L_{\max}(r) = -\frac{TK}{2}(1 + \ln 2\pi) - \frac{T}{2}\ln|\mathbf{S}_{00}| - \frac{T}{2}\sum_{i=1}^{r}\ln(1 - \hat{\lambda}_i)
$$

### 4.4 Testes de Rank

=== "Trace Test"

    Testa $H_0: r \leq r_0$ contra $H_1: r > r_0$:

    $$
    \lambda_{\text{trace}}(r_0) = -T \sum_{i=r_0+1}^{K} \ln(1 - \hat{\lambda}_i)
    $$

    Procedimento sequencial: testar $r_0 = 0, 1, 2, \ldots$ ate nao rejeitar $H_0$.

=== "Max-Eigenvalue Test"

    Testa $H_0: r = r_0$ contra $H_1: r = r_0 + 1$:

    $$
    \lambda_{\max}(r_0) = -T \ln(1 - \hat{\lambda}_{r_0+1})
    $$

    Mais poderoso contra alternativas especificas, mas pode gerar inconsistencias sequenciais.

!!! warning "Distribuicao Nao-Padrao"
    As estatisticas trace e max-eigenvalue **nao** seguem distribuicoes $\chi^2$ padrao.
    Seus valores criticos dependem de:

    - $K - r_0$ (dimensao sob $H_0$)
    - Especificacao dos termos deterministicos
    - Presenca de variaveis exogenas

    Johansen (1995) e Osterwald-Lenum (1992) tabulam os valores criticos para os
    cinco casos deterministicos.

---

## 5. Termos Deterministicos: Os Cinco Casos de Johansen

A especificacao dos termos deterministicos no VECM afeta fundamentalmente os
resultados de cointegracao. Johansen (1995) distingue cinco casos:

### Formulacao Geral

$$
\diff \mathbf{y}_t = \boldsymbol{\alpha}\left[\boldsymbol{\beta}^\top \mathbf{y}_{t-1} + \boldsymbol{\rho}_0 + \boldsymbol{\rho}_1 t\right] + \boldsymbol{\mu}_0 + \boldsymbol{\mu}_1 t + \sum_{i=1}^{p-1} \boldsymbol{\Gamma}_i \diff\mathbf{y}_{t-i} + \mathbf{u}_t
$$

| Caso | Intercepto | Tendencia | Descricao |
|---|---|---|---|
| 1 | $\boldsymbol{\mu}_0 = \mathbf{0}$, $\boldsymbol{\rho}_0 = \mathbf{0}$ | $\boldsymbol{\mu}_1 = \mathbf{0}$, $\boldsymbol{\rho}_1 = \mathbf{0}$ | Sem constante, sem tendencia |
| 2 | $\boldsymbol{\mu}_0 = \boldsymbol{\alpha}\boldsymbol{\rho}_0$ (restrito) | $\boldsymbol{\mu}_1 = \mathbf{0}$, $\boldsymbol{\rho}_1 = \mathbf{0}$ | Constante na relacao de cointegracao |
| 3 | $\boldsymbol{\mu}_0$ livre | $\boldsymbol{\mu}_1 = \mathbf{0}$, $\boldsymbol{\rho}_1 = \mathbf{0}$ | Constante irrestrita (tendencia linear nos niveis) |
| 4 | $\boldsymbol{\mu}_0$ livre | $\boldsymbol{\mu}_1 = \boldsymbol{\alpha}\boldsymbol{\rho}_1$ (restrita) | Tendencia na relacao de cointegracao |
| 5 | $\boldsymbol{\mu}_0$ livre | $\boldsymbol{\mu}_1$ livre | Tendencia quadratica nos niveis |

!!! tip "Escolha na Pratica"
    - **Caso 2**: mais comum em aplicacoes macroeconomicas (equilibrios com intercepto)
    - **Caso 3**: permite drift linear nos niveis das variaveis
    - **Caso 4**: raramente usado; tendencia nos dados e tambem na relacao de cointegracao
    - **Casos 1 e 5**: raramente adequados para dados economicos

    Johansen (1995) recomenda o **procedimento Pantula**: testar sequencialmente
    do caso mais restritivo ao menos restritivo, parando no primeiro que nao
    rejeita a hipotese nula.

---

## 6. Estimacao FIML

### 6.1 Propriedades do Estimador de Johansen

O estimador de Johansen e um estimador de **maxima verossimilhanca de informacao
completa (FIML)**, com propriedades:

**Teorema (Johansen, 1991).** Sob condicoes de regularidade:

1. $\hat{\boldsymbol{\beta}}$ e **super-consistente**: converge a taxa $T$ (nao $\sqrt{T}$):

$$
T(\hat{\boldsymbol{\beta}} - \boldsymbol{\beta}_0) \xrightarrow{d} \text{funcional do movimento Browniano}
$$

2. $\hat{\boldsymbol{\alpha}}$ e $\hat{\boldsymbol{\Gamma}}_i$ convergem a taxa $\sqrt{T}$:

$$
\sqrt{T}(\hat{\boldsymbol{\alpha}} - \boldsymbol{\alpha}_0) \xrightarrow{d} N(\mathbf{0}, \boldsymbol{\Sigma}_\alpha)
$$

3. A inferencia sobre $\boldsymbol{\alpha}$ pode ser realizada com distribuicoes assintoticas
   padrao ($\chi^2$, normal), mas a inferencia sobre $\boldsymbol{\beta}$ requer distribuicoes
   nao-padrao.

### 6.2 Testes sobre $\boldsymbol{\beta}$

Hipoteses lineares sobre os vetores de cointegracao $H_0: \boldsymbol{\beta} = \mathbf{H}\boldsymbol{\varphi}$
(onde $\mathbf{H}$ e uma matriz de restricao conhecida e $\boldsymbol{\varphi}$ e o parametro livre)
sao testadas via razao de verossimilhanca:

$$
LR = -T \sum_{i=1}^{r} \left[\ln(1 - \hat{\lambda}_i^*) - \ln(1 - \hat{\lambda}_i)\right] \xrightarrow{d} \chi^2(\nu)
$$

onde $\hat{\lambda}_i^*$ sao os eigenvalues sob a restricao e $\nu$ sao os graus de liberdade.

### 6.3 Testes sobre $\boldsymbol{\alpha}$

A hipotese de **fraca exogeneidade** $H_0: \alpha_{i\cdot} = \mathbf{0}$ (a variavel $i$ nao
se ajusta ao equilibrio) e testada via LR com distribuicao $\chi^2(r)$.

---

## 7. Representacao de Tendencia Comum

### 7.1 Decomposicao de Gonzalo-Granger

Gonzalo & Granger (1995) decompoe $\mathbf{y}_t$ em componentes permanentes e transitorios:

$$
\mathbf{y}_t = \boldsymbol{\beta}_\perp (\boldsymbol{\alpha}_\perp^\top \boldsymbol{\beta}_\perp)^{-1} \boldsymbol{\alpha}_\perp^\top \mathbf{y}_t + \boldsymbol{\alpha}(\boldsymbol{\beta}^\top \boldsymbol{\alpha})^{-1} \boldsymbol{\beta}^\top \mathbf{y}_t
$$

onde:

- $\boldsymbol{\alpha}_\perp^\top \mathbf{y}_t$ sao as $K - r$ **tendencias comuns** (componente permanente)
- $\boldsymbol{\beta}^\top \mathbf{y}_t$ sao as $r$ **relacoes de cointegracao** (componente transitorio)
- $\boldsymbol{\alpha}_\perp$ e $\boldsymbol{\beta}_\perp$ sao os complementos ortogonais ($\boldsymbol{\alpha}^\top \boldsymbol{\alpha}_\perp = \mathbf{0}$, $\boldsymbol{\beta}^\top \boldsymbol{\beta}_\perp = \mathbf{0}$)

### 7.2 Interpretacao

Em um sistema com $K$ variaveis e $r$ relacoes de cointegracao:

- Existem $K - r$ tendencias estocasticas comuns (passeios aleatorios que dirigem o comportamento de longo prazo)
- As $r$ relacoes de cointegracao "eliminam" estas tendencias, produzindo combinacoes estacionarias

---

## 8. Metodo de Dois Passos de Engle-Granger

### 8.1 Procedimento

Como alternativa ao procedimento de Johansen, Engle & Granger (1987) propuseram:

**Passo 1.** Estimar a regressao de cointegracao via OLS:

$$
y_{1t} = \hat{\beta}_0 + \hat{\beta}_2 y_{2t} + \cdots + \hat{\beta}_K y_{Kt} + \hat{e}_t
$$

**Passo 2.** Testar se os residuos $\hat{e}_t$ sao estacionarios via teste ADF.
Se rejeitar a hipotese de raiz unitaria, ha evidencia de cointegracao.

### 8.2 Limitacoes

!!! warning "Desvantagens do Metodo de Engle-Granger"
    - Assume **no maximo uma** relacao de cointegracao
    - Resultado depende da variavel escolhida como dependente (nao e invariante a normalizacao)
    - Estimadores de segundo passo tem distribuicao complexa
    - Nao permite testar restricoes sobre $\boldsymbol{\beta}$

    Para $K > 2$ ou quando multiplas relacoes de cointegracao sao possiveis,
    o procedimento de Johansen e fortemente preferido.

---

## 9. Previsao com VECM

### 9.1 Previsao Iterativa

A previsao $h$ passos a frente e obtida iterativamente:

$$
\diff \hat{\mathbf{y}}_{T+h|T} = \hat{\boldsymbol{\nu}} + \hat{\boldsymbol{\alpha}}\hat{\boldsymbol{\beta}}^\top \hat{\mathbf{y}}_{T+h-1|T} + \sum_{i=1}^{p-1} \hat{\boldsymbol{\Gamma}}_i \diff\hat{\mathbf{y}}_{T+h-i|T}
$$

### 9.2 Vantagem sobre VAR em Diferencas

O VECM utiliza informacao sobre as relacoes de longo prazo, gerando previsoes
que respeitam o equilibrio. Um VAR em diferencas (que ignora a cointegracao)
pode produzir previsoes de longo prazo que divergem sistematicamente do equilibrio.

---

## Referencias

- Engle, R. F. & Granger, C. W. J. (1987). "Co-integration and Error Correction: Representation, Estimation, and Testing." *Econometrica*, 55(2), 251-276.
- Gonzalo, J. & Granger, C. W. J. (1995). "Estimation of Common Long-Memory Components in Cointegrated Systems." *Journal of Business & Economic Statistics*, 13(1), 27-35.
- Johansen, S. (1988). "Statistical Analysis of Cointegration Vectors." *Journal of Economic Dynamics and Control*, 12(2-3), 231-254.
- Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models." *Econometrica*, 59(6), 1551-1580.
- Johansen, S. (1995). *Likelihood-Based Inference in Cointegrated Vector Autoregressive Models*. Oxford University Press.
- Osterwald-Lenum, M. (1992). "A Note with Quantiles of the Asymptotic Distribution of the Maximum Likelihood Cointegration Rank Test Statistics." *Oxford Bulletin of Economics and Statistics*, 54(3), 461-472.
- Phillips, P. C. B. (1991). "Optimal Inference in Cointegrated Systems." *Econometrica*, 59(2), 283-306.

---

## Veja Tambem

- [VAR](var-theory.md) — Modelo VAR e suas propriedades
- [SVAR](svar-theory.md) — Identificacao estrutural
- [ARIMA](arima-theory.md) — Raizes unitarias e diferenciacao no caso univariado
