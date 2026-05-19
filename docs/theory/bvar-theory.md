---
title: "BVAR: Fundamentos Teoricos"
description: "Teoria completa de modelos VAR Bayesianos — inferencia prior-posterior, Minnesota prior, Normal-Wishart conjugado, SSVS, Gibbs sampler e selecao de modelos via marginal likelihood."
---

# BVAR: Fundamentos Teoricos

!!! abstract "Resumo"
    Esta pagina apresenta a fundamentacao matematica rigorosa dos modelos de
    Vetores Autorregressivos Bayesianos (BVAR). Partindo dos principios de
    inferencia Bayesiana, cobrimos as principais distribuicoes a priori
    (Minnesota, Normal-Wishart, SSVS), os algoritmos de amostragem posterior
    e a selecao de modelos via verossimilhanca marginal. O tratamento segue
    Koop & Korobilis (2010), Karlsson (2013) e Litterman (1986).

---

## 1. Inferencia Bayesiana: Visao Geral

### 1.1 Teorema de Bayes

O paradigma Bayesiano combina informacao a priori com os dados observados
para obter a distribuicao a posteriori dos parametros:

$$
p(\boldsymbol{\theta} | \mathbf{Y}) = \frac{p(\mathbf{Y} | \boldsymbol{\theta})\, p(\boldsymbol{\theta})}{p(\mathbf{Y})}
$$

onde:

- $p(\boldsymbol{\theta})$ e a **distribuicao a priori** (prior)
- $p(\mathbf{Y} | \boldsymbol{\theta})$ e a **funcao de verossimilhanca** (likelihood)
- $p(\boldsymbol{\theta} | \mathbf{Y})$ e a **distribuicao a posteriori** (posterior)
- $p(\mathbf{Y}) = \int p(\mathbf{Y} | \boldsymbol{\theta})\, p(\boldsymbol{\theta})\, d\boldsymbol{\theta}$ e a **verossimilhanca marginal** (marginal likelihood)

!!! note "Prior x Likelihood = Posterior"
    A essencia da inferencia Bayesiana e a combinacao de crencas a priori
    (codificadas na prior) com a evidencia dos dados (codificada na likelihood).
    Conforme a amostra cresce, a posterior e dominada pela likelihood, e a
    influencia da prior desaparece assintoticamente.

### 1.2 Motivacao para o BVAR

Em modelos VAR de dimensao moderada a alta, o numero de parametros cresce
como $K^2 p + K$ (interceptos). Para $K = 10$ variaveis e $p = 4$ lags, sao
$410$ parametros, levando a:

- **Sobreparametrizacao**: estimativas OLS imprecisas
- **Curse of dimensionality**: performance de previsao degradada
- **Instabilidade**: matrizes de covariancia mal condicionadas

A abordagem Bayesiana resolve esses problemas via **regularizacao probabilistica**:
a prior encolhe (shrinks) os parametros em direcao a valores plausiveis, reduzindo
a variancia das estimativas ao custo de um vies controlado.

---

## 2. Modelo VAR em Forma Matricial

### 2.1 Especificacao

O modelo VAR(p) para $\mathbf{y}_t \in \mathbb{R}^K$:

$$
\mathbf{y}_t = \boldsymbol{\nu} + \mathbf{A}_1 \mathbf{y}_{t-1} + \cdots + \mathbf{A}_p \mathbf{y}_{t-p} + \mathbf{u}_t, \quad \mathbf{u}_t \sim N(\mathbf{0}, \boldsymbol{\Sigma})
$$

### 2.2 Forma Empilhada

Definindo $\mathbf{x}_t = (1, \mathbf{y}_{t-1}^\top, \ldots, \mathbf{y}_{t-p}^\top)^\top \in \mathbb{R}^{m}$
com $m = Kp + 1$ e $\mathbf{B} = (\boldsymbol{\nu}, \mathbf{A}_1, \ldots, \mathbf{A}_p)^\top \in \mathbb{R}^{m \times K}$:

$$
\mathbf{y}_t^\top = \mathbf{x}_t^\top \mathbf{B} + \mathbf{u}_t^\top
$$

Empilhando $T$ observacoes:

$$
\underbrace{\mathbf{Y}}_{T \times K} = \underbrace{\mathbf{X}}_{T \times m}\, \underbrace{\mathbf{B}}_{m \times K} + \underbrace{\mathbf{U}}_{T \times K}
$$

### 2.3 Verossimilhanca

Sob normalidade dos erros, a log-verossimilhanca e:

$$
\ln p(\mathbf{Y} | \mathbf{B}, \boldsymbol{\Sigma}) = -\frac{TK}{2} \ln(2\pi) - \frac{T}{2} \ln|\boldsymbol{\Sigma}| - \frac{1}{2} \text{tr}\left[\boldsymbol{\Sigma}^{-1} (\mathbf{Y} - \mathbf{X}\mathbf{B})^\top (\mathbf{Y} - \mathbf{X}\mathbf{B})\right]
$$

Equivalentemente, vetorizando $\boldsymbol{\beta} = \text{vec}(\mathbf{B})$:

$$
\text{vec}(\mathbf{Y}^\top) | \boldsymbol{\beta}, \boldsymbol{\Sigma} \sim N\left((\mathbf{X} \otimes \mathbf{I}_K)\, \boldsymbol{\beta},\; \mathbf{I}_T \otimes \boldsymbol{\Sigma}\right)
$$

---

## 3. Minnesota Prior (Litterman, 1986)

### 3.1 Intuicao

A **Minnesota prior**, proposta por Litterman (1986) e Doan, Litterman & Sims (1984)
no Federal Reserve Bank of Minneapolis, codifica a crenca de que cada variavel
segue um passeio aleatorio com drift como caso central:

$$
y_{it} = y_{i,t-1} + \varepsilon_{it}
$$

Os coeficientes sao encolhidos em direcao a $\mathbf{A}_1 = \mathbf{I}_K$ (diagonal unitaria
no primeiro lag) e $\mathbf{A}_j = \mathbf{0}$ para $j > 1$.

### 3.2 Especificacao

A prior Minnesota e uma distribuicao normal independente sobre cada coeficiente
$a_{ij}^{(\ell)}$ (elemento $(i,j)$ da matriz $\mathbf{A}_\ell$):

$$
a_{ij}^{(\ell)} \sim N\left(\delta_{ij} \cdot \mathbf{1}[\ell = 1],\; v_{ij}^{(\ell)}\right)
$$

onde $\delta_{ij}$ e o delta de Kronecker e a variancia a priori e:

$$
v_{ij}^{(\ell)} = \begin{cases}
\displaystyle \frac{\lambda_1^2}{\ell^{\lambda_3}} & \text{se } i = j \text{ (own lag)} \\[10pt]
\displaystyle \frac{\lambda_1^2 \lambda_2}{\ell^{\lambda_3}} \cdot \frac{\sigma_i^2}{\sigma_j^2} & \text{se } i \neq j \text{ (cross lag)}
\end{cases}
$$

### 3.3 Hiperparametros

| Hiperparametro | Interpretacao | Valor tipico |
|---|---|---|
| $\lambda_1$ | Overall tightness (shrinkage global) | $0.1$ a $0.2$ |
| $\lambda_2$ | Cross-variable shrinkage (relativo a own) | $0.5$ a $1.0$ |
| $\lambda_3$ | Lag decay (taxa de decaimento por lag) | $1$ ou $2$ |
| $\sigma_i^2$ | Escala (variancia residual de AR(p) univariado para variavel $i$) | Estimada dos dados |

!!! tip "Interpretacao dos Hiperparametros"
    - $\lambda_1 \to 0$: prior dominante, todos os coeficientes encolhidos para random walk
    - $\lambda_1 \to \infty$: prior vaga, estimativas convergem para OLS
    - $\lambda_2 < 1$: coeficientes cruzados mais encolhidos que os proprios
    - $\lambda_3 > 1$: lags distantes mais fortemente encolhidos

### 3.4 Implementacao via Dummy Observations

A Minnesota prior pode ser implementada de forma elegante adicionando
**observacoes artificiais** (dummy observations) a matriz de dados. Definindo:

$$
\mathbf{Y}_d = \begin{pmatrix}
\text{diag}(\delta_1 \sigma_1, \ldots, \delta_K \sigma_K) / \lambda_1 \\
\mathbf{0}_{K(p-1) \times K} \\
\text{diag}(\sigma_1, \ldots, \sigma_K) \\
\mathbf{0}_{1 \times K}
\end{pmatrix}, \quad
\mathbf{X}_d = \begin{pmatrix}
\mathbf{J}_p \otimes \text{diag}(\sigma_1, \ldots, \sigma_K) / \lambda_1 & \mathbf{0} \\
\mathbf{0} & \mathbf{0} \\
\mathbf{0} & \epsilon
\end{pmatrix}
$$

O estimador resultante e equivalente ao estimador GLS com a prior Minnesota.

---

## 4. Prior Normal-Wishart Conjugada

### 4.1 Especificacao

A prior **Natural Conjugada** (ou Normal-Wishart) especifica:

$$
\begin{aligned}
\boldsymbol{\beta} | \boldsymbol{\Sigma} &\sim N\left(\boldsymbol{\beta}_0,\; \boldsymbol{\Sigma} \otimes \boldsymbol{\Omega}_0\right) \\
\boldsymbol{\Sigma} &\sim \mathcal{IW}(\mathbf{S}_0, \nu_0)
\end{aligned}
$$

onde:

- $\boldsymbol{\beta}_0$ e a media a priori dos coeficientes (tipicamente configurada como na Minnesota)
- $\boldsymbol{\Omega}_0$ e a matriz de covariancia a priori dos coeficientes (mesma para todas as equacoes)
- $\mathbf{S}_0$ e a matriz de escala a priori para $\boldsymbol{\Sigma}$
- $\nu_0 > K - 1$ sao os graus de liberdade a priori

### 4.2 Posterior Analitica

A conjugacao produz posteriors em forma fechada:

$$
\begin{aligned}
\boldsymbol{\beta} | \boldsymbol{\Sigma}, \mathbf{Y} &\sim N\left(\bar{\boldsymbol{\beta}},\; \boldsymbol{\Sigma} \otimes \bar{\boldsymbol{\Omega}}\right) \\
\boldsymbol{\Sigma} | \mathbf{Y} &\sim \mathcal{IW}(\bar{\mathbf{S}}, \bar{\nu})
\end{aligned}
$$

onde as quantidades posteriors sao:

$$
\begin{aligned}
\bar{\boldsymbol{\Omega}} &= \left(\boldsymbol{\Omega}_0^{-1} + \mathbf{X}^\top \mathbf{X}\right)^{-1} \\
\bar{\mathbf{B}} &= \bar{\boldsymbol{\Omega}} \left(\boldsymbol{\Omega}_0^{-1} \mathbf{B}_0 + \mathbf{X}^\top \mathbf{X}\, \hat{\mathbf{B}}_{\text{OLS}}\right) \\
\bar{\nu} &= \nu_0 + T \\
\bar{\mathbf{S}} &= \mathbf{S}_0 + \hat{\mathbf{S}} + (\hat{\mathbf{B}}_{\text{OLS}} - \mathbf{B}_0)^\top \left(\boldsymbol{\Omega}_0 + (\mathbf{X}^\top \mathbf{X})^{-1}\right)^{-1} (\hat{\mathbf{B}}_{\text{OLS}} - \mathbf{B}_0)
\end{aligned}
$$

com $\hat{\mathbf{B}}_{\text{OLS}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{Y}$ e
$\hat{\mathbf{S}} = (\mathbf{Y} - \mathbf{X}\hat{\mathbf{B}}_{\text{OLS}})^\top (\mathbf{Y} - \mathbf{X}\hat{\mathbf{B}}_{\text{OLS}})$.

!!! note "Media Posterior como Media Ponderada"
    A media posterior $\bar{\mathbf{B}}$ e uma **media ponderada matricial** entre a
    media a priori $\mathbf{B}_0$ e o estimador OLS $\hat{\mathbf{B}}_{\text{OLS}}$,
    com pesos determinados pela precisao relativa da prior ($\boldsymbol{\Omega}_0^{-1}$)
    e dos dados ($\mathbf{X}^\top \mathbf{X}$).

### 4.3 Previsao Posterior

A distribuicao preditiva $h$ passos a frente e obtida integrando sobre
a incerteza parametrica:

$$
p(\mathbf{y}_{T+h} | \mathbf{Y}) = \int p(\mathbf{y}_{T+h} | \boldsymbol{\theta}, \mathbf{Y})\, p(\boldsymbol{\theta} | \mathbf{Y})\, d\boldsymbol{\theta}
$$

Na pratica, isso e computado via simulacao de Monte Carlo a partir da posterior.

---

## 5. Prior SSVS (Stochastic Search Variable Selection)

### 5.1 Motivacao

A prior **SSVS** de George, Sun & Ni (2008) permite **selecao de variaveis
Bayesiana** dentro do VAR, decidindo automaticamente quais coeficientes
sao relevantes (nao-zero) e quais devem ser excluidos (zero).

### 5.2 Especificacao Hierarquica

Para cada coeficiente $\beta_j$, a prior SSVS e uma mistura de duas normais:

$$
\beta_j | \gamma_j \sim (1 - \gamma_j)\, N(0, \tau_{0j}^2) + \gamma_j\, N(0, \tau_{1j}^2)
$$

onde:

- $\gamma_j \in \{0, 1\}$ e um indicador de inclusao
- $\tau_{0j}^2$ e uma variancia **pequena** (coeficiente essencialmente zero)
- $\tau_{1j}^2$ e uma variancia **grande** (coeficiente irrestrito)
- $\gamma_j | q_j \sim \text{Bernoulli}(q_j)$

!!! warning "Escolha de $\tau_0$ e $\tau_1$"
    A performance do SSVS depende criticamente da razao $\tau_{1j}/\tau_{0j}$.
    George et al. (2008) recomendam $\tau_{0j} = 0.1 \cdot \text{se}(\hat{\beta}_j^{\text{OLS}})$
    e $\tau_{1j} = 10 \cdot \text{se}(\hat{\beta}_j^{\text{OLS}})$, calibrando
    as escalas a partir das estimativas frequentistas.

### 5.3 Prior para a Covariancia

O SSVS pode ser estendido a matriz de covariancia $\boldsymbol{\Sigma}$.
Decompondo $\boldsymbol{\Sigma}^{-1} = \boldsymbol{\Psi}^\top \boldsymbol{\Psi}$
com $\boldsymbol{\Psi}$ triangular superior, aplica-se a mistura de normais
aos elementos fora da diagonal de $\boldsymbol{\Psi}$:

$$
\psi_{ij} | \omega_{ij} \sim (1 - \omega_{ij})\, N(0, \kappa_{0,ij}^2) + \omega_{ij}\, N(0, \kappa_{1,ij}^2)
$$

---

## 6. Gibbs Sampler para BVAR

### 6.1 Principio

O **Gibbs sampler** e um algoritmo de Monte Carlo via Cadeias de Markov (MCMC)
que gera amostras da posterior conjunta $p(\boldsymbol{\beta}, \boldsymbol{\Sigma} | \mathbf{Y})$
amostrando iterativamente das condicionais completas.

### 6.2 Algoritmo para Normal-Wishart

O Gibbs sampler para o BVAR com prior Natural Conjugada e direto, pois
as condicionais completas possuem forma fechada:

**Algoritmo: Gibbs Sampler para BVAR Normal-Wishart**

Para $s = 1, \ldots, S$ (numero de iteracoes):

1. **Amostrar** $\boldsymbol{\Sigma}^{(s)} | \mathbf{Y} \sim \mathcal{IW}(\bar{\mathbf{S}}, \bar{\nu})$

2. **Amostrar** $\boldsymbol{\beta}^{(s)} | \boldsymbol{\Sigma}^{(s)}, \mathbf{Y} \sim N(\bar{\boldsymbol{\beta}},\; \boldsymbol{\Sigma}^{(s)} \otimes \bar{\boldsymbol{\Omega}})$

3. **Armazenar** $(\boldsymbol{\beta}^{(s)}, \boldsymbol{\Sigma}^{(s)})$

Descartar as primeiras $S_0$ iteracoes como **burn-in** e usar as restantes
$S - S_0$ como amostras aproximadas da posterior.

### 6.3 Algoritmo para SSVS

O Gibbs sampler com prior SSVS adiciona passos para os indicadores:

Para $s = 1, \ldots, S$:

1. **Amostrar** $\boldsymbol{\gamma}^{(s)} | \boldsymbol{\beta}^{(s-1)}, \mathbf{q}$:
   para cada $j$, a probabilidade condicional de inclusao e:

$$
P(\gamma_j = 1 | \beta_j, q_j) = \frac{q_j\, \phi(\beta_j; 0, \tau_{1j}^2)}{q_j\, \phi(\beta_j; 0, \tau_{1j}^2) + (1 - q_j)\, \phi(\beta_j; 0, \tau_{0j}^2)}
$$

2. **Amostrar** $\boldsymbol{\Sigma}^{(s)} | \boldsymbol{\beta}^{(s-1)}, \mathbf{Y}$

3. **Amostrar** $\boldsymbol{\beta}^{(s)} | \boldsymbol{\Sigma}^{(s)}, \boldsymbol{\gamma}^{(s)}, \mathbf{Y}$

### 6.4 Diagnostico de Convergencia

!!! warning "Verificacao de Convergencia"
    A convergencia do Gibbs sampler deve ser verificada via:

    - **Traceplots**: inspecao visual das cadeias
    - **Estatistica de Geweke** (1992): teste de igualdade de medias entre
      a primeira e ultima parte da cadeia
    - **Fator de reducao de escala de Gelman-Rubin** (1992): comparacao
      entre-cadeias vs. dentro-da-cadeia ($\hat{R} < 1.1$)
    - **Effective sample size (ESS)**: numero efetivo de amostras independentes

---

## 7. Verossimilhanca Marginal e Comparacao de Modelos

### 7.1 Definicao

A **verossimilhanca marginal** (ou evidencia do modelo) e a probabilidade
dos dados sob o modelo $\mathcal{M}$, integrando sobre todo o espaco parametrico:

$$
p(\mathbf{Y} | \mathcal{M}) = \int p(\mathbf{Y} | \boldsymbol{\theta}, \mathcal{M})\, p(\boldsymbol{\theta} | \mathcal{M})\, d\boldsymbol{\theta}
$$

### 7.2 Forma Fechada (Normal-Wishart)

Para a prior Natural Conjugada, a verossimilhanca marginal possui forma analitica:

$$
\ln p(\mathbf{Y}) = -\frac{TK}{2} \ln(\pi) + \frac{K}{2} \ln\frac{|\boldsymbol{\Omega}_0|}{|\bar{\boldsymbol{\Omega}}|} + \frac{\nu_0}{2} \ln|\mathbf{S}_0| - \frac{\bar{\nu}}{2} \ln|\bar{\mathbf{S}}| + \sum_{k=1}^{K} \left[\ln \Gamma\!\left(\frac{\bar{\nu} + 1 - k}{2}\right) - \ln \Gamma\!\left(\frac{\nu_0 + 1 - k}{2}\right)\right]
$$

### 7.3 Fator de Bayes

O **fator de Bayes** para comparar dois modelos $\mathcal{M}_1$ e $\mathcal{M}_2$:

$$
BF_{12} = \frac{p(\mathbf{Y} | \mathcal{M}_1)}{p(\mathbf{Y} | \mathcal{M}_2)}
$$

| $\ln BF_{12}$ | Evidencia a favor de $\mathcal{M}_1$ |
|---|---|
| $0$ a $1$ | Fraca |
| $1$ a $3$ | Positiva |
| $3$ a $5$ | Forte |
| $> 5$ | Muito forte |

(Escala de Kass & Raftery, 1995)

!!! note "Navalha de Occam Bayesiana"
    A verossimilhanca marginal penaliza automaticamente a complexidade do modelo,
    implementando uma forma natural da navalha de Occam. Modelos mais complexos
    espalham sua probabilidade a priori sobre uma regiao maior do espaco parametrico,
    resultando em menor evidencia marginal quando os dados nao suportam essa
    complexidade adicional.

---

## 8. Calibracao de Hiperparametros

### 8.1 Abordagem Empirica Bayesiana

Os hiperparametros $\boldsymbol{\lambda} = (\lambda_1, \lambda_2, \lambda_3)$ da
Minnesota prior podem ser otimizados maximizando a verossimilhanca marginal:

$$
\hat{\boldsymbol{\lambda}} = \arg\max_{\boldsymbol{\lambda}} \ln p(\mathbf{Y} | \boldsymbol{\lambda})
$$

Esta abordagem, conhecida como **Empirical Bayes** ou **Type-II Maximum Likelihood**,
permite que os dados informem o grau de shrinkage.

### 8.2 Abordagem Hierarquica Completa

Alternativamente, coloca-se uma hiperprior sobre $\boldsymbol{\lambda}$:

$$
p(\boldsymbol{\theta}, \boldsymbol{\lambda} | \mathbf{Y}) \propto p(\mathbf{Y} | \boldsymbol{\theta})\, p(\boldsymbol{\theta} | \boldsymbol{\lambda})\, p(\boldsymbol{\lambda})
$$

e amostra-se $\boldsymbol{\lambda}$ juntamente com $\boldsymbol{\theta}$ no Gibbs sampler.
Giannone, Lenza & Primiceri (2015) propoem uma implementacao eficiente deste esquema.

### 8.3 Grid Search

Uma abordagem pratica e avaliar a verossimilhanca marginal sobre uma grade
de valores de $\boldsymbol{\lambda}$ e selecionar a combinacao otima:

$$
\hat{\boldsymbol{\lambda}} = \arg\max_{\boldsymbol{\lambda} \in \Lambda} \ln p(\mathbf{Y} | \boldsymbol{\lambda})
$$

onde $\Lambda = \{\lambda_1^{(1)}, \ldots\} \times \{\lambda_2^{(1)}, \ldots\} \times \cdots$
e a grade de busca.

---

## 9. Implicacoes Praticas

### 9.1 Quando Usar BVAR

| Cenario | Recomendacao |
|---|---|
| $K$ grande, $T$ pequeno | BVAR fortemente recomendado (regularizacao essencial) |
| Muitos lags necessarios | BVAR com decay forte ($\lambda_3 = 2$) |
| Previsao macroeconomica | BVAR Minnesota (padrao na literatura) |
| Selecao de variaveis | BVAR com SSVS |
| Incerteza nos hiperparametros | Hierarquica completa (Giannone et al., 2015) |

### 9.2 Relacao com Regularizacao Frequentista

!!! tip "Conexao com Ridge e LASSO"
    A Minnesota prior com variancia $\lambda_1^2$ e equivalente a regularizacao
    **Ridge** (penalidade $L_2$) sobre os coeficientes do VAR. A prior SSVS,
    por sua vez, aproxima a regularizacao **LASSO** (penalidade $L_1$) via
    selecao estocastica. Assim, o BVAR pode ser visto como uma generalizacao
    probabilistica das tecnicas de regularizacao frequentistas.

---

## Referencias

- Doan, T., Litterman, R. & Sims, C. A. (1984). "Forecasting and Conditional Projection Using Realistic Prior Distributions." *Econometric Reviews*, 3(1), 1-100.
- Gelman, A. & Rubin, D. B. (1992). "Inference from Iterative Simulation Using Multiple Sequences." *Statistical Science*, 7(4), 457-472.
- George, E. I., Sun, D. & Ni, S. (2008). "Bayesian Stochastic Search for VAR Model Restrictions." *Journal of Econometrics*, 142(1), 553-580.
- Geweke, J. (1992). "Evaluating the Accuracy of Sampling-Based Approaches to the Calculation of Posterior Moments." In *Bayesian Statistics 4*, Oxford University Press.
- Giannone, D., Lenza, M. & Primiceri, G. E. (2015). "Prior Selection for Vector Autoregressions." *Review of Economics and Statistics*, 97(2), 436-451.
- Karlsson, S. (2013). "Forecasting with Bayesian Vector Autoregressions." In *Handbook of Economic Forecasting*, Vol. 2, Elsevier.
- Kass, R. E. & Raftery, A. E. (1995). "Bayes Factors." *Journal of the American Statistical Association*, 90(430), 773-795.
- Koop, G. & Korobilis, D. (2010). "Bayesian Multivariate Time Series Methods for Empirical Macroeconomics." *Foundations and Trends in Econometrics*, 3(4), 267-358.
- Litterman, R. B. (1986). "Forecasting with Bayesian Vector Autoregressions — Five Years of Experience." *Journal of Business & Economic Statistics*, 4(1), 25-38.
- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.

---

## Veja Tambem

- [VAR](var-theory.md) — Fundamentos frequentistas do VAR
- [SVAR](svar-theory.md) — Identificacao estrutural
- [VECM & Cointegracao](vecm-theory.md) — Variaveis integradas e cointegradas
