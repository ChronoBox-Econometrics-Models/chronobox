---
title: "ARIMA: Fundamentos Teoricos"
description: "Teoria completa de modelos ARIMA — processos estocasticos, estacionariedade, Wold decomposition, estimacao MLE via Kalman e propriedades assintoticas."
---

# ARIMA: Fundamentos Teoricos

!!! abstract "Resumo"
    Esta pagina apresenta a fundamentacao matematica rigorosa dos modelos
    ARIMA (AutoRegressive Integrated Moving Average), desde a definicao de
    processos estocasticos ate a estimacao por maxima verossimilhanca via
    filtro de Kalman. O tratamento e formal e dirigido a pesquisadores e
    estudantes de pos-graduacao em econometria.

---

## 1. Processos Estocasticos e Estacionariedade

### 1.1 Definicoes Fundamentais

Um **processo estocastico** e uma colecao de variaveis aleatorias $\{y_t : t \in \mathbb{Z}\}$
definidas em um espaco de probabilidade $(\Omega, \mathcal{F}, P)$.

!!! note "Estacionariedade Fraca (Covariancia)"
    Um processo $\{y_t\}$ e **fracamente estacionario** (ou estacionario de segunda ordem) se:

    1. $\E[y_t] = \mu$ para todo $t$ (media constante)
    2. $\Var(y_t) = \gamma(0) < \infty$ para todo $t$ (variancia finita e constante)
    3. $\Cov(y_t, y_{t-h}) = \gamma(h)$ depende apenas de $h$, nao de $t$

A **funcao de autocovariancia** $\gamma(h) = \Cov(y_t, y_{t-h})$ satisfaz:

- $\gamma(0) \geq 0$
- $\gamma(h) = \gamma(-h)$ (simetria)
- $|\gamma(h)| \leq \gamma(0)$ (desigualdade de Cauchy-Schwarz)
- $\gamma(\cdot)$ e semi-definida positiva

**Estacionariedade estrita** requer que a distribuicao conjunta de $(y_{t_1}, \ldots, y_{t_k})$
seja invariante a translacoes temporais, i.e., identica a de $(y_{t_1+h}, \ldots, y_{t_k+h})$
para todo $h$ e todo conjunto de indices $\{t_1, \ldots, t_k\}$.

### 1.2 Funcao de Autocorrelacao (ACF)

A **funcao de autocorrelacao** e definida como:

$$
\rho(h) = \frac{\gamma(h)}{\gamma(0)} = \Corr(y_t, y_{t-h})
$$

Para processos estacionarios, $\rho(0) = 1$ e $|\rho(h)| \leq 1$ para todo $h$.

### 1.3 Funcao de Autocorrelacao Parcial (PACF)

A **autocorrelacao parcial** $\alpha(h)$ mede a correlacao entre $y_t$ e $y_{t-h}$
apos remover o efeito linear das variaveis intermediarias $y_{t-1}, \ldots, y_{t-h+1}$:

$$
\alpha(h) = \Corr(y_t - P(y_t | y_{t-1}, \ldots, y_{t-h+1}),\; y_{t-h} - P(y_{t-h} | y_{t-1}, \ldots, y_{t-h+1}))
$$

onde $P(\cdot | \cdot)$ denota a projecao linear. A PACF pode ser obtida recursivamente
via algoritmo de Durbin-Levinson.

---

## 2. Teorema de Decomposicao de Wold

!!! tip "Resultado Fundamental"
    O Teorema de Wold (1938) e a base teorica para toda a classe de modelos ARMA.

**Teorema (Wold, 1938).** Todo processo estacionario de covariancia $\{y_t\}$ com media zero
admite a decomposicao unica:

$$
y_t = \sum_{j=0}^{\infty} \psi_j \eps_{t-j} + \eta_t
$$

onde:

- $\psi_0 = 1$ e $\sum_{j=0}^{\infty} \psi_j^2 < \infty$
- $\{\eps_t\}$ e um ruido branco com $\E[\eps_t] = 0$, $\Var(\eps_t) = \sigma^2$
- $\eta_t$ e um componente deterministico (previsivel a partir de seu passado)
- $\eps_t$ e $\eta_s$ sao nao-correlacionados para todo $t, s$

A componente $\sum_{j=0}^{\infty} \psi_j \eps_{t-j}$ e chamada **inovacao** e constitui
o melhor preditor linear de $y_t$ dado seu passado infinito. Para processos puramente
nao-deterministicos ($\eta_t = 0$), a representacao MA($\infty$) e completa.

---

## 3. Operador Backshift e Polinomios AR/MA

### 3.1 Operador Backshift

O **operador backshift** (ou lag) $B$ e definido por $By_t = y_{t-1}$, de modo que
$B^k y_t = y_{t-k}$. O **operador diferenca** e $\diff = 1 - B$.

### 3.2 Processo AR(p)

Um processo **autorregressivo de ordem $p$** e definido por:

$$
\phi(B) y_t = c + \eps_t
$$

onde $\phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p$ e o **polinomio autorregressivo**.

Explicitamente:

$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \eps_t, \quad \eps_t \sim WN(0, \sigma^2)
$$

**Propriedades do AR(p):**

| Caracteristica | Padrao |
|---|---|
| ACF | Decaimento exponencial ou senoidal amortecido |
| PACF | Trunca em lag $p$ ($\alpha(h) = 0$ para $h > p$) |
| Media | $\mu = c / (1 - \phi_1 - \cdots - \phi_p)$ |

### 3.3 Processo MA(q)

Um processo **de medias moveis de ordem $q$**:

$$
y_t = c + \theta(B) \eps_t
$$

onde $\theta(B) = 1 + \theta_1 B + \theta_2 B^2 + \cdots + \theta_q B^q$ e o **polinomio de medias moveis**.

**Propriedades do MA(q):**

| Caracteristica | Padrao |
|---|---|
| ACF | Trunca em lag $q$ ($\rho(h) = 0$ para $h > q$) |
| PACF | Decaimento exponencial ou senoidal amortecido |
| Variancia | $\gamma(0) = \sigma^2 (1 + \theta_1^2 + \cdots + \theta_q^2)$ |

### 3.4 Processo ARMA(p,q)

Combinando ambos:

$$
\phi(B) y_t = c + \theta(B) \eps_t
$$

ou equivalentemente:

$$
y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \eps_t + \sum_{j=1}^{q} \theta_j \eps_{t-j}
$$

---

## 4. Raizes Unitarias e Diferenciacao

### 4.1 Condicao de Estacionariedade

O processo AR(p) e estacionario se e somente se todas as raizes do polinomio
caracteristico $\phi(z) = 1 - \phi_1 z - \cdots - \phi_p z^p = 0$ estao **fora do circulo
unitario** no plano complexo, i.e., $|z_i| > 1$ para todo $i = 1, \ldots, p$.

!!! warning "Raizes Unitarias"
    Se $\phi(z) = 0$ possui uma raiz com $|z| = 1$ (raiz unitaria), o processo e
    **nao-estacionario**. O caso mais simples e o passeio aleatorio $y_t = y_{t-1} + \eps_t$,
    onde $\phi(z) = 1 - z$ tem raiz $z = 1$.

### 4.2 Diferenciacao e Integracao

Um processo $\{y_t\}$ e **integrado de ordem $d$**, denotado $y_t \sim I(d)$, se
$\diff^d y_t$ e estacionario mas $\diff^{d-1} y_t$ nao e.

O **operador diferenca de ordem $d$**:

$$
\diff^d y_t = (1 - B)^d y_t
$$

Para $d = 1$: $\diff y_t = y_t - y_{t-1}$.
Para $d = 2$: $\diff^2 y_t = y_t - 2y_{t-1} + y_{t-2}$.

### 4.3 Modelo ARIMA(p,d,q)

O modelo ARIMA aplica um ARMA(p,q) a serie diferenciada $d$ vezes:

$$
\phi(B)(1-B)^d y_t = c + \theta(B) \eps_t
$$

onde $\phi(B)$ e $\theta(B)$ nao possuem raizes sobre ou dentro do circulo unitario.

### 4.4 SARIMA

O modelo ARIMA sazonal, SARIMA$(p,d,q)(P,D,Q)_s$:

$$
\Phi_P(B^s)\, \phi_p(B)\, (1-B)^d (1-B^s)^D\, y_t = c + \Theta_Q(B^s)\, \theta_q(B)\, \eps_t
$$

onde:

- $\Phi_P(B^s) = 1 - \Phi_1 B^s - \cdots - \Phi_P B^{Ps}$ (AR sazonal)
- $\Theta_Q(B^s) = 1 + \Theta_1 B^s + \cdots + \Theta_Q B^{Qs}$ (MA sazonal)
- $s$ e o periodo sazonal

---

## 5. Invertibilidade

### 5.1 Condicao de Invertibilidade

O processo MA(q) e **invertivel** se todas as raizes de $\theta(z) = 0$ estao fora do
circulo unitario. A invertibilidade garante uma representacao AR($\infty$) convergente:

$$
\theta(B)^{-1} y_t = \pi(B) y_t = \eps_t
$$

onde $\pi(B) = 1 - \pi_1 B - \pi_2 B^2 - \cdots$ com $\sum_{j=1}^{\infty} |\pi_j| < \infty$.

### 5.2 Importancia Pratica

A invertibilidade resolve o problema de **identificabilidade**: para cada processo MA(q)
nao-invertivel, existe um processo MA(q) invertivel com a mesma funcao de autocovariancia.
A restricao de invertibilidade seleciona a representacao unica.

---

## 6. Representacao em Espaco de Estados

### 6.1 Forma Geral

O modelo ARIMA pode ser escrito na forma estado-espaco:

$$
\begin{aligned}
\boldsymbol{\alpha}_{t+1} &= \mathbf{T}\, \boldsymbol{\alpha}_t + \mathbf{R}\, \eta_t, \quad \eta_t \sim N(0, \mathbf{Q}) \\
y_t &= \mathbf{Z}\, \boldsymbol{\alpha}_t + d_t
\end{aligned}
$$

onde $\boldsymbol{\alpha}_t$ e o vetor de estado, $\mathbf{T}$ e a matriz de transicao,
$\mathbf{Z}$ e o vetor de observacao, e $\mathbf{R}$ seleciona as perturbacoes.

### 6.2 Parametrizacao de Harvey

Para um ARIMA(p,d,q) com $r = \max(p+d, q+1)$, a parametrizacao de Harvey (1989) define:

$$
\mathbf{T} = \begin{pmatrix}
\tilde{\phi}_1 & 1 & 0 & \cdots & 0 \\
\tilde{\phi}_2 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\tilde{\phi}_{r-1} & 0 & 0 & \cdots & 1 \\
\tilde{\phi}_r & 0 & 0 & \cdots & 0
\end{pmatrix}, \quad
\mathbf{Z} = \begin{pmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{pmatrix}^\top
$$

onde $\tilde{\phi}(B) = \phi(B)(1-B)^d$ e o polinomio AR generalizado.

---

## 7. Estimacao

### 7.1 Conditional Sum of Squares (CSS)

O metodo CSS condiciona nas primeiras $\max(p+d, q)$ observacoes e minimiza:

$$
S(\boldsymbol{\theta}) = \sum_{t=t_0+1}^{T} \eps_t^2(\boldsymbol{\theta})
$$

onde $\eps_t$ sao os residuos computados recursivamente. E computacionalmente simples
mas **nao e eficiente** para amostras pequenas, pois descarta informacao das observacoes iniciais.

### 7.2 MLE Exata via Filtro de Kalman

A verossimilhanca exata e computada via decomposicao do erro de previsao
(prediction error decomposition) do filtro de Kalman:

$$
\ln L(\boldsymbol{\theta}) = -\frac{T}{2} \ln(2\pi) - \frac{1}{2} \sum_{t=1}^{T} \left[\ln |F_t| + v_t^\top F_t^{-1} v_t \right]
$$

onde:

- $v_t = y_t - \mathbf{Z}\, \boldsymbol{\alpha}_{t|t-1}$ e a **inovacao** (erro de previsao um passo a frente)
- $F_t = \mathbf{Z}\, \mathbf{P}_{t|t-1}\, \mathbf{Z}^\top$ e a variancia da inovacao
- $\boldsymbol{\alpha}_{t|t-1}$ e $\mathbf{P}_{t|t-1}$ sao o estado previsto e sua covariancia

!!! info "Implementacao no ChronoBox"
    O ChronoBox utiliza a biblioteca `kalmanbox` para executar o filtro de Kalman,
    garantindo estabilidade numerica via filtragem em raiz quadrada e tratamento
    adequado de estados difusos para a inicializacao de modelos nao-estacionarios.

**Recursoes do Filtro de Kalman:**

=== "Previsao"

    $$
    \begin{aligned}
    \boldsymbol{\alpha}_{t|t-1} &= \mathbf{T}\, \boldsymbol{\alpha}_{t-1|t-1} \\
    \mathbf{P}_{t|t-1} &= \mathbf{T}\, \mathbf{P}_{t-1|t-1}\, \mathbf{T}^\top + \mathbf{R}\, \mathbf{Q}\, \mathbf{R}^\top
    \end{aligned}
    $$

=== "Atualizacao"

    $$
    \begin{aligned}
    v_t &= y_t - \mathbf{Z}\, \boldsymbol{\alpha}_{t|t-1} \\
    F_t &= \mathbf{Z}\, \mathbf{P}_{t|t-1}\, \mathbf{Z}^\top \\
    K_t &= \mathbf{P}_{t|t-1}\, \mathbf{Z}^\top F_t^{-1} \\
    \boldsymbol{\alpha}_{t|t} &= \boldsymbol{\alpha}_{t|t-1} + K_t\, v_t \\
    \mathbf{P}_{t|t} &= (\mathbf{I} - K_t\, \mathbf{Z})\, \mathbf{P}_{t|t-1}
    \end{aligned}
    $$

### 7.3 Inicializacao Difusa

Para modelos ARIMA com $d > 0$, os estados iniciais sao nao-estacionarios.
A inicializacao exata difusa (de Jong, 1991; Koopman, 1997) decompoe a
covariancia inicial:

$$
\mathbf{P}_{1|0} = \mathbf{P}_* + \kappa\, \mathbf{P}_\infty
$$

onde $\mathbf{P}_*$ corresponde a componente estacionaria, $\mathbf{P}_\infty$ a componente
difusa, e $\kappa \to \infty$. O filtro de Kalman difuso trata $\mathbf{P}_\infty$
analiticamente, evitando instabilidade numerica.

---

## 8. Propriedades Assintoticas dos Estimadores

### 8.1 Consistencia

Sob condicoes de regularidade (processo invertivel e estacionario, parametro
verdadeiro no interior do espaco parametrico), o estimador de maxima verossimilhanca
$\hat{\boldsymbol{\theta}}_T$ e **consistente**:

$$
\hat{\boldsymbol{\theta}}_T \xrightarrow{p} \boldsymbol{\theta}_0 \quad \text{quando } T \to \infty
$$

### 8.2 Normalidade Assintotica

O estimador MLE e **assintoticamente normal**:

$$
\sqrt{T}(\hat{\boldsymbol{\theta}}_T - \boldsymbol{\theta}_0) \xrightarrow{d} N(\mathbf{0}, \mathbf{V})
$$

onde $\mathbf{V} = \mathcal{I}(\boldsymbol{\theta}_0)^{-1}$ e o inverso da **matriz de informacao de Fisher**:

$$
\mathcal{I}(\boldsymbol{\theta}) = -\E\left[\frac{\partial^2 \ln L(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}\, \partial \boldsymbol{\theta}^\top}\right]
$$

### 8.3 Eficiencia

O estimador MLE atinge o **limite inferior de Cramer-Rao**, sendo assintoticamente
eficiente entre todos os estimadores regulares.

!!! warning "Condicoes de Regularidade"
    A normalidade assintotica requer que o modelo verdadeiro seja estacionario
    e invertivel (raizes estritamente fora do circulo unitario). Quando raizes
    estao proximo da fronteira (quasi-nao-estacionariedade), a convergencia
    pode ser lenta e as aproximacoes assintoticas imprecisas em amostras finitas.

---

## 9. Selecao de Modelo

### 9.1 Criterios de Informacao

Os criterios de informacao equilibram o ajuste (verossimilhanca) com a
complexidade (numero de parametros $k$):

$$
\text{AIC} = -2 \ln L + 2k
$$

$$
\text{AICc} = \text{AIC} + \frac{2k(k+1)}{T - k - 1}
$$

$$
\text{BIC} = -2 \ln L + k \ln T
$$

$$
\text{HQIC} = -2 \ln L + 2k \ln(\ln T)
$$

| Criterio | Penalidade | Propriedade | Uso recomendado |
|---|---|---|---|
| AIC | $2k$ | Eficiente (minimiza erro de previsao) | Previsao |
| AICc | AIC + correcao amostral | AIC corrigido para $T$ pequeno | $T/k < 40$ |
| BIC | $k \ln T$ | Consistente (seleciona ordem verdadeira) | Identificacao |
| HQIC | $2k \ln(\ln T)$ | Consistente, menos conservador que BIC | Compromisso |

### 9.2 Metodologia Box-Jenkins

A abordagem classica de Box & Jenkins (1970):

1. **Identificacao**: analise de ACF/PACF da serie (diferenciada se necessario)
    para propor ordens candidatas
2. **Estimacao**: ajuste dos modelos candidatos via MLE
3. **Diagnostico**: verificar se os residuos sao ruido branco
    (teste Ljung-Box, normalidade, homocedasticidade)
4. **Previsao**: se o modelo passa nos diagnosticos, usar para gerar previsoes

### 9.3 Selecao Automatica

O ChronoBox implementa busca automatica sobre uma grade de ordens $(p,d,q)$,
avaliando cada combinacao por AICc (seguindo Hyndman & Khandakar, 2008).

---

## 10. Previsao

### 10.1 Previsao Pontual

A previsao otima (no sentido de minimizar o erro quadratico medio) $h$ passos
a frente e a esperanca condicional:

$$
\hat{y}_{T+h|T} = \E[y_{T+h} | y_1, \ldots, y_T]
$$

### 10.2 Intervalos de Previsao

O erro de previsao $h$ passos a frente tem variancia:

$$
\Var(e_{T+h|T}) = \sigma^2 \sum_{j=0}^{h-1} \psi_j^2
$$

onde $\psi_j$ sao os coeficientes da representacao MA($\infty$). Sob normalidade:

$$
\hat{y}_{T+h|T} \pm z_{\alpha/2}\, \sigma \sqrt{\sum_{j=0}^{h-1} \psi_j^2}
$$

!!! note "Crescimento dos Intervalos"
    Para modelos ARIMA com $d \geq 1$, a variancia do erro de previsao cresce
    sem limite conforme $h \to \infty$, refletindo a incerteza crescente sobre
    o nivel futuro da serie.

---

## Referencias

- Box, G. E. P. & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Brockwell, P. J. & Davis, R. A. (1991). *Time Series: Theory and Methods*. 2nd ed., Springer.
- de Jong, P. (1991). "The Diffuse Kalman Filter." *Annals of Statistics*, 19(2), 1073-1083.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
- Hyndman, R. J. & Khandakar, Y. (2008). "Automatic Time Series Forecasting: The forecast Package for R." *Journal of Statistical Software*, 27(3).
- Koopman, S. J. (1997). "Exact Initial Kalman Filtering and Smoothing for Nonstationary Time Series Models." *Journal of the American Statistical Association*, 92(440), 1630-1638.
- Wold, H. (1938). *A Study in the Analysis of Stationary Time Series*. Almqvist & Wiksell.

---

## Veja Tambem

- [ETS & State-Space](ets-theory.md) — Modelos de suavizacao exponencial em espaco de estados
- [VAR](var-theory.md) — Extensao multivariada do AR
- [Filtros](filters-theory.md) — Filtros de Hodrick-Prescott, Baxter-King e Christiano-Fitzgerald
