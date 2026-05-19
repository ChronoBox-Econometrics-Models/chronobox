---
title: "ETS: Fundamentos Teoricos"
description: "Teoria completa de modelos ETS — framework state-space para exponential smoothing, taxonomia de erros, likelihood via inovacoes e relacao ETS-ARIMA."
---

# ETS: Fundamentos Teoricos

!!! abstract "Resumo"
    Esta pagina apresenta a teoria matematica dos modelos ETS (Error, Trend,
    Seasonality), que fornecem um framework state-space unificado para todos os
    metodos de suavizacao exponencial. O tratamento cobre a taxonomia completa,
    estimacao por maxima verossimilhanca, condicoes de estabilidade e as
    correspondencias formais entre ETS e ARIMA.

---

## 1. Suavizacao Exponencial: Motivacao

Os metodos de suavizacao exponencial, introduzidos por Brown (1959) e Holt (1957),
foram originalmente formulados como **procedimentos pontuais de previsao** sem um
modelo probabilistico subjacente. A contribuicao fundamental de Ord, Koehler &
Snyder (1997) e Hyndman et al. (2002, 2008) foi fornecer modelos estatisticos
(state-space) que geram esses metodos como previsoes otimas.

---

## 2. Framework State-Space

### 2.1 Estrutura Geral

Cada modelo ETS e definido por duas equacoes:

$$
\begin{aligned}
y_t &= w(\boldsymbol{x}_{t-1}) + r(\boldsymbol{x}_{t-1})\, \eps_t & \text{(equacao de observacao)} \\
\boldsymbol{x}_t &= f(\boldsymbol{x}_{t-1}) + g(\boldsymbol{x}_{t-1})\, \eps_t & \text{(equacao de transicao)}
\end{aligned}
$$

onde:

- $y_t$ e a observacao no tempo $t$
- $\boldsymbol{x}_t = (\ell_t, b_t, s_t, s_{t-1}, \ldots, s_{t-m+1})^\top$ e o **vetor de estado** contendo nivel ($\ell_t$), tendencia ($b_t$) e sazonalidade ($s_t$)
- $\eps_t \sim NID(0, \sigma^2)$ e o **erro de inovacao** (i.i.d.)
- $w, r, f, g$ sao funcoes que definem a estrutura do modelo

### 2.2 Taxonomia ETS

A notacao ETS(E, T, S) classifica os modelos em tres dimensoes:

| Componente | Opcoes | Significado |
|---|---|---|
| **E**rro | A (aditivo), M (multiplicativo) | Tipo de perturbacao estocastica |
| **T**endencia | N (nenhuma), A (aditiva), A$_d$ (amortecida) | Dinamica do nivel |
| **S**azonalidade | N (nenhuma), A (aditiva), M (multiplicativa) | Padrao sazonal |

Isso gera $2 \times 5 \times 3 = 30$ combinacoes potenciais, das quais **15** sao
mais comumente usadas (considerando tendencia multiplicativa e multiplicativa amortecida
como casos adicionais).

---

## 3. Modelos com Erro Aditivo

### 3.1 ETS(A,N,N) — Suavizacao Exponencial Simples

$$
\begin{aligned}
y_t &= \ell_{t-1} + \eps_t \\
\ell_t &= \ell_{t-1} + \alpha\, \eps_t
\end{aligned}
$$

onde $\alpha \in (0,1)$ e o parametro de suavizacao. A previsao um passo a frente e
$\hat{y}_{t|t-1} = \ell_{t-1}$.

### 3.2 ETS(A,A,N) — Metodo de Holt (Tendencia Aditiva)

$$
\begin{aligned}
y_t &= \ell_{t-1} + b_{t-1} + \eps_t \\
\ell_t &= \ell_{t-1} + b_{t-1} + \alpha\, \eps_t \\
b_t &= b_{t-1} + \beta\, \eps_t
\end{aligned}
$$

onde $\beta \in (0,1)$ controla a suavizacao da tendencia.

### 3.3 ETS(A,A$_d$,N) — Tendencia Amortecida (Gardner & McKenzie, 1985)

$$
\begin{aligned}
y_t &= \ell_{t-1} + \phi\, b_{t-1} + \eps_t \\
\ell_t &= \ell_{t-1} + \phi\, b_{t-1} + \alpha\, \eps_t \\
b_t &= \phi\, b_{t-1} + \beta\, \eps_t
\end{aligned}
$$

onde $\phi \in (0,1)$ e o parametro de amortecimento. Quando $\phi < 1$, a tendencia
converge para $b_\infty = 0$, produzindo previsoes de longo prazo que se estabilizam
no nivel $\ell_T + \phi b_T / (1 - \phi)$.

### 3.4 ETS(A,A,A) — Holt-Winters Aditivo

$$
\begin{aligned}
y_t &= \ell_{t-1} + b_{t-1} + s_{t-m} + \eps_t \\
\ell_t &= \ell_{t-1} + b_{t-1} + \alpha\, \eps_t \\
b_t &= b_{t-1} + \beta\, \eps_t \\
s_t &= s_{t-m} + \gamma\, \eps_t
\end{aligned}
$$

onde $m$ e o periodo sazonal e $\gamma \in (0,1)$ controla a suavizacao sazonal.

---

## 4. Modelos com Erro Multiplicativo

### 4.1 Definicao do Erro

Nos modelos multiplicativos, o erro e proporcional ao nivel da serie:

$$
\eps_t = \frac{y_t - \hat{y}_{t|t-1}}{\hat{y}_{t|t-1}}
$$

de modo que $y_t = \hat{y}_{t|t-1}(1 + \eps_t)$.

### 4.2 ETS(M,N,N) — SES Multiplicativo

$$
\begin{aligned}
y_t &= \ell_{t-1}(1 + \eps_t) \\
\ell_t &= \ell_{t-1}(1 + \alpha\, \eps_t)
\end{aligned}
$$

### 4.3 ETS(M,A$_d$,M) — Holt-Winters Multiplicativo Amortecido

$$
\begin{aligned}
y_t &= (\ell_{t-1} + \phi\, b_{t-1})\, s_{t-m}\, (1 + \eps_t) \\
\ell_t &= (\ell_{t-1} + \phi\, b_{t-1})(1 + \alpha\, \eps_t) \\
b_t &= \phi\, b_{t-1} + \beta\, (\ell_{t-1} + \phi\, b_{t-1})\, \eps_t \\
s_t &= s_{t-m}(1 + \gamma\, \eps_t)
\end{aligned}
$$

!!! warning "Restricao de Positividade"
    Modelos com erro multiplicativo requerem $y_t > 0$ para todo $t$. Alem disso,
    os estados $\ell_t$ e $s_t$ devem permanecer positivos, o que impoe restricoes
    implicitas sobre a magnitude dos erros.

---

## 5. Forma Matricial Unificada

### 5.1 Modelos Aditivos

Todos os modelos com erro aditivo podem ser escritos na forma linear:

$$
\begin{aligned}
y_t &= \mathbf{w}^\top \boldsymbol{x}_{t-1} + \eps_t \\
\boldsymbol{x}_t &= \mathbf{F}\, \boldsymbol{x}_{t-1} + \mathbf{g}\, \eps_t
\end{aligned}
$$

**Exemplo — ETS(A,A,A):**

$$
\mathbf{w} = \begin{pmatrix} 1 \\ 1 \\ 0 \\ \vdots \\ 0 \\ 1 \end{pmatrix}, \quad
\mathbf{F} = \begin{pmatrix}
1 & 1 & 0 & \cdots & 0 & 0 \\
0 & 1 & 0 & \cdots & 0 & 0 \\
0 & 0 & 0 & \cdots & 0 & 1 \\
0 & 0 & 1 & \cdots & 0 & 0 \\
\vdots & & & \ddots & & \vdots \\
0 & 0 & 0 & \cdots & 1 & 0
\end{pmatrix}, \quad
\mathbf{g} = \begin{pmatrix} \alpha \\ \beta \\ \gamma \\ 0 \\ \vdots \\ 0 \end{pmatrix}
$$

### 5.2 Modelos Multiplicativos

Para modelos com erro multiplicativo, a forma e nao-linear:

$$
\begin{aligned}
y_t &= w(\boldsymbol{x}_{t-1})(1 + \eps_t) \\
\boldsymbol{x}_t &= f(\boldsymbol{x}_{t-1}) + g(\boldsymbol{x}_{t-1})\, \eps_t
\end{aligned}
$$

onde $w(\cdot)$, $f(\cdot)$ e $g(\cdot)$ sao funcoes nao-lineares especificas de cada modelo.

---

## 6. Estimacao por Maxima Verossimilhanca

### 6.1 Likelihood via Inovacoes

Para modelos com **erro aditivo**, as inovacoes $\eps_t$ tem variancia constante $\sigma^2$:

$$
\ln L(\boldsymbol{\theta}, \boldsymbol{x}_0) = -\frac{T}{2} \ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{t=1}^{T} \eps_t^2
$$

onde $\eps_t = y_t - \mathbf{w}^\top \boldsymbol{x}_{t-1}$.

O estimador concentrado de $\sigma^2$ e:

$$
\hat{\sigma}^2 = \frac{1}{T} \sum_{t=1}^{T} \hat{\eps}_t^2
$$

Substituindo na log-verossimilhanca:

$$
\ln L^*(\boldsymbol{\theta}, \boldsymbol{x}_0) = -\frac{T}{2} \ln\left(\frac{2\pi}{T} \sum_{t=1}^{T} \hat{\eps}_t^2\right) - \frac{T}{2}
$$

### 6.2 Likelihood Multiplicativa

Para modelos com **erro multiplicativo**, a transformacao jacobiana introduz um
termo adicional. Se $\eps_t = (y_t - \mu_t)/\mu_t$ onde $\mu_t = w(\boldsymbol{x}_{t-1})$:

$$
\ln L(\boldsymbol{\theta}, \boldsymbol{x}_0) = -\frac{T}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{t=1}^{T}\eps_t^2 - \sum_{t=1}^{T}\ln|\mu_t|
$$

O termo $-\sum \ln|\mu_t|$ penaliza modelos com niveis altos, distinguindo
modelos aditivos de multiplicativos na selecao automatica.

### 6.3 Otimizacao

A maximizacao e realizada numericamente, tipicamente via L-BFGS-B ou
Nelder-Mead, com restricoes de caixa nos parametros de suavizacao.

!!! tip "Estados Iniciais"
    Os estados iniciais $\boldsymbol{x}_0 = (\ell_0, b_0, s_{1-m}, \ldots, s_0)$
    sao estimados conjuntamente com os parametros de suavizacao. Uma heuristica
    comum e inicializar $\ell_0$ e $b_0$ via regressao linear nos primeiros periodos
    e os componentes sazonais via medias sazonais.

---

## 7. Estabilidade e Admissibilidade

### 7.1 Condicoes de Estabilidade

Um modelo ETS aditivo e **estavel** se todos os eigenvalues de $\mathbf{F} - \mathbf{g}\,\mathbf{w}^\top$
estao dentro do circulo unitario:

$$
|\lambda_i(\mathbf{F} - \mathbf{g}\,\mathbf{w}^\top)| < 1, \quad \forall\, i
$$

!!! note "Regiao Admissivel"
    As restricoes tradicionais $0 < \alpha < 1$, $0 < \beta < \alpha$,
    $0 < \gamma < 1 - \alpha$ sao **suficientes** mas nao necessarias para estabilidade.
    A regiao admissivel completa e definida pela condicao de eigenvalues acima e pode
    incluir valores negativos dos parametros de suavizacao (Hyndman et al., 2008, Cap. 10).

### 7.2 Condicao de Forcastability

Hyndman et al. (2008) introduzem o conceito de **forecastability**: um modelo
e forecastable se as previsoes convergem para uma funcao deterministica conforme o
horizonte cresce. Para modelos aditivos, forecastability equivale a estabilidade.

---

## 8. Previsao

### 8.1 Previsao Pontual

Para modelos aditivos, a previsao $h$ passos a frente:

$$
\hat{y}_{T+h|T} = \mathbf{w}_h^\top \boldsymbol{x}_T
$$

onde $\mathbf{w}_h$ depende da estrutura do modelo e do horizonte.

Para o ETS(A,A,N):

$$
\hat{y}_{T+h|T} = \ell_T + h\, b_T
$$

Para o ETS(A,A$_d$,N):

$$
\hat{y}_{T+h|T} = \ell_T + (\phi + \phi^2 + \cdots + \phi^h)\, b_T = \ell_T + \phi\frac{1 - \phi^h}{1 - \phi}\, b_T
$$

### 8.2 Intervalos de Previsao

Para modelos com **erro aditivo**, a variancia do erro de previsao tem forma analitica:

$$
\Var(e_{T+h|T}) = \sigma^2 \left[1 + \sum_{j=1}^{h-1} \mathbf{w}^\top (\mathbf{F} - \mathbf{g}\,\mathbf{w}^\top)^{j-1}\, \mathbf{g}\, \mathbf{g}^\top \left((\mathbf{F} - \mathbf{g}\,\mathbf{w}^\top)^\top\right)^{j-1}\, \mathbf{w}\right]
$$

Para modelos com **erro multiplicativo**, intervalos analiticos nao estao disponíveis
e sao obtidos via simulacao (bootstrap das inovacoes).

---

## 9. Relacao ETS-ARIMA

### 9.1 Correspondencias Exatas

Alguns modelos ETS com erro aditivo geram os mesmos processos estocasticos que
modelos ARIMA especificos:

| Modelo ETS | Modelo ARIMA equivalente |
|---|---|
| ETS(A,N,N) | ARIMA(0,1,1) |
| ETS(A,A,N) | ARIMA(0,2,2) |
| ETS(A,A$_d$,N) | ARIMA(1,1,2) |
| ETS(A,N,A) com periodo $m$ | ARIMA(0,1,$m$)(0,1,0)$_m$ |
| ETS(A,A,A) com periodo $m$ | ARIMA(0,1,$m+1$)(0,1,0)$_m$ |

### 9.2 Diferenca Fundamental

!!! warning "Modelos Nao Equivalentes"
    Apesar das correspondencias nas previsoes pontuais, os modelos **nao sao
    identicos**:

    - Modelos ETS com erro **multiplicativo** nao tem correspondencia ARIMA
    - As verossimilhancas podem diferir (especialmente para modelos multiplicativos)
    - ETS fornece um framework natural para lidar com sazonalidade multiplicativa
    - ARIMA e mais flexivel para estruturas AR complexas

### 9.3 Demonstracao: ETS(A,N,N) $\equiv$ ARIMA(0,1,1)

Do ETS(A,N,N):

$$
\begin{aligned}
y_t &= \ell_{t-1} + \eps_t \\
\ell_t &= \ell_{t-1} + \alpha\, \eps_t
\end{aligned}
$$

Substituindo $\ell_{t-1} = \ell_{t-2} + \alpha\, \eps_{t-1}$ na equacao de observacao:

$$
y_t = y_{t-1} - \eps_{t-1} + \alpha\, \eps_{t-1} + \eps_t = y_{t-1} + \eps_t - (1-\alpha)\,\eps_{t-1}
$$

que e um ARIMA(0,1,1) com $\theta_1 = -(1-\alpha)$, i.e.:

$$
(1-B)\, y_t = (1 - (1-\alpha)B)\, \eps_t
$$

---

## 10. Selecao Automatica de Modelos

O algoritmo de selecao automatica (Hyndman et al., 2002) avalia todos os
modelos admissiveis da taxonomia ETS e seleciona aquele com menor AICc:

1. Estimar todos os 30 modelos candidatos (com restricoes de admissibilidade)
2. Calcular o AICc de cada modelo (com log-likelihood incluindo o jacobiano para modelos multiplicativos)
3. Selecionar o modelo com menor AICc

!!! info "Comparabilidade"
    Modelos com erro aditivo e multiplicativo **nao podem** ser comparados diretamente
    pelo valor da log-verossimilhanca sem o termo jacobiano. O AICc, computado com a
    verossimilhanca completa (incluindo $-\sum \ln|\mu_t|$ para modelos multiplicativos),
    permite comparacao direta entre todas as especificacoes.

---

## Referencias

- Brown, R. G. (1959). *Statistical Forecasting for Inventory Control*. McGraw-Hill.
- Gardner, E. S. & McKenzie, E. (1985). "Forecasting Trends in Time Series." *Management Science*, 31(10), 1237-1246.
- Holt, C. C. (1957). "Forecasting Seasonals and Trends by Exponentially Weighted Moving Averages." ONR Memorandum No. 52. (Republicado em *International Journal of Forecasting*, 2004, 20(1), 5-10.)
- Hyndman, R. J., Koehler, A. B., Snyder, R. D. & Grose, S. (2002). "A State Space Framework for Automatic Forecasting Using Exponential Smoothing Methods." *International Journal of Forecasting*, 18(3), 439-454.
- Hyndman, R. J., Koehler, A. B., Ord, J. K. & Snyder, R. D. (2008). *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
- Ord, J. K., Koehler, A. B. & Snyder, R. D. (1997). "Estimation and Prediction for a Class of Dynamic Nonlinear Statistical Models." *Journal of the American Statistical Association*, 92(440), 1621-1629.

---

## Veja Tambem

- [ARIMA](arima-theory.md) — Modelos ARIMA e representacao estado-espaco
- [VAR](var-theory.md) — Extensao multivariada para multiplas series
- [Filtros](filters-theory.md) — Decomposicao de tendencia e ciclo
