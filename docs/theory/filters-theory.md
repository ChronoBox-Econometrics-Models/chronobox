---
title: "Filtros Economicos: Fundamentos Teoricos"
description: "Teoria completa dos filtros de extracao de tendencia e ciclo — representacao espectral, HP, Baxter-King, Christiano-Fitzgerald, Hamilton e Beveridge-Nelson, com comparacao espectral."
---

# Filtros Economicos: Fundamentos Teoricos

!!! abstract "Resumo"
    Esta pagina apresenta a fundamentacao matematica dos principais filtros
    utilizados em macroeconomia para decomposicao tendencia-ciclo. Partindo
    da representacao espectral de series temporais, derivamos as propriedades
    dos filtros de Hodrick-Prescott, Baxter-King, Christiano-Fitzgerald,
    Hamilton e Beveridge-Nelson, com enfase em suas funcoes de transferencia
    e limitacoes praticas.

---

## 1. Representacao Espectral de Series Temporais

### 1.1 Densidade Espectral

Para um processo estacionario $\{y_t\}$ com funcao de autocovariancia
$\gamma(h)$ absolutamente somavel, a **densidade espectral de potencia** e:

$$
S_y(\omega) = \frac{1}{2\pi} \sum_{h=-\infty}^{\infty} \gamma(h)\, e^{-i\omega h}, \quad \omega \in [-\pi, \pi]
$$

A funcao de autocovariancia e recuperada via transformada inversa:

$$
\gamma(h) = \int_{-\pi}^{\pi} S_y(\omega)\, e^{i\omega h}\, d\omega
$$

Em particular, a variancia e $\gamma(0) = \int_{-\pi}^{\pi} S_y(\omega)\, d\omega$.

### 1.2 Interpretacao

A densidade espectral decompoe a variancia total do processo em contribuicoes
de diferentes frequencias $\omega$. Na analise de ciclos economicos, as faixas
relevantes sao:

| Componente | Periodo | Frequencia $\omega$ |
|---|---|---|
| Tendencia | $> 32$ trimestres ($> 8$ anos) | $\omega < 2\pi/32$ |
| Ciclo de negocios | $6$-$32$ trimestres ($1.5$-$8$ anos) | $2\pi/32 \leq \omega \leq 2\pi/6$ |
| Ruido / alta frequencia | $< 6$ trimestres | $\omega > 2\pi/6$ |

### 1.3 Funcao de Transferencia de um Filtro Linear

Um filtro linear $\{b_j\}_{j=-\infty}^{\infty}$ aplicado a $y_t$ produz:

$$
\tilde{y}_t = \sum_{j=-\infty}^{\infty} b_j\, y_{t-j} = b(B)\, y_t
$$

A **funcao de transferencia** (ou resposta em frequencia) e:

$$
\mathcal{B}(\omega) = \sum_{j=-\infty}^{\infty} b_j\, e^{-i\omega j}
$$

A densidade espectral da serie filtrada e:

$$
S_{\tilde{y}}(\omega) = |\mathcal{B}(\omega)|^2\, S_y(\omega)
$$

O **ganho espectral** $|\mathcal{B}(\omega)|^2$ determina quanto de cada frequencia
e preservado ou atenuado pelo filtro.

---

## 2. Filtro Ideal

### 2.1 Definicao

O **filtro band-pass ideal** extrai componentes com frequencias em $[\omega_L, \omega_H]$:

$$
\mathcal{B}^*(\omega) = \begin{cases}
1 & \text{se } \omega_L \leq |\omega| \leq \omega_H \\
0 & \text{caso contrario}
\end{cases}
$$

Os coeficientes do filtro ideal sao:

$$
b_j^* = \frac{1}{2\pi} \int_{-\pi}^{\pi} \mathcal{B}^*(\omega)\, e^{i\omega j}\, d\omega = \frac{\sin(\omega_H j) - \sin(\omega_L j)}{\pi j}, \quad j \neq 0
$$

$$
b_0^* = \frac{\omega_H - \omega_L}{\pi}
$$

### 2.2 Problema Pratico

O filtro ideal requer **infinitos** coeficientes ($j \to \pm\infty$), sendo
inaplicavel a amostras finitas. Os filtros discutidos a seguir sao
**aproximacoes** do filtro ideal, cada um com diferentes trade-offs entre
fidelidade espectral, vies nos endpoints e complexidade computacional.

---

## 3. Filtro de Hodrick-Prescott (HP)

### 3.1 Formulacao

O filtro HP (Hodrick & Prescott, 1997) decompoe $y_t = \tau_t + c_t$
(tendencia + ciclo) resolvendo:

$$
\min_{\{\tau_t\}_{t=1}^{T}} \left\{\sum_{t=1}^{T} (y_t - \tau_t)^2 + \lambda \sum_{t=3}^{T} (\Delta^2 \tau_t)^2 \right\}
$$

onde $\Delta^2 \tau_t = \tau_t - 2\tau_{t-1} + \tau_{t-2}$ e a segunda diferenca, e
$\lambda > 0$ e o parametro de suavizacao.

### 3.2 Solucao Matricial

Em forma matricial, a tendencia estimada e:

$$
\hat{\boldsymbol{\tau}} = (\mathbf{I}_T + \lambda \mathbf{K}^\top \mathbf{K})^{-1}\, \mathbf{y}
$$

onde $\mathbf{K}$ e a matriz $(T-2) \times T$ de segundas diferencas.

O componente ciclico e:

$$
\hat{\mathbf{c}} = \mathbf{y} - \hat{\boldsymbol{\tau}} = \lambda \mathbf{K}^\top \mathbf{K}\, (\mathbf{I}_T + \lambda \mathbf{K}^\top \mathbf{K})^{-1}\, \mathbf{y}
$$

### 3.3 Propriedades Espectrais

No dominio da frequencia, o ganho do filtro HP para a componente ciclica e:

$$
|\mathcal{B}_{\text{HP}}(\omega)|^2 = \frac{4\lambda(1 - \cos\omega)^2}{1 + 4\lambda(1 - \cos\omega)^2}
$$

!!! note "Escolha de $\lambda$"
    O parametro $\lambda$ controla o trade-off suavidade-ajuste:

    - $\lambda = 1\,600$: dados trimestrais (Hodrick & Prescott, 1997)
    - $\lambda = 129\,600$: dados mensais (Ravn & Uhlig, 2002)
    - $\lambda = 6.25$: dados anuais (Ravn & Uhlig, 2002)

    A regra de Ravn & Uhlig estabelece $\lambda = 1600 \cdot (s/4)^4$, onde $s$
    e o numero de observacoes por ano.

### 3.4 Vies nos Endpoints

!!! warning "Problema dos Endpoints"
    O filtro HP e um filtro **bilateral finito** que utiliza observacoes passadas
    e futuras. Nas extremidades da amostra, o filtro torna-se efetivamente
    unilateral, introduzindo:

    - **Vies de nivel**: a tendencia estimada e atraida em direcao as ultimas
      observacoes
    - **Revisoes**: novas observacoes alteram as estimativas dos periodos
      anteriores mais proximos ao final da amostra
    - **Distorcao espectral**: o ganho efetivo nos endpoints difere do ganho
      no interior da amostra

    Essas limitacoes sao particularmente relevantes para analise de politica
    em tempo real (Orphanides & van Norden, 2002).

---

## 4. Filtro de Baxter-King (BK)

### 4.1 Formulacao

O filtro BK (Baxter & King, 1999) e uma **aproximacao simetrica finita**
do filtro band-pass ideal. Trunca-se o filtro ideal em $K$ lags/leads e
aplica-se uma correcao para garantir que o ganho em $\omega = 0$ seja zero:

$$
\hat{b}_j = b_j^* - \frac{1}{2K+1} \sum_{k=-K}^{K} b_k^*, \quad j = -K, \ldots, K
$$

onde $b_j^*$ sao os coeficientes do filtro ideal.

### 4.2 Propriedades

- **Simetria**: $\hat{b}_j = \hat{b}_{-j}$, garantindo defasagem (phase shift) zero
- **Filtragem de tendencia**: $\sum_{j=-K}^{K} \hat{b}_j = 0$, removendo componentes deterministicos
- **Perda de observacoes**: $K$ observacoes sao perdidas em cada extremidade da amostra

### 4.3 Funcao de Transferencia

$$
\mathcal{B}_{\text{BK}}(\omega) = \sum_{j=-K}^{K} \hat{b}_j\, e^{-i\omega j} = \hat{b}_0 + 2 \sum_{j=1}^{K} \hat{b}_j \cos(\omega j)
$$

### 4.4 Escolha de $K$

| Frequencia dos dados | $K$ recomendado | Observacoes perdidas |
|---|---|---|
| Trimestral | $K = 12$ | 24 observacoes (6 anos) |
| Mensal | $K = 36$ | 72 observacoes (6 anos) |
| Anual | $K = 3$ | 6 observacoes |

!!! warning "Trade-off"
    Valores maiores de $K$ melhoram a aproximacao do filtro ideal,
    mas custam mais observacoes nas extremidades. Para amostras curtas,
    a perda pode ser inaceitavel.

---

## 5. Filtro de Christiano-Fitzgerald (CF)

### 5.1 Formulacao

O filtro CF (Christiano & Fitzgerald, 2003) e uma aproximacao **assimetrica**
otima do filtro band-pass ideal, que utiliza toda a amostra disponivel
(sem perda de observacoes).

Para $t = 1, \ldots, T$, o filtro e:

$$
\tilde{y}_t = b_0 y_t + \sum_{j=1}^{t-1} b_j y_{t-j} + \sum_{j=1}^{T-t} b_j y_{t+j} + \tilde{b}_{T-t} y_T + \tilde{b}_{t-1} y_1
$$

onde os coeficientes $b_j$ sao os do filtro ideal e $\tilde{b}_j$ sao termos
de correcao que garantem a propriedade de filtragem de tendencia.

### 5.2 Derivacao Otima

Christiano & Fitzgerald derivam os pesos assimetricos minimizando o erro
quadratico medio entre o filtro aplicado e o filtro ideal, sob a hipotese
de que a serie segue um passeio aleatorio:

$$
\min_{\{b_j^t\}} \; \E\left[\left(\tilde{y}_t^{\text{ideal}} - \sum_j b_j^t\, y_{t-j}\right)^2\right]
$$

### 5.3 Propriedades

- **Sem perda de observacoes**: utiliza pesos assimetricos nos endpoints
- **Filtro otimo**: minimiza MSE sob hipotese de random walk
- **Defasagem nao-zero nos endpoints**: como consequencia da assimetria,
  pode haver phase shift nas extremidades

### 5.4 Variantes

=== "Full-Sample (padrao)"

    Utiliza toda a amostra para construir os pesos, resultando em um
    filtro two-sided com pesos variando por $t$.

=== "Fixed-Length"

    Trunca os pesos em um numero fixo de leads/lags, aproximando-se
    do Baxter-King mas com correcao otima nos endpoints.

=== "Drift-Adjusted"

    Permite uma tendencia linear na serie geradora, alterando os
    pesos de correcao.

---

## 6. Filtro de Hamilton (2018)

### 6.1 Motivacao

Hamilton (2018) critica o filtro HP por introduzir **dinamicas espurias**
(ciclos artificiais) e propoe uma alternativa baseada em regressao:

!!! warning "Criticas ao HP"
    Hamilton (2018) demonstra que o filtro HP:

    1. Produz relacoes dinamicas espurias entre variaveis filtradas
    2. Gera correlacoes seriais que nao existem nos dados
    3. Depende criticamente de observacoes futuras (vies de endpoint)
    4. Nao tem fundamentacao estatistica formal como estimador

### 6.2 Formulacao

O filtro de Hamilton consiste em estimar a regressao:

$$
y_{t+h} = \beta_0 + \beta_1 y_t + \beta_2 y_{t-1} + \beta_3 y_{t-2} + \beta_4 y_{t-3} + v_{t+h}
$$

O **componente ciclico** e definido como o residuo:

$$
\hat{c}_t = y_{t+h} - \hat{y}_{t+h|t}
$$

e a **tendencia** como o valor ajustado:

$$
\hat{\tau}_{t+h} = \hat{\beta}_0 + \hat{\beta}_1 y_t + \hat{\beta}_2 y_{t-1} + \hat{\beta}_3 y_{t-2} + \hat{\beta}_4 y_{t-3}
$$

### 6.3 Escolha de $h$

Hamilton recomenda $h = 8$ para dados trimestrais ($2$ anos a frente),
justificando que:

- Corresponde a definicao padrao de ciclo de negocios (NBER)
- Evita a correlacao serial espuria do HP
- Produz um residuo interpretavel como desvio da previsao de medio prazo

### 6.4 Propriedades

| Propriedade | HP | Hamilton |
|---|---|---|
| Tipo | Filtro bilateral | Regressao unilateral |
| Vies de endpoint | Severo | Nenhum (apos perda de $h$ obs.) |
| Ciclos espurios | Sim | Nao |
| Fundamentacao estatistica | Ad hoc | Regressao OLS padrao |
| Observacoes perdidas | Nenhuma (mas vies) | $h + p$ iniciais |
| Flexibilidade | Parametro $\lambda$ | Parametros $h$ e $p$ |

---

## 7. Decomposicao de Beveridge-Nelson (BN)

### 7.1 Formulacao

A decomposicao BN (Beveridge & Nelson, 1981) decompoe uma serie $I(1)$
em um componente **permanente** (tendencia estocastica) e um componente
**transitorio** (ciclo):

$$
y_t = \tau_t^{BN} + c_t^{BN}
$$

### 7.2 Derivacao

Seja $\Delta y_t = \mu + C(B)\varepsilon_t$ a representacao de Wold da serie diferenciada,
onde $C(B) = \sum_{j=0}^{\infty} c_j B^j$ com $c_0 = 1$. A decomposicao BN e:

$$
\tau_t^{BN} = \tau_{t-1}^{BN} + \mu + C(1)\varepsilon_t
$$

$$
c_t^{BN} = -C^*(B)\varepsilon_t
$$

onde $C(1) = \sum_{j=0}^{\infty} c_j$ e o **ganho de longo prazo** e
$C^*(B) = \sum_{j=0}^{\infty} c_j^* B^j$ com $c_j^* = -\sum_{k=j+1}^{\infty} c_k$.

### 7.3 Propriedades

- A tendencia BN e um **passeio aleatorio com drift**: $\tau_t^{BN} = \mu + \tau_{t-1}^{BN} + C(1)\varepsilon_t$
- O ciclo BN e um processo **estacionario**: $c_t^{BN} \sim I(0)$
- A inovacao e **compartilhada**: tendencia e ciclo sao perfeitamente correlacionados contemporaneamente
- A tendencia captura o **componente permanente** dos choques

!!! note "Relacao com ARIMA"
    Para um ARIMA(p,1,q), os coeficientes $c_j$ sao obtidos a partir dos
    polinomios AR e MA. O ganho de longo prazo $C(1) = \theta(1)/\phi(1)$
    e computado diretamente dos parametros estimados.

### 7.4 Extensao Multivariada

A decomposicao BN estende-se naturalmente a sistemas VAR/VECM via a
representacao de Granger (Secao 10 da pagina [VAR](var-theory.md)):

$$
\boldsymbol{\tau}_t^{BN} = \boldsymbol{\tau}_{t-1}^{BN} + \boldsymbol{\mu} + \mathbf{C}(1)\, \mathbf{u}_t
$$

---

## 8. Comparacao Espectral dos Filtros

### 8.1 Funcoes de Ganho

A tabela abaixo resume o ganho espectral $|\mathcal{B}(\omega)|^2$
dos principais filtros na faixa de ciclos de negocios ($\omega \in [2\pi/32, 2\pi/6]$
para dados trimestrais):

| Filtro | Ganho em $\omega = 0$ | Passagem de banda | Leakage | Phase shift |
|---|---|---|---|---|
| Ideal | $0$ | Perfeita | Nenhum | Zero |
| HP ($\lambda=1600$) | $0$ | Aproximada (vaza baixas freq.) | Moderado | Zero (interior) |
| BK ($K=12$) | $0$ | Boa | Baixo | Zero |
| CF | $0$ | Boa | Baixo | Proximo de zero |
| Hamilton | N/A | Implicita | N/A | N/A (regressao) |

### 8.2 Criterios de Escolha

!!! tip "Recomendacoes Praticas"

    1. **Analise em tempo real**: Hamilton (sem vies de endpoint)
    2. **Amostra longa, analise retrospectiva**: BK ou CF
    3. **Comparacao com literatura existente**: HP (mais usado, facilita comparabilidade)
    4. **Decomposicao permanente-transitoria**: Beveridge-Nelson
    5. **Amostra curta**: CF (sem perda de observacoes) ou HP

---

## Referencias

- Baxter, M. & King, R. G. (1999). "Measuring Business Cycles: Approximate Band-Pass Filters for Economic Time Series." *Review of Economics and Statistics*, 81(4), 575-593.
- Beveridge, S. & Nelson, C. R. (1981). "A New Approach to Decomposition of Economic Time Series into Permanent and Transitory Components with Particular Attention to Measurement of the 'Business Cycle'." *Journal of Monetary Economics*, 7(2), 151-174.
- Christiano, L. J. & Fitzgerald, T. J. (2003). "The Band Pass Filter." *International Economic Review*, 44(2), 435-465.
- Hamilton, J. D. (2018). "Why You Should Never Use the Hodrick-Prescott Filter." *Review of Economics and Statistics*, 100(5), 831-843.
- Hodrick, R. J. & Prescott, E. C. (1997). "Postwar U.S. Business Cycles: An Empirical Investigation." *Journal of Money, Credit and Banking*, 29(1), 1-16.
- Orphanides, A. & van Norden, S. (2002). "The Unreliability of Output-Gap Estimates in Real Time." *Review of Economics and Statistics*, 84(4), 569-583.
- Ravn, M. O. & Uhlig, H. (2002). "On Adjusting the Hodrick-Prescott Filter for the Frequency of Observations." *Review of Economics and Statistics*, 84(2), 371-376.

---

## Veja Tambem

- [ARIMA](arima-theory.md) — Representacao ARIMA e decomposicao de Wold
- [VAR](var-theory.md) — Modelos vetoriais autorregressivos
- [ETS & State-Space](ets-theory.md) — Decomposicao via espaco de estados
