---
title: VAR & VECM
description: Modelos Vetoriais Autoregressivos --- VAR, VECM, IRF, FEVD, Granger e previsao multivariada.
---

# VAR & VECM

Modelos multivariados permitem analisar **multiplas series temporais conjuntamente**,
capturando interdependencias dinamicas que modelos univariados ignoram. O VAR
(Vector Autoregression) e o VECM (Vector Error Correction Model) sao os pilares
da macroeconometria moderna.

---

## Por que modelar multiplas series conjuntamente?

Variaveis macroeconomicas --- PIB, inflacao, taxa de juros, cambio --- nao se
movem de forma independente. Modelos univariados tratam cada serie isoladamente e
perdem informacao sobre:

- **Feedback entre variaveis**: PIB afeta juros e juros afetam PIB
- **Transmissao de choques**: como um choque monetario se propaga pela economia
- **Relacoes de longo prazo**: equiliibrio entre series cointegradas

Modelos VAR capturam essas interacoes de forma **ateoretica** --- sem impor
restricoes estruturais a priori --- usando apenas a dinamica dos dados.

---

## VAR vs VECM: quando usar cada um

| Cenario | Modelo | Razao |
|---|---|---|
| Series estacionarias (ou diferenciadas) | [VAR](var.md) | Modelo padrao para series I(0) |
| Series I(1) sem cointegracao | [VAR em diferencas](var.md) | Diferenciar para estacionarizar |
| Series I(1) com cointegracao | [VECM](vecm.md) | Preserva relacao de longo prazo |
| Analise de choques | [IRF](irf.md) + [FEVD](fevd.md) | Impulso-resposta e decomposicao |
| Direcao de causalidade | [Granger](granger.md) | Teste de precedencia temporal |

!!! warning "Series I(1) cointegradas"
    Se as series sao integradas de ordem 1 e cointegradas, estimar um VAR em
    diferencas **descarta a relacao de longo prazo**. Use o VECM, que modela
    tanto a dinamica de curto prazo quanto o equilibrio de longo prazo.

---

## Modelos da Secao

<div class="grid cards" markdown>

-   :material-vector-polyline:{ .lg .middle } **VAR(p)**

    ---

    Modelo vetorial autoregressivo: multiplas series com $p$ defasagens.

    [:octicons-arrow-right-24: VAR](var.md)

-   :material-link-variant:{ .lg .middle } **VECM**

    ---

    Modelo de correcao de erros vetorial para series cointegradas.

    [:octicons-arrow-right-24: VECM](vecm.md)

-   :material-pulse:{ .lg .middle } **Impulse Response (IRF)**

    ---

    Funcoes impulso-resposta: como choques se propagam no sistema.

    [:octicons-arrow-right-24: IRF](irf.md)

-   :material-chart-bar-stacked:{ .lg .middle } **FEVD**

    ---

    Decomposicao da variancia do erro de previsao por fonte de choque.

    [:octicons-arrow-right-24: FEVD](fevd.md)

-   :material-arrow-decision:{ .lg .middle } **Granger Causality**

    ---

    Testes de causalidade de Granger: bivariado e em bloco.

    [:octicons-arrow-right-24: Granger](granger.md)

-   :material-chart-timeline-variant:{ .lg .middle } **Previsao Multivariada**

    ---

    Previsao h-passos a frente com intervalos de confianca.

    [:octicons-arrow-right-24: Previsao](forecasting.md)

</div>

---

## Fluxo de Trabalho Tipico

```python
from chronobox import VAR
from chronobox.tests_stat import adf_test, johansen_test

import pandas as pd

# 1. Carregar dados multivariados
data = pd.DataFrame({
    "gdp": gdp_series,
    "inflation": inflation_series,
    "interest_rate": rate_series,
})

# 2. Testar estacionariedade de cada serie
for col in data.columns:
    result = adf_test(data[col])
    print(f"{col}: ADF p-value = {result.pvalue:.4f}")

# 3. Testar cointegracao (se series sao I(1))
joh = johansen_test(data, det_order=0, k_ar_diff=2)
print(joh.summary())

# 4a. Se nao cointegradas → VAR em diferencas
model = VAR(lags=2)
results = model.fit(data.diff().dropna())

# 4b. Se cointegradas → VECM
from chronobox import VECM
model = VECM(lags=2, coint_rank=1)
results = model.fit(data)

# 5. Analise de choques
irf = results.irf(steps=20)
fevd = results.fevd(steps=20)

# 6. Previsao
fc = results.forecast(steps=12)
```

---

## Referencias

- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Johansen, S. (1995). *Likelihood-Based Inference in Cointegrated Vector
  Autoregressive Models*. Oxford University Press.
