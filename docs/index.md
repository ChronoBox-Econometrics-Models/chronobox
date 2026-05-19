---
title: ChronoBox - Econometria de Series Temporais em Python
description: Biblioteca Python completa para econometria de series temporais com ARIMA, ETS, VAR, SVAR, testes estatisticos, filtros economicos e muito mais
hide:
  - navigation
  - toc
---

# ChronoBox

**Econometria de Series Temporais em Python**

[![PyPI](https://img.shields.io/pypi/v/chronobox)](https://pypi.org/project/chronobox/)
[![Python](https://img.shields.io/pypi/pyversions/chronobox)](https://pypi.org/project/chronobox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

ChronoBox e uma biblioteca Python completa para econometria de series temporais.
Da modelagem univariada classica (ARIMA, ETS) a sistemas multivariados avancados
(SVAR, BVAR, FAVAR, TVP-VAR, GVAR), a biblioteca oferece uma API consistente para
estimacao, diagnostico, previsao e visualizacao --- tudo em um unico pacote.

A biblioteca cobre todo o fluxo de trabalho empirico: selecao automatica de modelos,
30 especificacoes de suavizacao exponencial, testes de raiz unitaria e cointegracao,
filtros para extracao de ciclos economicos, analise de spillover de Diebold-Yilmaz
e geracao de reports publicaveis em HTML, LaTeX e Markdown.

ChronoBox e construida sobre o [KalmanBox](https://github.com/NodesEcon/kalmanbox),
que fornece a infraestrutura de state-space e estimacao por maxima verossimilhanca (MLE).
Essa separacao permite que cada camada evolua de forma independente, mantendo
implementacoes numericamente robustas e testadas.

---

## Quick Start

```python
from chronobox import ARIMA
from chronobox.datasets import load_dataset

data = load_dataset("airline")
model = ARIMA(order=(0,1,1), seasonal_order=(0,1,1,12))
result = model.fit(data["passengers"])
forecast = result.forecast(steps=12)
result.plot_forecast(forecast)
```

---

## O que a ChronoBox Oferece

<div class="grid cards" markdown>

-   :material-chart-line: **Modelos Univariados**

    ---

    ARIMA, SARIMA, ARFIMA, Auto-ARIMA com selecao automatica de ordem

    [:octicons-arrow-right-24: User Guide](user-guide/arima/index.md)

-   :material-sine-wave: **Suavizacao Exponencial**

    ---

    30 modelos ETS, Theta Method, Auto-ETS com selecao por AICc

    [:octicons-arrow-right-24: User Guide](user-guide/ets/index.md)

-   :material-chart-timeline-variant: **Modelos VAR**

    ---

    VAR, VECM, Impulse Response (IRF), FEVD, Granger Causality

    [:octicons-arrow-right-24: User Guide](user-guide/var/index.md)

-   :material-vector-triangle: **Modelos Estruturais**

    ---

    SVAR, BVAR, FAVAR, TVP-VAR, GVAR, Historical Decomposition

    [:octicons-arrow-right-24: User Guide](user-guide/svar/index.md)

-   :material-test-tube: **Testes Estatisticos**

    ---

    Raiz unitaria (ADF, PP, KPSS, Zivot-Andrews), cointegracao (Johansen, Engle-Granger), especificacao

    [:octicons-arrow-right-24: Diagnostics](diagnostics/index.md)

-   :material-filter-variant: **Filtros Economicos**

    ---

    Hodrick-Prescott, Baxter-King, Christiano-Fitzgerald, Hamilton, Beveridge-Nelson

    [:octicons-arrow-right-24: User Guide](user-guide/filters/index.md)

-   :material-link-variant: **ARDL e ECM**

    ---

    Bounds test de Pesaran, Error Correction Model, relacoes de longo prazo

    [:octicons-arrow-right-24: User Guide](user-guide/ardl/index.md)

-   :material-swap-horizontal: **Spillover Analysis**

    ---

    Diebold-Yilmaz estatico e dinamico, conectividade direcional e total

    [:octicons-arrow-right-24: User Guide](user-guide/spillover/index.md)

-   :material-chart-bar: **Visualizacao**

    ---

    Graficos de IRF, FEVD, forecasts, diagnosticos, ACF/PACF e filtros

    [:octicons-arrow-right-24: Visualization](visualization/index.md)

-   :material-file-document-outline: **Reports e CLI**

    ---

    Reports em HTML, LaTeX e Markdown; interface CLI para automacao

    [:octicons-arrow-right-24: API Reference](api/reports.md)

</div>

---

## Instalacao

```bash
pip install chronobox
```

Com extras opcionais:

```bash
pip install chronobox[dev]     # Ferramentas de desenvolvimento
pip install chronobox[docs]    # Ferramentas de documentacao
pip install chronobox[test]    # Ferramentas de teste
```

Veja o [Guia de Instalacao](getting-started/installation.md) para instrucoes detalhadas.

---

## Ecossistema NodesEcon

ChronoBox faz parte do ecossistema **NodesEcon** de bibliotecas para econometria aplicada em Python:

| Biblioteca | Dominio | Relacao com ChronoBox |
|:-----------|:--------|:----------------------|
| **[KalmanBox](https://github.com/NodesEcon/kalmanbox)** | State-space e Filtro de Kalman | **Dependencia direta** --- fornece state-space, MLE e filtro de Kalman usados internamente por ETS, ARIMA e modelos TVP |
| **[PanelBox](https://github.com/PanelBox-Econometrics-Model/panelbox)** | Dados em Painel | Biblioteca irma --- mesma filosofia de API, compartilha padroes de reports e visualizacao |

!!! info "Por que KalmanBox?"
    A representacao state-space e a base de modelos como ETS e TVP-VAR. Em vez de
    reimplementar o filtro de Kalman, a ChronoBox delega essa camada para o KalmanBox,
    garantindo estabilidade numerica e permitindo que melhorias no filtro beneficiem
    automaticamente todos os modelos da ChronoBox.

---

## Explore por Topico

<div class="grid cards" markdown>

-   :material-rocket-launch: **Getting Started**

    ---

    Instale e rode seu primeiro modelo em 5 minutos

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-book-open-variant: **User Guide**

    ---

    Guias completos para todas as familias de modelos

    [:octicons-arrow-right-24: User Guide](user-guide/index.md)

-   :material-notebook: **Tutorials**

    ---

    Notebooks interativos com exemplos praticos

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-code-tags: **API Reference**

    ---

    Referencia tecnica completa de classes e funcoes

    [:octicons-arrow-right-24: API Reference](api/index.md)

-   :material-sigma: **Theory**

    ---

    Fundamentos matematicos e background econometrico

    [:octicons-arrow-right-24: Theory](theory/arima-theory.md)

-   :material-test-tube: **Diagnostics**

    ---

    Testes de raiz unitaria, cointegracao, especificacao e estabilidade

    [:octicons-arrow-right-24: Diagnostics](diagnostics/index.md)

</div>

---

## Design Philosophy

ChronoBox e construida sobre quatro principios:

- **API Consistente** --- Todos os modelos seguem o padrao `model.fit()` / `result.forecast()` / `result.plot()`, facilitando a transicao entre familias de modelos.
- **Rigor Academico** --- Cada estimador segue artigos publicados e e validado contra R e Stata.
- **KalmanBox como Base** --- State-space e MLE delegados ao KalmanBox garantem estabilidade numerica sem duplicacao de codigo.
- **Output Publicavel** --- Reports em HTML, LaTeX e Markdown prontos para publicacao, com visualizacoes integradas.

---

## Citation

Se voce usar a ChronoBox em pesquisa academica, por favor cite:

```bibtex
@software{chronobox2026,
  title = {ChronoBox: Econometria de Series Temporais em Python},
  author = {NodesEcon Development Team},
  year = {2026},
  url = {https://github.com/NodesEcon/chronobox},
  version = {0.1.0}
}
```
