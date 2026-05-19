---
title: Decomposicao de Series Temporais
description: Metodos de decomposicao sazonal --- STL e X-13 ARIMA-SEATS para separar tendencia, sazonalidade e residuo.
---

# Decomposicao de Series Temporais

A **decomposicao** de uma serie temporal consiste em separa-la em componentes
nao observados --- tipicamente **tendencia**, **sazonalidade** e **residuo** ---
para facilitar a analise, o ajuste sazonal e a modelagem.

---

## O que e decomposicao?

Dada uma serie temporal $y_t$, o objetivo e expressa-la como funcao de
componentes estruturais:

- **Tendencia** ($T_t$): movimento de longo prazo (crescimento, declinio)
- **Sazonalidade** ($S_t$): padroes que se repetem em intervalos fixos
- **Residuo** ($R_t$): variacao irregular nao explicada

A decomposicao e utilizada para:

- **Analise exploratoria**: entender a estrutura da serie
- **Ajuste sazonal**: remover efeitos sazonais para comparar periodos
- **Pre-processamento**: isolar componentes antes da modelagem
- **Deteccao de anomalias**: identificar observacoes atipicas no residuo

---

## Aditiva vs Multiplicativa

### Decomposicao Aditiva

$$
y_t = T_t + S_t + R_t
$$

Os componentes sao **somados**. Adequada quando a amplitude sazonal e
**constante** ao longo do tempo, independente do nivel da serie.

### Decomposicao Multiplicativa

$$
y_t = T_t \cdot S_t \cdot R_t
$$

Os componentes sao **multiplicados**. Adequada quando a amplitude sazonal
e **proporcional ao nivel** da serie --- quanto maior o nivel, maior a
oscilacao sazonal.

!!! tip "Como escolher?"
    - Se o grafico da serie mostra oscilacoes sazonais de **amplitude constante**,
      use decomposicao **aditiva**
    - Se a amplitude sazonal **cresce com o nivel**, use **multiplicativa**
    - Na duvida, aplique $\log$ a serie (transforma multiplicativa em aditiva):

    $$
    \log(y_t) = \log(T_t) + \log(S_t) + \log(R_t)
    $$

### Comparacao Visual

| Caracteristica | Aditiva | Multiplicativa |
|---|---|---|
| Formula | $y_t = T_t + S_t + R_t$ | $y_t = T_t \cdot S_t \cdot R_t$ |
| Amplitude sazonal | Constante | Proporcional ao nivel |
| Residuo | Escala absoluta | Escala relativa (ratio) |
| Dados negativos | Permitidos | Requer $y_t > 0$ |
| Transformacao log | N/A | Converte para aditiva |

---

## Quando decompor?

| Cenario | Decomposicao ajuda? |
|---|---|
| Analise exploratoria de serie sazonal | Sim |
| Ajuste sazonal para indicadores economicos | Sim |
| Identificar mudancas na tendencia | Sim |
| Detectar outliers no componente residual | Sim |
| Alimentar modelo ARIMA com dados dessazonalizados | Sim |
| Serie sem sazonalidade | Nao (use filtros como HP) |

---

## Metodos Disponíveis

O **chronobox** oferece tres metodos de decomposicao:

<div class="grid cards" markdown>

-   :material-chart-timeline:{ .lg .middle } **Decomposicao Classica**

    ---

    Media movel centrada. Simples e didatico, mas limitado.

    ```python
    from chronobox import ClassicalDecomposition

    cd = ClassicalDecomposition(period=12, model='additive')
    result = cd.fit(y)
    ```

-   :material-sine-wave:{ .lg .middle } **STL**

    ---

    LOESS iterativo. Robusto, flexivel, padrao moderno.

    [:octicons-arrow-right-24: STL](stl.md)

-   :material-office-building:{ .lg .middle } **X-13 ARIMA-SEATS**

    ---

    Padrao de agencias estatisticas. Ajuste sazonal oficial.

    [:octicons-arrow-right-24: X-13 ARIMA-SEATS](x13.md)

</div>

---

## Quick Comparison

```python
import numpy as np
from chronobox import STL, ClassicalDecomposition
from chronobox.datasets import load_dataset

# Carregar dados airline (passageiros mensais)
y = load_dataset("airline")

# Decomposicao classica
cd = ClassicalDecomposition(period=12, model='additive')
result_cd = cd.fit(y.values)

# STL
stl = STL(period=12)
result_stl = stl.fit(y.values)

# Comparar residuos
print(f"Classica - Std residuo: {np.nanstd(result_cd.remainder):.2f}")
print(f"STL      - Std residuo: {np.nanstd(result_stl.remainder):.2f}")
```

| Metodo | Vantagens | Limitacoes |
|---|---|---|
| Classica | Simples, didatico | Perde pontas, sazonalidade fixa |
| STL | Robusto, sazonalidade variavel | Apenas aditivo |
| X-13 ARIMA-SEATS | Padrao oficial, model-based | Requer executavel externo |

---

## Proximos Passos

- [:octicons-arrow-right-24: STL Decomposition](stl.md) --- metodo recomendado para uso geral
- [:octicons-arrow-right-24: X-13 ARIMA-SEATS](x13.md) --- ajuste sazonal oficial
