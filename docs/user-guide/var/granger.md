---
title: Granger Causality
description: Testes de causalidade de Granger --- bivariado, em bloco, limitacoes e interpretacao.
---

# Granger Causality

!!! info "Quick Reference"
    - **Funcao**: `chronobox.tests_stat.granger_causality_test`
    - **Import**: `from chronobox.tests_stat import granger_causality_test`
    - **R equivalente**: `vars::causality()`, `lmtest::grangertest()`
    - **Estatisticas**: F-test, $\chi^2$ (Wald)

---

## Overview

O teste de causalidade de Granger (1969) avalia se valores passados de uma
variavel $X$ contem informacao util para prever outra variavel $Y$, **alem da
informacao ja contida nos valores passados de $Y$**.

Formalmente, $X$ **Granger-causa** $Y$ se:

$$
\text{MSE}[E(Y_t | Y_{t-1}, \ldots)] > \text{MSE}[E(Y_t | Y_{t-1}, \ldots, X_{t-1}, \ldots)]
$$

Nao se trata de causalidade no sentido filosofico --- e um teste de
**precedencia temporal** e **poder preditivo incremental**.

### Quando usar

- Verificar se uma variavel ajuda a prever outra
- Explorar a direcao da relacao entre variaveis (unidirecional, bidirecional)
- Informar a especificacao de modelos VAR (incluir/excluir variaveis)
- Testar hipoteses economicas (ex: moeda causa output?)

---

## Formulacao Matematica

### Teste Bivariado

Considere o VAR(p) com duas variaveis:

$$
y_t = c_1 + \sum_{i=1}^{p} a_{11,i}\, y_{t-i} + \sum_{i=1}^{p} a_{12,i}\, x_{t-i} + u_{1t}
$$

$$
x_t = c_2 + \sum_{i=1}^{p} a_{21,i}\, y_{t-i} + \sum_{i=1}^{p} a_{22,i}\, x_{t-i} + u_{2t}
$$

O teste de Granger para "$X$ Granger-causa $Y$" testa:

$$
H_0: a_{12,1} = a_{12,2} = \cdots = a_{12,p} = 0
$$

Se $H_0$ e rejeitada, os lags de $X$ sao conjuntamente significantes na equacao
de $Y$ --- $X$ Granger-causa $Y$.

### Estatistica F

A estatistica do teste e:

$$
F = \frac{(\text{RSS}_R - \text{RSS}_U) / p}{\text{RSS}_U / (T - 2p - 1)} \sim F(p, T-2p-1)
$$

onde:

- $\text{RSS}_R$ = soma dos quadrados dos residuos do modelo **restrito** (sem lags de $X$)
- $\text{RSS}_U$ = soma dos quadrados dos residuos do modelo **irrestrito** (com lags de $X$)

### Teste em Bloco (Block Granger)

No contexto de um VAR com $K$ variaveis, o teste em bloco verifica se um
**conjunto** de variaveis Granger-causa outra variavel. Para testar se as
variaveis $\{x_2, x_3\}$ Granger-causam $x_1$:

$$
H_0: \mathbf{A}_{12,i} = \mathbf{A}_{13,i} = \mathbf{0} \quad \forall\, i = 1, \ldots, p
$$

A estatistica e um teste de Wald com distribuicao $\chi^2$:

$$
W = T(\hat{\boldsymbol{\Sigma}}_R - \hat{\boldsymbol{\Sigma}}_U) \hat{\boldsymbol{\Sigma}}_U^{-1} \sim \chi^2(2p)
$$

### Instantaneous Causality

Alem da causalidade de Granger (temporal), pode-se testar **causalidade
instantanea** --- se $X_t$ e $Y_t$ sao correlacionados contemporaneamente:

$$
H_0: \sigma_{12} = 0
$$

onde $\sigma_{12}$ e a covariancia entre $u_{1t}$ e $u_{2t}$.

---

## Quick Example

```python
from chronobox import VAR
from chronobox.tests_stat import granger_causality_test
from chronobox.datasets import load_macro

# Carregar dados
data = load_macro()

# Teste bivariado: inflacao Granger-causa GDP?
gc = granger_causality_test(data, cause='infl', effect='gdp', max_lags=4)
print(gc.summary())

# Teste via VAR ajustado
model = VAR(lags=2)
results = model.fit(data)
gc_var = results.test_granger_causality(cause='infl', effect='gdp')
print(gc_var.summary())
```

---

## Guia Detalhado

### Teste Bivariado Direto

```python
from chronobox.tests_stat import granger_causality_test

gc = granger_causality_test(
    data,              # DataFrame com as series
    cause='infl',      # Variavel "causa"
    effect='gdp',      # Variavel "efeito"
    max_lags=4,        # Maximo de lags a testar
    ic='aic'           # Criterio para selecao de lags
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `data` | `pd.DataFrame` | --- | Dados multivariados |
| `cause` | `str \| list[str]` | --- | Variavel(is) "causa" |
| `effect` | `str` | --- | Variavel "efeito" |
| `max_lags` | `int` | `4` | Maximo de lags |
| `ic` | `str \| None` | `None` | Criterio para selecao de lags (`'aic'`, `'bic'`) |

### Resultados

| Atributo | Tipo | Descricao |
|---|---|---|
| `statistic` | `float` | Estatistica F ou $\chi^2$ |
| `pvalue` | `float` | p-valor |
| `df` | `tuple` | Graus de liberdade |
| `lags` | `int` | Numero de lags utilizado |
| `test_type` | `str` | `'f'` ou `'chi2'` |

### Teste a Partir do VAR

```python
from chronobox import VAR

model = VAR(lags=4)
results = model.fit(data)

# Teste individual: rate Granger-causa gdp?
gc1 = results.test_granger_causality(cause='rate', effect='gdp')
print(f"F = {gc1.statistic:.4f}, p = {gc1.pvalue:.4f}")

# Teste em bloco: {infl, rate} Granger-causam gdp?
gc_block = results.test_granger_causality(
    cause=['infl', 'rate'],
    effect='gdp'
)
print(f"chi2 = {gc_block.statistic:.4f}, p = {gc_block.pvalue:.4f}")
```

### Teste Bidirecional

```python
# Testar ambas as direcoes
gc_xy = results.test_granger_causality(cause='infl', effect='gdp')
gc_yx = results.test_granger_causality(cause='gdp', effect='infl')

print(f"infl → gdp: p = {gc_xy.pvalue:.4f}")
print(f"gdp → infl: p = {gc_yx.pvalue:.4f}")
```

| Resultado | Interpretacao |
|---|---|
| Ambos significantes | Causalidade bidirecional (feedback) |
| So $X \to Y$ significante | Causalidade unidirecional |
| Nenhum significante | Sem relacao de Granger |

### Tabela Completa de Causalidade

```python
# Testar todas as direcoes pairwise
import pandas as pd

variables = data.columns.tolist()
results_table = []

for cause in variables:
    for effect in variables:
        if cause != effect:
            gc = results.test_granger_causality(cause=cause, effect=effect)
            results_table.append({
                'cause': cause,
                'effect': effect,
                'F_stat': gc.statistic,
                'p_value': gc.pvalue,
                'significant': '***' if gc.pvalue < 0.01
                    else '**' if gc.pvalue < 0.05
                    else '*' if gc.pvalue < 0.10
                    else ''
            })

df = pd.DataFrame(results_table)
print(df.to_string(index=False))
```

```text
  cause  effect   F_stat   p_value significant
   gdp    infl    3.421     0.012          **
   gdp    rate    5.213     0.001         ***
  infl     gdp    1.823     0.134
  infl    rate    2.456     0.052           *
  rate     gdp    2.891     0.031          **
  rate    infl    4.123     0.004         ***
```

### Sensibilidade ao Numero de Lags

```python
# Testar com diferentes numeros de lags
for p in range(1, 9):
    model = VAR(lags=p)
    res = model.fit(data)
    gc = res.test_granger_causality(cause='infl', effect='gdp')
    print(f"p={p}: F={gc.statistic:.3f}, p-value={gc.pvalue:.4f}")
```

!!! warning "Sensibilidade aos lags"
    Os resultados do teste de Granger podem variar significativamente com
    o numero de lags. Use criterios de informacao (AIC, BIC) para selecionar
    $p$, e reporte os resultados para diferentes especificacoes.

---

## Limitacoes

!!! warning "Limitacoes importantes"

    1. **Nao e causalidade verdadeira**: Granger causality testa *precedencia
       temporal*, nao causalidade no sentido de intervenção. Uma variavel
       omitida $Z$ pode causar tanto $X$ quanto $Y$, gerando Granger
       causality espuria.

    2. **Sensivel a especificacao**: o numero de lags, variaveis incluidas e
       transformacoes dos dados afetam o resultado.

    3. **Requer estacionariedade**: o teste padrao assume series estacionarias.
       Para series I(1), use o teste em niveis com cautela (Toda-Yamamoto) ou
       teste no contexto do VECM.

    4. **Nao detecta relacoes nao-lineares**: o teste e baseado em regressoes
       lineares. Relacoes nao lineares podem nao ser detectadas.

    5. **Problemas com agregacao temporal**: dados em frequencias diferentes
       podem produzir resultados contraditórios.

---

## Diagnosticos

### Verificar Pre-requisitos

```python
from chronobox.tests_stat import adf_test

# 1. Estacionariedade
for col in data.columns:
    adf = adf_test(data[col])
    status = "I(0)" if adf.pvalue < 0.05 else "I(1)"
    print(f"{col}: ADF p={adf.pvalue:.4f} → {status}")

# 2. Adequacao do VAR (residuos)
from chronobox.tests_stat import portmanteau_test

pt = portmanteau_test(results.residuals, lags=12)
print(f"Portmanteau p-value: {pt.pvalue:.4f}")
# Residuos devem ser white noise para o teste ser valido
```

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import VAR
    from chronobox.tests_stat import granger_causality_test

    # Via funcao direta
    gc = granger_causality_test(data, cause='infl', effect='gdp', max_lags=4)

    # Via VAR
    model = VAR(lags=4)
    results = model.fit(data)
    gc = results.test_granger_causality(cause='infl', effect='gdp')
    print(gc.summary())
    ```

=== "vars (R)"

    ```r
    library(vars)

    fit <- VAR(y, p = 4, type = "const")

    # Teste de Granger
    causality(fit, cause = "infl")
    # Testa se 'infl' Granger-causa todas as outras variaveis

    # Teste bivariado puro
    library(lmtest)
    grangertest(gdp ~ infl, order = 4, data = df)
    ```

**Mapeamento de parametros**:

| chronobox | vars / lmtest (R) | Descricao |
|---|---|---|
| `granger_causality_test(data, cause, effect)` | `grangertest(y ~ x, order)` | Teste bivariado |
| `results.test_granger_causality(cause, effect)` | `causality(fit, cause)` | Teste via VAR |
| `gc.pvalue` | Extrair do output | p-valor |
| `gc.statistic` | Extrair do output | Estatistica F |

!!! tip "Diferenca no vars::causality()"
    A funcao `causality()` do pacote `vars` em R testa se a variavel
    `cause` Granger-causa **todas as outras** variaveis conjuntamente.
    No chronobox, voce especifica `cause` e `effect` individualmente,
    o que permite testes mais granulares.

---

## Referencias

- Granger, C. W. J. (1969). Investigating Causal Relations by Econometric
  Models and Cross-spectral Methods. *Econometrica*, 37(3), 424--438.
- Toda, H. Y. & Yamamoto, T. (1995). Statistical Inference in Vector
  Autoregressions with Possibly Integrated Processes. *Journal of
  Econometrics*, 66(1-2), 225--250.
- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
  Capitulo 2.3.1.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
  Capitulo 11.2.
