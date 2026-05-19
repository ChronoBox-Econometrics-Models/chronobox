---
title: "Johansen Cointegration Test"
description: "Teste de Johansen para cointegracao no chronobox — trace test, max-eigenvalue, cinco casos deterministicos, valores criticos e exemplos praticos."
---

# Johansen Cointegration Test

!!! info "Quick Reference"
    **Classe:** `chronobox.models.vecm.VECM.johansen_test()`
    **H₀:** Rank de cointegracao $= r$ (no maximo $r$ relacoes)
    **H₁ (trace):** Rank $> r$
    **H₁ (max-eigen):** Rank $= r + 1$
    **Distribuicao:** Johansen (nao-padrao, baseada em movimento browniano)
    **Valores criticos:** Osterwald-Lenum (1992)
    **R equivalente:** `urca::ca.jo()`

## Hipoteses

O procedimento de Johansen testa sequencialmente o rank de cointegracao de um sistema de $K$ variaveis $I(1)$. Para cada valor de $r = 0, 1, \ldots, K-1$:

### Trace Test

$$H_0: \text{rank} \leq r \quad \text{vs} \quad H_1: \text{rank} > r$$

O trace test pergunta: "existem **mais do que** $r$ relacoes de cointegracao?"

### Max-Eigenvalue Test

$$H_0: \text{rank} = r \quad \text{vs} \quad H_1: \text{rank} = r + 1$$

O max-eigenvalue test pergunta: "existe **exatamente mais uma** relacao de cointegracao alem de $r$?"

!!! tip "Qual usar?"
    Na pratica, o **trace test** e mais utilizado por ser mais robusto. Quando trace e max-eigenvalue discordam, a literatura recomenda seguir o trace test (Johansen & Juselius, 1990).

## Procedimento de Johansen

O teste baseia-se na estimacao de um VECM por maxima verossimilhanca (FIML):

$$\Delta \mathbf{y}_t = \boldsymbol{\Pi} \mathbf{y}_{t-1} + \sum_{i=1}^{p-1} \boldsymbol{\Gamma}_i \Delta \mathbf{y}_{t-i} + \boldsymbol{\mu} + \boldsymbol{\varepsilon}_t$$

onde a matriz $\boldsymbol{\Pi}$ contem a informacao de longo prazo. Se $\text{rank}(\boldsymbol{\Pi}) = r$, entao:

$$\boldsymbol{\Pi} = \boldsymbol{\alpha} \boldsymbol{\beta}^\top$$

| Componente | Dimensao | Interpretacao |
|:-----------|:---------|:-------------|
| $\boldsymbol{\Pi}$ | $K \times K$ | Matriz de impacto de longo prazo |
| $\boldsymbol{\alpha}$ | $K \times r$ | Velocidades de ajustamento (loading matrix) |
| $\boldsymbol{\beta}$ | $K \times r$ | Vetores de cointegracao |
| $\boldsymbol{\Gamma}_i$ | $K \times K$ | Dinamica de curto prazo |

### Reduced Rank Regression

O procedimento resolve um problema de **autovalores generalizados** a partir das matrizes de momentos dos residuos concentrados:

$$\det(\lambda \mathbf{S}_{11} - \mathbf{S}_{10} \mathbf{S}_{00}^{-1} \mathbf{S}_{01}) = 0$$

Os autovalores resultantes $\hat{\lambda}_1 \geq \hat{\lambda}_2 \geq \cdots \geq \hat{\lambda}_K$ medem a correlacao canonica entre $\mathbf{y}_{t-1}$ e $\Delta \mathbf{y}_t$ (apos concentrar a dinamica de curto prazo).

## Estatisticas de Teste

### Trace Statistic

$$\lambda_{\text{trace}}(r) = -T \sum_{i=r+1}^{K} \ln(1 - \hat{\lambda}_i)$$

Testa se **todos** os autovalores apos o $r$-esimo sao zero. Se a estatistica excede o valor critico, rejeita-se $H_0: \text{rank} \leq r$.

### Max-Eigenvalue Statistic

$$\lambda_{\max}(r) = -T \ln(1 - \hat{\lambda}_{r+1})$$

Testa se o $(r+1)$-esimo autovalor e zero. Foca na contribuicao marginal de uma relacao adicional.

!!! warning "Distribuicao Nao-Padrao"
    As estatisticas nao seguem distribuicoes $\chi^2$ padrao. Seguem distribuicoes funcionais de movimentos brownianos que dependem do modelo deterministico e do numero de variaveis. Os valores criticos sao tabulados por Osterwald-Lenum (1992).

## Cinco Casos Deterministicos

A especificacao dos termos deterministicos afeta os valores criticos e a interpretacao:

=== "Caso 1: `'nc'` — Sem termos deterministicos"

    $$\Delta \mathbf{y}_t = \boldsymbol{\alpha}\boldsymbol{\beta}^\top \mathbf{y}_{t-1} + \sum_{i=1}^{p-1} \boldsymbol{\Gamma}_i \Delta \mathbf{y}_{t-i} + \boldsymbol{\varepsilon}_t$$

    **Quando usar:** Raramente. Series sem drift e sem intercepto (oscilam em torno de zero).

=== "Caso 2: `'ci'` — Constante restrita ao ECM"

    $$\Delta \mathbf{y}_t = \boldsymbol{\alpha}(\boldsymbol{\beta}^\top \mathbf{y}_{t-1} + \mu_0) + \sum_{i=1}^{p-1} \boldsymbol{\Gamma}_i \Delta \mathbf{y}_{t-i} + \boldsymbol{\varepsilon}_t$$

    **Quando usar:** Series $I(1)$ sem tendencia deterministica mas com nivel de equilibrio diferente de zero. **Caso mais comum** em aplicacoes macroeconomicas.

=== "Caso 3: `'co'` — Constante irrestrita"

    $$\Delta \mathbf{y}_t = \boldsymbol{\alpha}\boldsymbol{\beta}^\top \mathbf{y}_{t-1} + \boldsymbol{\mu} + \sum_{i=1}^{p-1} \boldsymbol{\Gamma}_i \Delta \mathbf{y}_{t-i} + \boldsymbol{\varepsilon}_t$$

    **Quando usar:** Permite drift linear nas variaveis em niveis. Adequado quando as series apresentam tendencia linear.

=== "Caso 4: `'li'` — Tendencia restrita ao ECM"

    $$\Delta \mathbf{y}_t = \boldsymbol{\alpha}(\boldsymbol{\beta}^\top \mathbf{y}_{t-1} + \mu_0 + \rho t) + \boldsymbol{\mu} + \sum_{i=1}^{p-1} \boldsymbol{\Gamma}_i \Delta \mathbf{y}_{t-i} + \boldsymbol{\varepsilon}_t$$

    **Quando usar:** Relacao de cointegracao com tendencia deterministica (trend-stationary equilibrium).

=== "Caso 5: `'lo'` — Tendencia irrestrita"

    $$\Delta \mathbf{y}_t = \boldsymbol{\alpha}\boldsymbol{\beta}^\top \mathbf{y}_{t-1} + \boldsymbol{\mu} + \boldsymbol{\delta} t + \sum_{i=1}^{p-1} \boldsymbol{\Gamma}_i \Delta \mathbf{y}_{t-i} + \boldsymbol{\varepsilon}_t$$

    **Quando usar:** Raramente. Permite tendencia quadratica nos dados em niveis.

!!! tip "Recomendacao Pratica"
    Comece com `'ci'` (caso 2) para dados macroeconomicos. Use `'co'` (caso 3) se as series apresentarem tendencia linear visivel. Evite `'nc'` e `'lo'` — sao casos extremos raramente apropriados.

## Valores Criticos

Os valores criticos dependem de:

1. **Numero de variaveis** $K$ menos o rank sob $H_0$ ($K - r$)
2. **Caso deterministico** (`'nc'`, `'ci'`, `'co'`, `'li'`, `'lo'`)

Valores criticos selecionados (Osterwald-Lenum, 1992) para o **trace test** com constante irrestrita (`'co'`):

| $K - r$ | 90% | 95% | 99% |
|:--------|:----|:----|:----|
| 1 | 6.50 | 8.18 | 11.65 |
| 2 | 15.66 | 17.95 | 23.52 |
| 3 | 28.71 | 31.52 | 37.22 |
| 4 | 45.23 | 48.28 | 55.43 |
| 5 | 66.49 | 70.60 | 78.87 |

*O chronobox tabula valores ate $K - r = 12$ para os cinco casos.*

## Procedimento Sequencial

O teste e aplicado **sequencialmente**, comecando com $r = 0$:

1. Testar $H_0: r = 0$ vs $H_1: r > 0$
2. Se rejeitar, testar $H_0: r \leq 1$ vs $H_1: r > 1$
3. Continuar ate nao rejeitar
4. O primeiro $r$ em que nao se rejeita $H_0$ e o **rank de cointegracao estimado**

| Etapa | $H_0$ | Decisao | Conclusao |
|:------|:------|:--------|:----------|
| 1 | $r = 0$ | Rejeita | Pelo menos 1 relacao |
| 2 | $r \leq 1$ | Rejeita | Pelo menos 2 relacoes |
| 3 | $r \leq 2$ | Nao rejeita | **Rank = 2** |

## Exemplo Pratico

### Determinar o Rank de Cointegracao

```python
import numpy as np
from chronobox.models.vecm import VECM

# Gerar sistema com 1 relacao de cointegracao entre 3 variaveis I(1)
np.random.seed(42)
T = 300

# Tendencia estocastica comum
trend = np.cumsum(np.random.randn(T))

# y1, y2 seguem o trend comum; y3 e independente
y1 = trend + 0.5 * np.random.randn(T)
y2 = 2 * trend + np.random.randn(T)
y3 = np.cumsum(np.random.randn(T))  # I(1) independente

data = np.column_stack([y1, y2, y3])

# Teste de Johansen com constante restrita (caso 2)
model = VECM(lags=2, deterministic="ci")
joh = model.johansen_test(data)
print(joh.summary())
```

Saida esperada:

```
==============================================================================
  Johansen Cointegration Test
  Deterministic: ci
  Observations: 297
==============================================================================

  Trace Test
------------------------------------------------------------------------------
  H0: r<=     Eigenvalue   Trace Stat     90% CV     95% CV     99% CV
  --------------------------------------------------------------------------
  0              0.1523      62.4517      32.00      34.91      41.07 **
  1              0.0412      13.5231      17.85      19.96      24.60
  2              0.0038       1.1234       7.52       9.24      12.97
  Selected rank (trace, 5%): 1

  Max-Eigenvalue Test
------------------------------------------------------------------------------
  H0: r=      Eigenvalue  Max-Eig Stat     90% CV     95% CV     99% CV
  --------------------------------------------------------------------------
  0              0.1523      48.9286      19.77      22.00      26.81 **
  1              0.0412      12.3997      13.75      15.67      20.20
  2              0.0038       1.1234       7.52       9.24      12.97
  Selected rank (max-eig, 5%): 1

  ** denotes rejection at 5% significance level
==============================================================================
```

!!! tip "Interpretacao"
    - **Trace test**: Rejeita $H_0: r = 0$ (62.45 > 34.91) mas nao rejeita $H_0: r \leq 1$ (13.52 < 19.96). Rank = **1**.
    - **Max-eigenvalue**: Mesmo resultado — rank = **1**.
    - Conclusao: Existe **uma** relacao de cointegracao entre as 3 variaveis. Usar VECM com `coint_rank=1`.

### Acessando Resultados Programaticamente

```python
# Rank selecionado
print(f"Rank (trace): {joh.rank_trace}")
print(f"Rank (max-eig): {joh.rank_maxeig}")

# Autovalores
print(f"Eigenvalues: {joh.eigenvalues}")

# Estatisticas de teste
print(f"Trace stats: {joh.trace_stat}")
print(f"Max-eig stats: {joh.max_eig_stat}")

# Valores criticos a 5% (coluna 1)
print(f"Trace 5% CV: {joh.trace_crit[:, 1]}")
print(f"Max-eig 5% CV: {joh.max_eig_crit[:, 1]}")

# Vetores de cointegracao (colunas de eigenvectors)
print(f"Vetor de cointegracao (1o):\n{joh.eigenvectors[:, 0]}")
```

### Usando o Rank para Estimar o VECM

```python
# Usar o rank detectado pelo Johansen
model = VECM(lags=2, coint_rank=joh.rank_trace, deterministic="ci")
results = model.fit(data)
print(results.summary())
```

### Comparando Casos Deterministicos

```python
# Testar com diferentes especificacoes deterministicas
for det in ["nc", "ci", "co"]:
    model = VECM(lags=2, deterministic=det)
    joh = model.johansen_test(data)
    print(f"Deterministic='{det}': rank_trace={joh.rank_trace}, "
          f"rank_maxeig={joh.rank_maxeig}")
```

## Selecao de Lags

O numero de lags $p$ no VECM afeta o resultado do teste. Recomendacoes:

1. **Estimar um VAR em niveis** e selecionar lags por AIC/BIC
2. O VECM usa $p - 1$ lags em diferencas (se o VAR tem $p$ lags)
3. Verificar que os residuos do VECM nao apresentam autocorrelacao

!!! warning "Sensibilidade a Lags"
    Poucos lags podem deixar autocorrelacao nos residuos, invalidando o teste. Muitos lags reduzem os graus de liberdade e o poder. Use criterios de informacao para selecao.

## Limitacoes

1. **Requer series $I(1)$**: O teste assume que todas as variaveis sao integradas de mesma ordem. Nao funciona com ordens mistas — use o [Bounds Test](bounds-test.md) nesse caso
2. **Sensivel ao numero de lags**: Resultados podem mudar com a especificacao de lags
3. **Sensivel ao caso deterministico**: A escolha incorreta pode levar a conclusoes erradas
4. **Amostras pequenas**: Em amostras pequenas ($T < 50$), o teste tem baixo poder e pode sub-estimar o rank
5. **Quebras estruturais**: A presenca de quebras pode afetar os resultados — considere o teste de Gregory-Hansen

## Equivalente R

=== "urca"

    ```r
    library(urca)

    # Teste de Johansen
    # type: "trace" ou "eigen" (max-eigenvalue)
    # ecdet: "none" (nc), "const" (ci), "trend" (li)
    # K: numero de lags em niveis (p, nao p-1)
    joh <- ca.jo(data, type = "trace", ecdet = "const", K = 2)
    summary(joh)

    # Equivalencias de 'deterministic':
    # chronobox 'nc' → urca ecdet = "none"
    # chronobox 'ci' → urca ecdet = "const"
    # chronobox 'li' → urca ecdet = "trend"
    ```

=== "tsDyn"

    ```r
    library(tsDyn)

    # VECM com teste de Johansen integrado
    vecm <- VECM(data, lag = 1, r = 1, estim = "ML", include = "const")
    summary(vecm)
    ```

## See Also

- [Engle-Granger](engle-granger.md) — Alternativa mais simples para 2 variaveis
- [Bounds Test](bounds-test.md) — Para ordens de integracao mistas
- [Cointegration Tests](index.md) — Visao geral
- [User Guide: VECM](../../user-guide/var/vecm.md) — Modelagem VECM
- [Theory: VECM](../../theory/vecm-theory.md) — Fundamentos teoricos completos

## Referencias

- Johansen, S. (1988). "Statistical analysis of cointegration vectors." *Journal of Economic Dynamics and Control*, 12(2-3), 231-254.
- Johansen, S. (1991). "Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models." *Econometrica*, 59(6), 1551-1580.
- Johansen, S. & Juselius, K. (1990). "Maximum likelihood estimation and inference on cointegration — with applications to the demand for money." *Oxford Bulletin of Economics and Statistics*, 52(2), 169-210.
- Osterwald-Lenum, M. (1992). "A note with quantiles of the asymptotic distribution of the maximum likelihood cointegration rank test statistics." *Oxford Bulletin of Economics and Statistics*, 54(3), 461-472.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
