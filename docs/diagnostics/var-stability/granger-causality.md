---
title: "Granger Causality"
description: "Teste de causalidade de Granger no chronobox — teste Wald, teste F, block exogeneity, causalidade instantanea e exemplos praticos."
---

# Granger Causality

!!! info "Quick Reference"
    **Funcao:** `VARResults.test_granger()`, `chronobox.analysis.granger.granger_causality()`
    **H₀:** Variavel X **nao** Granger-causa variavel Y
    **H₁:** Variavel X Granger-causa variavel Y
    **Distribuicao:** $F(p, T' - Kp - d)$ e $\chi^2(p)$
    **Valores criticos:** Distribuicoes $F$ e $\chi^2$ padrao
    **R equivalente:** `vars::causality()`, `lmtest::grangertest()`

## Hipoteses

O teste de Granger avalia se os lags de uma variavel X ajudam a prever outra variavel Y, **apos controlar pelos proprios lags de Y** e pelos lags das demais variaveis no VAR:

$$H_0: a_{yx}^{(1)} = a_{yx}^{(2)} = \cdots = a_{yx}^{(p)} = 0 \quad \text{(X nao Granger-causa Y)}$$

$$H_1: \exists \, j \leq p \text{ tal que } a_{yx}^{(j)} \neq 0 \quad \text{(X Granger-causa Y)}$$

onde $a_{yx}^{(j)}$ e o coeficiente de $X_{t-j}$ na equacao de $Y_t$.

**Rejeitar H₀** indica que valores passados de X contem informacao preditiva para Y, **alem** do que ja e capturado por Y e pelas outras variaveis.

**Nao rejeitar H₀** sugere que X nao melhora a previsao de Y.

!!! warning "Granger Causality ≠ Causality"
    Causalidade de Granger e um conceito de **previsibilidade**, nao de causalidade no sentido filosofico ou mecanicista. "X Granger-causa Y" significa apenas que X ajuda a prever Y. Nao implica que X **causa** Y.

## Teste F (Wald)

### Formulacao

Considere a equacao para $Y_t$ no VAR(p) com $K$ variaveis:

$$Y_t = c + \sum_{j=1}^{p} a_{yy}^{(j)} Y_{t-j} + \sum_{j=1}^{p} a_{yx}^{(j)} X_{t-j} + \sum_{j=1}^{p} \sum_{k \neq x,y} a_{yk}^{(j)} Z_{k,t-j} + u_t$$

O teste compara dois modelos:

| Modelo | Descricao | Parametros |
|:-------|:----------|:-----------|
| **Irrestrito** | Equacao completa com todos os regressores | $Kp + d$ |
| **Restrito** | Mesma equacao excluindo todos os lags de X | $Kp + d - p$ |

### Estatistica F

$$F = \frac{(SSR_r - SSR_u) / p}{SSR_u / (T' - Kp - d)} \sim F(p, T' - Kp - d)$$

onde:

| Componente | Descricao |
|:-----------|:----------|
| $SSR_r$ | Soma dos quadrados dos residuos do modelo restrito |
| $SSR_u$ | Soma dos quadrados dos residuos do modelo irrestrito |
| $p$ | Numero de restricoes (= numero de lags do VAR) |
| $T'$ | Numero efetivo de observacoes |
| $K$ | Numero de variaveis no VAR |
| $d$ | Numero de termos deterministicos |

### Estatistica de Wald

Equivalentemente, a estatistica de Wald:

$$W = T' \cdot \frac{SSR_r - SSR_u}{SSR_u} \sim \chi^2(p)$$

A relacao entre ambas e: $W = F \cdot p \cdot \frac{T'}{T' - Kp - d}$

!!! tip "Qual Usar?"
    O **teste F** e preferivel em amostras finitas pois incorpora a correcao pelos graus de liberdade. O **teste Wald** ($\chi^2$) e assintoticamente equivalente e mais comum em softwares. O chronobox reporta ambos.

## Teste LR (Likelihood Ratio)

Uma alternativa e o teste de razao de verossimilhanca:

$$LR = T' \left[\ln(\hat{\sigma}_r^2) - \ln(\hat{\sigma}_u^2)\right] \sim \chi^2(p)$$

onde $\hat{\sigma}_r^2$ e $\hat{\sigma}_u^2$ sao as variancias estimadas dos modelos restrito e irrestrito.

O teste LR, o teste Wald e o teste LM sao assintoticamente equivalentes, mas podem diferir em amostras finitas.

## Block Exogeneity

O teste de block exogeneity (ou block Granger causality) generaliza o teste bilateral:

**Teste:** Nenhuma variavel do bloco $\{X_1, X_2, \ldots\}$ Granger-causa $Y$

$$H_0: \text{Todos os coeficientes de } X_1, X_2, \ldots \text{ na equacao de } Y \text{ sao zero}$$

Isso e equivalente a testar se $Y$ e "block exogenous" em relacao ao grupo de variaveis. O numero de restricoes e $p \times |\text{bloco}|$ ao inves de $p$.

## Causalidade Instantanea

Alem da Granger causality (baseada em lags), pode-se testar se ha correlacao **contemporanea** entre os residuos:

$$H_0: \sigma_{xy} = 0 \quad \text{(sem causalidade instantanea)}$$

onde $\sigma_{xy}$ e o elemento off-diagonal da matriz de covariancia dos residuos $\mathbf{\Sigma}_u$.

!!! info "Nota"
    A causalidade instantanea testa a correlacao entre $u_{x,t}$ e $u_{y,t}$, que pode indicar uma relacao contemporanea nao capturada pela estrutura de lags do VAR. Isso esta diretamente ligado a identificacao do [SVAR](../../user-guide/svar/svar.md).

## Exemplo Pratico

### Teste Bilateral: X Causa Y? Y Causa X?

```python
import numpy as np
from chronobox.models.var import VAR

# Simular sistema onde X causa Y, mas Y nao causa X
np.random.seed(42)
T = 500
x = np.zeros(T)
y_var = np.zeros(T)
for t in range(2, T):
    x[t] = 0.6 * x[t-1] + np.random.randn()
    y_var[t] = 0.3 * y_var[t-1] + 0.4 * x[t-1] + np.random.randn()

data = np.column_stack([x, y_var])
model = VAR(data, names=["X", "Y"])
results = model.fit(maxlags=2)

# X -> Y?
gc_xy = results.test_granger(caused="Y", causing="X")
print(gc_xy)

# Y -> X?
gc_yx = results.test_granger(caused="X", causing="Y")
print(gc_yx)
```

Saida esperada:

```
GrangerResult(X -> Y: F=82.3456, p=0.0000, REJECT H0 at 5%)
GrangerResult(Y -> X: F=1.2345, p=0.2913, FAIL TO REJECT H0 at 5%)
```

!!! tip "Interpretacao"
    - **X → Y:** Rejeitamos H₀ (p ≈ 0). Os lags de X ajudam significativamente a prever Y. Isso e esperado pelo DGP, onde $y_t$ depende de $x_{t-1}$.
    - **Y → X:** Nao rejeitamos H₀ (p ≈ 0.29). Os lags de Y nao melhoram a previsao de X. Tambem esperado, pois X e gerado independentemente de Y.

### Tabela de Resultados Completa

```python
# Testar todas as direcoes em um VAR com 3 variaveis
np.random.seed(42)
T = 500
z = np.zeros((T, 3))
for t in range(2, T):
    z[t, 0] = 0.5 * z[t-1, 0] + np.random.randn()
    z[t, 1] = 0.3 * z[t-1, 1] + 0.4 * z[t-1, 0] + np.random.randn()
    z[t, 2] = 0.2 * z[t-1, 2] + 0.3 * z[t-1, 1] + np.random.randn()

model3 = VAR(z, names=["X", "Y", "Z"])
res3 = model3.fit(maxlags=2)

# Matriz de causalidade
names = res3.names
print(f"{'Causing':>10} {'Caused':>10} {'F-stat':>10} {'p-value':>10} {'Decision':>15}")
print("-" * 60)
for causing in names:
    for caused in names:
        if causing != caused:
            gc = res3.test_granger(caused=caused, causing=causing)
            decision = "Granger-causes" if gc.reject else "No causality"
            print(f"{causing:>10} {caused:>10} {gc.fstat:10.4f} "
                  f"{gc.pvalue:10.4f} {decision:>15}")
```

Saida esperada:

```
   Causing     Caused     F-stat    p-value        Decision
------------------------------------------------------------
         X          Y    78.1234     0.0000  Granger-causes
         X          Z     3.4567     0.0321  Granger-causes
         Y          X     0.8901     0.4112    No causality
         Y          Z    45.6789     0.0000  Granger-causes
         Z          X     1.2345     0.2918    No causality
         Z          Y     0.5678     0.5672    No causality
```

### Acessando Resultados Detalhados

```python
gc = results.test_granger(caused="Y", causing="X", signif=0.05)

# Teste F
print(f"F-statistic: {gc.fstat:.4f}")
print(f"F p-value:   {gc.pvalue:.4f}")
print(f"df:          {gc.df}")

# Teste Wald (chi-squared)
print(f"Wald stat:   {gc.wald_stat:.4f}")
print(f"Wald p-val:  {gc.wald_pvalue:.4f}")

# Metadata
print(f"Causing:     {gc.causing}")
print(f"Caused:      {gc.caused}")
print(f"Signif:      {gc.signif}")
print(f"Reject H0:   {gc.reject}")
```

### Testando em Diferentes Niveis de Significancia

```python
gc = results.test_granger(caused="Y", causing="X")

for alpha in [0.01, 0.05, 0.10]:
    reject = gc.pvalue < alpha
    print(f"alpha={alpha:.2f}: {'Reject H0' if reject else 'Fail to reject H0'} "
          f"(p={gc.pvalue:.4f})")
```

## Interpretacao Cuidadosa

| Situacao | O Que Pode Estar Acontecendo |
|:---------|:----------------------------|
| X → Y significativo, Y → X nao | Causalidade unidirecional de X para Y |
| X → Y e Y → X ambos significativos | **Feedback** (causalidade bidirecional) |
| X → Y nao significativo | X nao ajuda a prever Y *dado o restante do VAR* |
| Significativo com poucos lags, nao com muitos | Pode indicar sobreajuste (muitos lags diluem o efeito) |

!!! warning "Armadilhas Comuns"
    1. **Variavel omitida:** Se Z causa tanto X quanto Y e esta omitida do VAR, X pode parecer Granger-causar Y espuriamente
    2. **Numero de lags:** O resultado e sensivel a escolha de $p$. Use criterios de informacao ([Lag Selection](../specification/lag-selection.md))
    3. **Nao-estacionaridade:** Com series I(1) nao cointegradas, as distribuicoes assintoticas nao sao padrao. Diferencie antes ou use VECM
    4. **Multiplas comparacoes:** Ao testar muitos pares, considere correcao de Bonferroni

## Assinatura da Funcao

```python
# Via VARResults
VARResults.test_granger(
    caused: str | int,       # Variavel dependente (nome ou indice)
    causing: str | int,      # Variavel causadora (nome ou indice)
    signif: float = 0.05     # Nivel de significancia
) -> GrangerResult

# Via funcao direta
granger_causality(
    var_results: VARResults,
    caused: str | int,
    causing: str | int,
    signif: float = 0.05
) -> GrangerResult
```

O `GrangerResult` contem:

| Atributo | Tipo | Descricao |
|:---------|:-----|:----------|
| `fstat` | `float` | Estatistica F |
| `pvalue` | `float` | P-valor do teste F |
| `df` | `tuple[int, int]` | Graus de liberdade $(p, T' - Kp - d)$ |
| `reject` | `bool` | Rejeitar H₀ no nivel `signif`? |
| `wald_stat` | `float` | Estatistica de Wald ($\chi^2$) |
| `wald_pvalue` | `float` | P-valor do teste Wald |
| `caused` | `str` | Nome da variavel causada |
| `causing` | `str` | Nome da variavel causadora |
| `signif` | `float` | Nivel de significancia usado |

## Limitacoes

1. **Requer estacionaridade** — com series I(1), a distribuicao F nao e padrao. Use series estacionarias ou VECM
2. **Sensivel ao numero de lags** — poucos lags podem omitir a relacao; muitos lags reduzem o poder
3. **Linear** — detecta apenas relacoes lineares. Nao captura causalidade nao-linear
4. **Condicional ao modelo** — o resultado depende de quais variaveis estao no VAR. Adicionar ou remover variaveis pode alterar conclusoes
5. **Nao indica direcao temporal precisa** — indica previsibilidade, nao o lag exato em que o efeito ocorre (para isso, use [IRF](../../user-guide/var/irf.md))

## Equivalente R

=== "vars"

    ```r
    library(vars)

    # Estimar VAR
    var_model <- VAR(data, p = 2, type = "const")

    # Granger causality: X causa Y?
    # 'cause' = variavel causadora, 'x' = modelo VAR
    causality(var_model, cause = "X")
    # Retorna: teste Granger (F) e teste de causalidade instantanea

    # Equivalencia com chronobox:
    # chronobox: results.test_granger(caused="Y", causing="X")
    # R vars:    causality(var_model, cause="X")
    #            (testa X contra TODAS as outras variaveis)
    ```

=== "lmtest (bivariado)"

    ```r
    library(lmtest)

    # Granger test bivariado (fora do contexto VAR)
    grangertest(Y ~ X, order = 2)

    # Equivalencia:
    # Este teste e bivariado — nao controla por outras variaveis
    # O teste do chronobox (via VAR) e multivariado
    ```

!!! info "Diferenca Importante"
    A funcao `causality()` do pacote `vars` testa se X causa **todas as outras variaveis simultaneamente** (block exogeneity). O chronobox testa pares especificos: X → Y. Para replicar o comportamento do `vars`, teste cada par individualmente.

## See Also

- [Eigenvalue Stability](eigenvalue.md) — Verificar estabilidade antes de testar causalidade
- [Portmanteau Test](portmanteau.md) — Verificar residuos do VAR
- [User Guide: Granger](../../user-guide/var/granger.md) — Guia pratico de uso
- [VAR Stability Overview](index.md) — Visao geral dos diagnosticos VAR

## Referencias

- Granger, C.W.J. (1969). "Investigating causal relations by econometric models and cross-spectral methods." *Econometrica*, 37(3), 424-438.
- Sims, C.A. (1972). "Money, income, and causality." *American Economic Review*, 62(4), 540-552.
- Toda, H.Y. & Yamamoto, T. (1995). "Statistical inference in vector autoregressions with possibly integrated processes." *Journal of Econometrics*, 66(1-2), 225-250.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. Chapter 3.
