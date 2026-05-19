---
title: "Portmanteau Test"
description: "Teste portmanteau multivariado (Ljung-Box) para autocorrelacao residual em modelos VAR no chronobox — estatistica Q, Hosking vs Li-McLeod e exemplos praticos."
---

# Portmanteau Test (Multivariate)

!!! info "Quick Reference"
    **Funcao:** `VARResults.test_whiteness()`
    **H₀:** Residuos sao **white noise multivariado** (sem autocorrelacao)
    **H₁:** Residuos possuem autocorrelacao serial
    **Distribuicao:** $\chi^2(K^2(h - p))$
    **Valores criticos:** Distribuicao $\chi^2$ padrao
    **R equivalente:** `vars::serial.test()`, `MTS::mq()`

## Hipoteses

O teste portmanteau multivariado avalia se os residuos do VAR sao **white noise multivariado** — ou seja, sem autocorrelacao serial em nenhuma combinacao de variaveis:

$$H_0: \mathbf{C}_1 = \mathbf{C}_2 = \cdots = \mathbf{C}_h = \mathbf{0} \quad \text{(white noise multivariado)}$$

$$H_1: \exists \, i \leq h \text{ tal que } \mathbf{C}_i \neq \mathbf{0} \quad \text{(autocorrelacao presente)}$$

onde $\mathbf{C}_i$ e a **matriz de autocovariancia cruzada** dos residuos no lag $i$:

$$\mathbf{C}_i = \frac{1}{T'} \sum_{t=i+1}^{T'} \hat{\mathbf{u}}_t \hat{\mathbf{u}}_{t-i}'$$

**Rejeitar H₀** indica que o modelo VAR nao capturou toda a estrutura de dependencia temporal dos dados — os lags sao insuficientes ou a especificacao esta incorreta.

**Nao rejeitar H₀** sugere que os residuos nao possuem autocorrelacao significativa e o modelo esta adequadamente especificado.

## Por Que o Teste Univariado Nao Basta?

O teste [Ljung-Box univariado](../specification/ljung-box.md) testa autocorrelacao em **cada serie** separadamente. Em um VAR com $K$ variaveis, o teste multivariado e necessario porque:

1. **Autocorrelacao cruzada** — $u_{1,t}$ pode ser correlacionado com $u_{2,t-j}$ sem que nenhuma serie individual mostre autocorrelacao
2. **Poder do teste** — testar $K$ series separadamente infla o erro tipo I (multiplas comparacoes)
3. **Consistencia** — o sistema deve ser avaliado como um todo

## Estatistica de Teste

### Ljung-Box Multivariada (Ajustada)

A estatistica implementada no chronobox e a versao multivariada da Ljung-Box:

$$Q_h = T'(T' + 2) \sum_{i=1}^{h} \frac{1}{T' - i} \text{tr}\left(\hat{\mathbf{C}}_i' \hat{\mathbf{C}}_0^{-1} \hat{\mathbf{C}}_i \hat{\mathbf{C}}_0^{-1}\right)$$

onde:

| Componente | Descricao |
|:-----------|:----------|
| $T'$ | Numero efetivo de observacoes |
| $h$ | Numero de lags testados |
| $\hat{\mathbf{C}}_i$ | Autocovariancia cruzada residual no lag $i$ ($K \times K$) |
| $\hat{\mathbf{C}}_0$ | Covariancia residual no lag 0 ($K \times K$) |
| $\text{tr}(\cdot)$ | Traco da matriz |

Sob $H_0$:

$$Q_h \overset{a}{\sim} \chi^2\left(K^2(h - p)\right)$$

onde $p$ e a ordem do VAR.

!!! warning "Escolha de $h$"
    O numero de lags $h$ deve ser **maior que $p$** (ordem do VAR), pois os graus de liberdade sao $K^2(h - p)$. Se $h \leq p$, os graus de liberdade sao nao-positivos e o teste nao e valido. Recomendacao: $h \geq 2p$ ou $h \approx 10$.

### Variantes

=== "Hosking (1980)"

    $$Q_h^{H} = T'^2 \sum_{i=1}^{h} \frac{1}{T' - i} \text{tr}\left(\hat{\mathbf{C}}_i' \hat{\mathbf{C}}_0^{-1} \hat{\mathbf{C}}_i \hat{\mathbf{C}}_0^{-1}\right)$$

    Versao original sem o fator de correcao $(T' + 2)$. Equivalente ao teste de Box-Pierce multivariado.

=== "Li-McLeod (1981)"

    $$Q_h^{LM} = T'^2 \sum_{i=1}^{h} \frac{1}{T' - i} \text{tr}\left(\hat{\mathbf{C}}_i' \hat{\mathbf{C}}_0^{-1} \hat{\mathbf{C}}_i \hat{\mathbf{C}}_0^{-1}\right) + \frac{K^2 h(h+1)}{2T'}$$

    Inclui um termo de correcao adicional para melhorar o tamanho do teste em amostras finitas.

=== "Ljung-Box Multivariada"

    $$Q_h^{LB} = T'(T' + 2) \sum_{i=1}^{h} \frac{1}{T' - i} \text{tr}\left(\hat{\mathbf{C}}_i' \hat{\mathbf{C}}_0^{-1} \hat{\mathbf{C}}_i \hat{\mathbf{C}}_0^{-1}\right)$$

    Versao implementada no chronobox. Analogia direta com o Ljung-Box univariado, com melhor tamanho em amostras finitas que Hosking.

As tres variantes sao assintoticamente equivalentes e seguem $\chi^2(K^2(h - p))$ sob $H_0$.

## Intuicao

O traco $\text{tr}(\hat{\mathbf{C}}_i' \hat{\mathbf{C}}_0^{-1} \hat{\mathbf{C}}_i \hat{\mathbf{C}}_0^{-1})$ mede a **autocorrelacao canonica quadrada** no lag $i$:

- Se os residuos sao white noise: $\hat{\mathbf{C}}_i \approx \mathbf{0}$ → traco $\approx 0$ → $Q_h$ pequeno
- Se ha autocorrelacao: $\hat{\mathbf{C}}_i \neq \mathbf{0}$ → traco > 0 → $Q_h$ grande

A soma sobre $i = 1, \ldots, h$ acumula evidencia de autocorrelacao em todos os lags ate $h$.

## Graus de Liberdade

| Parametro | Formula |
|:----------|:--------|
| Graus de liberdade | $df = K^2(h - p)$ |
| $K$ | Numero de variaveis no VAR |
| $h$ | Numero de lags testados |
| $p$ | Ordem do VAR |

Para um VAR(2) com 3 variaveis e $h = 10$:

$$df = 3^2 \times (10 - 2) = 9 \times 8 = 72$$

## Exemplo Pratico

### Residuos Sem Autocorrelacao

```python
import numpy as np
from chronobox.models.var import VAR

# VAR(2) bem especificado
np.random.seed(42)
T = 500
y = np.zeros((T, 2))
for t in range(2, T):
    y[t, 0] = 0.5 * y[t-1, 0] - 0.2 * y[t-2, 0] + np.random.randn()
    y[t, 1] = 0.3 * y[t-1, 1] + 0.1 * y[t-2, 1] + np.random.randn()

model = VAR(y, names=["X", "Y"])
results = model.fit(maxlags=2)

# Portmanteau test
wn = results.test_whiteness(nlags=10)
print(f"Q statistic: {wn['statistic']:.4f}")
print(f"p-value:     {wn['pvalue']:.4f}")
print(f"df:          {wn['df']}")
print(f"Reject H0:   {wn['reject']}")
```

Saida esperada:

```
Q statistic: 28.4567
p-value:     0.5612
df:          32
Reject H0:   False
```

!!! tip "Interpretacao"
    $Q_{10} = 28.46$ com $df = 4 \times (10 - 2) = 32$. O p-valor (0.56) e alto — nao rejeitamos $H_0$. Os residuos do VAR(2) sao consistentes com white noise multivariado. O modelo esta bem especificado.

### Residuos Com Autocorrelacao (Modelo Sub-especificado)

```python
# Gerar VAR(4) mas estimar VAR(1) — sub-especificacao
np.random.seed(42)
T = 500
y2 = np.zeros((T, 2))
for t in range(4, T):
    y2[t, 0] = 0.3 * y2[t-1, 0] + 0.2 * y2[t-2, 0] + \
               0.15 * y2[t-3, 0] + 0.1 * y2[t-4, 0] + np.random.randn()
    y2[t, 1] = 0.4 * y2[t-1, 1] - 0.1 * y2[t-3, 1] + np.random.randn()

model2 = VAR(y2, names=["X", "Y"])
results_bad = model2.fit(maxlags=1)  # Sub-especificado!

wn_bad = results_bad.test_whiteness(nlags=10)
print(f"Q = {wn_bad['statistic']:.4f}, p = {wn_bad['pvalue']:.6f}")
print(f"Reject H0: {wn_bad['reject']}")
# Esperado: rejeita H0 — residuos autocorrelacionados
```

!!! warning "O Que Fazer Se o Teste Rejeita"
    1. **Aumentar o numero de lags** do VAR — a causa mais comum e sub-especificacao
    2. **Verificar com criterios de informacao** — use [Lag Selection](../specification/lag-selection.md) para escolher $p$
    3. **Verificar estacionaridade** — series I(1) em niveis podem gerar autocorrelacao espuria
    4. **Considerar sazonalidade** — se os dados sao sazonais, incluir dummies sazonais

### Testando Diferentes Valores de $h$

```python
print(f"{'h':>4} {'Q_h':>10} {'df':>5} {'p-value':>10} {'Decision':>12}")
print("-" * 45)
for h in [5, 10, 15, 20, 30]:
    wn = results.test_whiteness(nlags=h)
    decision = "Reject" if wn['reject'] else "OK"
    print(f"{h:4d} {wn['statistic']:10.4f} {wn['df']:5d} "
          f"{wn['pvalue']:10.4f} {decision:>12}")
```

### Acessando Resultados Programaticamente

```python
wn = results.test_whiteness(nlags=10)

if wn['reject']:
    print(f"Autocorrelacao detectada (Q={wn['statistic']:.4f}, p={wn['pvalue']:.4f})")
    print("Considere aumentar o numero de lags do VAR")
else:
    print(f"Residuos OK — white noise (Q={wn['statistic']:.4f}, p={wn['pvalue']:.4f})")
    print("Modelo adequadamente especificado")
```

## Valores Criticos

| $df$ | 5% | 1% |
|:-----|:---|:---|
| 8 | 15.51 | 20.09 |
| 16 | 26.30 | 32.00 |
| 32 | 43.77 | 53.49 |
| 72 | 92.81 | 107.26 |
| 128 | 155.40 | 175.28 |

*Rejeita-se $H_0$ quando $Q_h > \chi^2_{1-\alpha}(df)$.*

## Relacao com o Ljung-Box Univariado

O teste portmanteau multivariado generaliza o [Ljung-Box](../specification/ljung-box.md):

| Aspecto | Ljung-Box Univariado | Portmanteau Multivariado |
|:--------|:--------------------|:------------------------|
| Dados | Uma serie $\{e_t\}$ | $K$ series $\{\mathbf{u}_t\}$ |
| Autocovariancia | Escalar $\hat{\rho}_i$ | Matriz $\hat{\mathbf{C}}_i$ ($K \times K$) |
| Estatistica | $Q = T(T+2)\sum \hat{\rho}_i^2/(T-i)$ | $Q_h$ com traco de matrizes |
| Graus de liberdade | $h - p$ | $K^2(h - p)$ |
| Testa | Autocorrelacao propria | Autocorrelacao propria **e cruzada** |

## Assinatura da Funcao

```python
VARResults.test_whiteness(
    nlags: int = 10      # Numero de lags h a testar
) -> dict[str, Any]
```

Retorna um dicionario com:

| Chave | Tipo | Descricao |
|:------|:-----|:----------|
| `'statistic'` | `float` | Estatistica $Q_h$ |
| `'pvalue'` | `float` | P-valor ($\chi^2$) |
| `'df'` | `int` | Graus de liberdade $K^2(h - p)$ |
| `'reject'` | `bool` | Rejeitar $H_0$ a 5%? |

## Limitacoes

1. **Assintotico** — em amostras pequenas ($T < 100$), o teste pode ter distorcao de tamanho (rejeita mais que o nominal)
2. **Sensivel a $h$** — valores de $h$ muito altos reduzem o poder; muito baixos podem nao captar autocorrelacao em lags distantes
3. **Requer $h > p$** — se $h \leq p$, os graus de liberdade sao nao-positivos
4. **Normalidade** — assume residuos com momentos finitos. Em presenca de caudas pesadas (dados financeiros), considere bootstrapped versions
5. **Nao indica qual lag** — rejeicao diz que ha autocorrelacao em algum lag ate $h$, mas nao em qual especificamente

## Equivalente R

=== "vars"

    ```r
    library(vars)

    # Estimar VAR
    var_model <- VAR(data, p = 2, type = "const")

    # Portmanteau test (tipo padrao: "PT.adjusted" = Ljung-Box multivariado)
    serial.test(var_model, lags.pt = 10, type = "PT.adjusted")

    # Alternativas:
    # type = "PT.asymptotic"   → Hosking (Box-Pierce multivariado)
    # type = "BG"              → Breusch-Godfrey multivariado (LM test)
    # type = "ES"              → Edgerton-Shukur (F-test)

    # Equivalencia com chronobox:
    # chronobox: results.test_whiteness(nlags=10)
    # R vars:    serial.test(var_model, lags.pt=10, type="PT.adjusted")
    ```

=== "MTS"

    ```r
    library(MTS)

    # Ljung-Box multivariado
    mq(residuals(var_model), lag = 10)

    # Retorna estatisticas Q para cada lag ate h
    # Mais detalhado que serial.test — mostra a evolucao lag a lag
    ```

## See Also

- [Ljung-Box](../specification/ljung-box.md) — Versao univariada do teste portmanteau
- [Eigenvalue Stability](eigenvalue.md) — Verificar estabilidade antes do portmanteau
- [Granger Causality](granger-causality.md) — Testar causalidade entre variaveis
- [VAR Stability Overview](index.md) — Visao geral dos diagnosticos VAR
- [Lag Selection](../specification/lag-selection.md) — Escolher a ordem otima do VAR

## Referencias

- Hosking, J.R.M. (1980). "The multivariate portmanteau statistic." *JASA*, 75(371), 602-608.
- Li, W.K. & McLeod, A.I. (1981). "Distribution of the residual autocorrelations in multivariate ARMA time series models." *JRSS Series B*, 43(2), 231-239.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. Section 4.4.
- Pfaff, B. (2008). "VAR, SVAR and SVEC Models: Implementation Within R Package vars." *Journal of Statistical Software*, 27(4).
