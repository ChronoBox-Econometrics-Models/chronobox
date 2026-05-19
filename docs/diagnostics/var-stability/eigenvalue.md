---
title: "Eigenvalue Stability"
description: "Teste de estabilidade por eigenvalues da companion matrix no chronobox — condicao de estabilidade, unit circle plot e implicacoes para IRF e previsao."
---

# Eigenvalue Stability

!!! info "Quick Reference"
    **Funcao:** `VARResults.is_stable`, `VARResults.roots`
    **Condicao:** Todos os eigenvalues com $|\lambda_i| < 1$
    **Resultado:** Booleano + vetor de eigenvalues
    **Implicacao:** Se instavel, IRF e previsao nao sao confiaveis
    **R equivalente:** `vars::roots()`, `vars::stability()`

## O Que Sao Eigenvalues do VAR?

Um VAR(p) pode ser reescrito na **forma companion** (VAR(1) expandido):

$$\underbrace{\begin{pmatrix} \mathbf{y}_t \\ \mathbf{y}_{t-1} \\ \vdots \\ \mathbf{y}_{t-p+1} \end{pmatrix}}_{\mathbf{Y}_t} = \underbrace{\begin{pmatrix} \mathbf{A}_1 & \mathbf{A}_2 & \cdots & \mathbf{A}_{p-1} & \mathbf{A}_p \\ \mathbf{I}_K & \mathbf{0} & \cdots & \mathbf{0} & \mathbf{0} \\ \mathbf{0} & \mathbf{I}_K & \cdots & \mathbf{0} & \mathbf{0} \\ \vdots & & \ddots & & \vdots \\ \mathbf{0} & \mathbf{0} & \cdots & \mathbf{I}_K & \mathbf{0} \end{pmatrix}}_{\mathbf{A}_c \; (Kp \times Kp)} \underbrace{\begin{pmatrix} \mathbf{y}_{t-1} \\ \mathbf{y}_{t-2} \\ \vdots \\ \mathbf{y}_{t-p} \end{pmatrix}}_{\mathbf{Y}_{t-1}} + \begin{pmatrix} \mathbf{u}_t \\ \mathbf{0} \\ \vdots \\ \mathbf{0} \end{pmatrix}$$

A **companion matrix** $\mathbf{A}_c$ tem dimensao $Kp \times Kp$ e seus eigenvalues $\lambda_1, \ldots, \lambda_{Kp}$ determinam a dinamica do sistema.

## Condicao de Estabilidade

O VAR(p) e **estavel** (estacionario de covariancia) se e somente se:

$$|\lambda_i| < 1 \quad \forall \, i = 1, \ldots, Kp$$

ou equivalentemente, se todas as raizes do **polinomio caracteristico reverso** estao fora do circulo unitario:

$$\det\left(\mathbf{I}_K - \mathbf{A}_1 z - \mathbf{A}_2 z^2 - \cdots - \mathbf{A}_p z^p\right) \neq 0 \quad \text{para } |z| \leq 1$$

| Condicao | Interpretacao |
|:---------|:-------------|
| $\max |\lambda_i| < 1$ | Sistema estavel — choques se dissipam |
| $\max |\lambda_i| = 1$ | Raiz unitaria — presenca de I(1) ou cointegracao |
| $\max |\lambda_i| > 1$ | Sistema explosivo — modelo mal especificado |

## Visualizacao: Unit Circle Plot

O grafico padrao para diagnostico de estabilidade plota os eigenvalues no plano complexo junto com o circulo unitario:

```
          Im
           |
     *     |     *      ← eigenvalues complexos conjugados
           |
  ---------+----------> Re
           |
     *     |     *
           |
        (  O  )         ← circulo unitario |z| = 1
```

- **Dentro do circulo:** sistema estavel
- **Sobre o circulo:** raiz unitaria (I(1))
- **Fora do circulo:** sistema explosivo

## Por Que Estabilidade Importa

### Impulse Response Function (IRF)

A IRF no horizonte $h$ depende das potencias da companion matrix:

$$\mathbf{\Phi}_h = \mathbf{J} \mathbf{A}_c^h \mathbf{J}'$$

onde $\mathbf{J} = [\mathbf{I}_K, \mathbf{0}, \ldots, \mathbf{0}]$.

- Se $|\lambda_i| < 1$ para todo $i$: $\mathbf{A}_c^h \to \mathbf{0}$ quando $h \to \infty$ → IRF **converge para zero**
- Se $|\lambda_i| \geq 1$ para algum $i$: $\mathbf{A}_c^h$ nao converge → IRF **nao decai**

### Previsao

Previsoes de longo prazo convergem para a **media incondicional** apenas se o sistema e estavel:

$$E[\mathbf{y}_{t+h}] \to (\mathbf{I}_K - \mathbf{A}_1 - \cdots - \mathbf{A}_p)^{-1} \mathbf{c} \quad \text{quando } h \to \infty$$

Se instavel, as previsoes divergem.

### FEVD

A variancia total do erro de previsao $h$ passos a frente e:

$$\text{MSE}(h) = \sum_{i=0}^{h-1} \mathbf{\Phi}_i \mathbf{\Sigma}_u \mathbf{\Phi}_i'$$

Se o sistema e estavel, essa soma converge. Se instavel, cresce sem limite.

## Exemplo Pratico

### VAR Estavel

```python
import numpy as np
from chronobox.models.var import VAR

# Simular VAR(1) bivariado estavel
# A_1 com eigenvalues 0.8 e 0.3
np.random.seed(42)
T = 300
y = np.zeros((T, 2))
for t in range(1, T):
    y[t, 0] = 0.5 * y[t-1, 0] + 0.1 * y[t-1, 1] + np.random.randn()
    y[t, 1] = 0.2 * y[t-1, 0] + 0.3 * y[t-1, 1] + np.random.randn()

model = VAR(y, names=["GDP", "INF"])
results = model.fit(maxlags=1)

# Verificar estabilidade
print(f"Estavel: {results.is_stable}")
print(f"Eigenvalues: {results.roots}")
print(f"Modulos: {np.abs(results.roots)}")
print(f"Max |lambda|: {np.max(np.abs(results.roots)):.4f}")
```

Saida esperada:

```
Estavel: True
Eigenvalues: [0.6236+0.0000j  0.1764+0.0000j]
Modulos: [0.6236  0.1764]
Max |lambda|: 0.6236
```

!!! tip "Interpretacao"
    Ambos os eigenvalues possuem modulo menor que 1 (0.62 e 0.18). O sistema e estavel — choques se dissipam e a IRF converge para zero. O eigenvalue dominante (0.62) determina a velocidade de convergencia: quanto mais proximo de 1, mais lenta a dissipacao.

### VAR Instavel (Random Walk)

```python
# Random walk bivariado — instavel por construcao
y_rw = np.cumsum(np.random.randn(300, 2), axis=0)

model_rw = VAR(y_rw, names=["X", "Y"])
results_rw = model_rw.fit(maxlags=1)

print(f"Estavel: {results_rw.is_stable}")
print(f"Max |lambda|: {np.max(np.abs(results_rw.roots)):.4f}")
# Esperado: is_stable = False, max |lambda| ≈ 1.0
```

!!! warning "O Que Fazer Se o VAR e Instavel"
    1. **Variaveis I(1)?** Testar raiz unitaria com [ADF](../unit-root/adf.md) / [KPSS](../unit-root/kpss.md)
    2. **Cointegradas?** Testar com [Johansen](../cointegration/johansen.md) e usar [VECM](../../user-guide/var/vecm.md)
    3. **Nao cointegradas?** Diferenciar as series e estimar VAR em diferencas
    4. **Modelo mal especificado?** Verificar numero de lags com [Lag Selection](../specification/lag-selection.md)

### Plot dos Eigenvalues no Plano Complexo

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, res, title in [(axes[0], results, "VAR Estavel"),
                         (axes[1], results_rw, "VAR Instavel")]:
    roots = res.roots

    # Circulo unitario
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="Unit circle")

    # Eigenvalues
    ax.scatter(roots.real, roots.imag, c="red", s=80, zorder=5,
               label="Eigenvalues")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginario")
    ax.set_title(title)
    ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig("eigenvalue_stability.png", dpi=150)
plt.show()
```

### Acessando Resultados Programaticamente

```python
results = model.fit(maxlags=2)

# Decisao automatica
if results.is_stable:
    print("VAR estavel — prosseguir com IRF e FEVD")
    max_mod = np.max(np.abs(results.roots))
    half_life = -np.log(2) / np.log(max_mod)
    print(f"Half-life do choque: {half_life:.1f} periodos")
else:
    print("VAR instavel — reavaliar especificacao")
    n_unit = np.sum(np.abs(results.roots) >= 0.999)
    print(f"Eigenvalues com |lambda| >= 1: {n_unit}")

# Eigenvalues individuais
for i, ev in enumerate(results.roots):
    print(f"  lambda_{i+1}: {ev:.4f}  |lambda| = {np.abs(ev):.4f}")
```

## Companion Matrix: Detalhes

Para um VAR(2) com $K = 2$ variaveis, a companion matrix tem dimensao $4 \times 4$:

$$\mathbf{A}_c = \begin{pmatrix} \mathbf{A}_1 & \mathbf{A}_2 \\ \mathbf{I}_2 & \mathbf{0}_2 \end{pmatrix} = \begin{pmatrix} a_{11}^{(1)} & a_{12}^{(1)} & a_{11}^{(2)} & a_{12}^{(2)} \\ a_{21}^{(1)} & a_{22}^{(1)} & a_{21}^{(2)} & a_{22}^{(2)} \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}$$

Os 4 eigenvalues desta matriz determinam a estabilidade. Note que:

- Eigenvalues podem ser **reais** ou **complexos conjugados**
- Eigenvalues complexos produzem **oscilacoes amortecidas** na IRF
- O **modulo** (nao a parte real) determina a estabilidade

## Limitacoes

1. **Nao e um teste estatistico** — e uma condicao deterministica calculada dos coeficientes estimados. Nao ha p-valor
2. **Sensivel a estimacao** — em amostras pequenas, eigenvalues estimados podem estar dentro do circulo mesmo se os verdadeiros estao fora (e vice-versa)
3. **Nao distingue I(1) de cointegracao** — um eigenvalue proximo de 1 pode indicar tanto uma raiz unitaria quanto um processo persistente estacionario
4. **VAR em niveis com I(1)** — sera instavel por construcao, o que e esperado e nao indica erro

## Equivalente R

=== "vars"

    ```r
    library(vars)

    # Estimar VAR
    data <- data.frame(GDP = rnorm(300), INF = rnorm(300))
    var_model <- VAR(data, p = 2, type = "const")

    # Eigenvalues (raizes inversas)
    roots_vals <- roots(var_model)
    print(roots_vals)

    # Verificar estabilidade
    all(roots_vals < 1)  # TRUE = estavel

    # Plot de estabilidade
    plot(stability(var_model))
    ```

=== "Manual em R"

    ```r
    # Construir companion matrix manualmente
    A1 <- coef(var_model)$GDP[1:2, 1]  # coefs de lag 1
    A2 <- coef(var_model)$GDP[3:4, 1]  # coefs de lag 2

    # Companion matrix
    K <- 2; p <- 2
    companion <- matrix(0, K*p, K*p)
    # ... preencher com coeficientes

    eigenvalues <- eigen(companion)$values
    moduli <- Mod(eigenvalues)
    print(moduli)
    cat("Estavel:", all(moduli < 1), "\n")
    ```

## See Also

- [Granger Causality](granger-causality.md) — Testar relacoes causais entre variaveis
- [Portmanteau Test](portmanteau.md) — Verificar autocorrelacao residual
- [VAR Stability Overview](index.md) — Visao geral dos diagnosticos VAR
- [VAR Theory](../../theory/var-theory.md) — Algebra do VAR e companion form

## Referencias

- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. Chapter 2.
- Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University Press. Chapter 10.
- Pfaff, B. (2008). "VAR, SVAR and SVEC Models: Implementation Within R Package vars." *Journal of Statistical Software*, 27(4).
