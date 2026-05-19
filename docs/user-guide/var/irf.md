---
title: Impulse Response Functions (IRF)
description: Funcoes impulso-resposta --- Cholesky, generalizada, bootstrap e IRF acumulada.
---

# Impulse Response Functions (IRF)

!!! info "Quick Reference"
    - **Metodo**: `results.irf(steps, method, orth)`
    - **Import**: `from chronobox import VAR` (ou `VECM`)
    - **R equivalente**: `vars::irf()`
    - **Tipos**: Ortogonal (Cholesky), Generalizada (Pesaran-Shin)

---

## Overview

As funcoes impulso-resposta (IRF) medem **como o sistema reage a um choque**
em uma das variaveis ao longo do tempo. Dado um choque unitario (ou de um
desvio-padrao) na variavel $j$ no periodo $t$, a IRF mostra o efeito sobre
a variavel $i$ nos periodos $t, t+1, t+2, \ldots, t+h$.

A IRF e a ferramenta mais utilizada para interpretar modelos VAR, pois
transforma os coeficientes matriciais --- dificeis de interpretar diretamente
--- em trajetorias temporais intuitivas.

### Quando usar

- Analisar transmissao de choques (ex: choque monetario sobre PIB)
- Comparar velocidade e magnitude de reacao entre variaveis
- Verificar se o sistema e estavel (respostas convergem a zero)
- Comunicar resultados de VARs para audiencia nao tecnica

---

## Formulacao Matematica

### Representacao VMA($\infty$)

Um VAR(p) estavel pode ser reescrito como um processo de medias moveis vetorial:

$$
\mathbf{y}_t = \boldsymbol{\mu} + \sum_{h=0}^{\infty} \boldsymbol{\Phi}_h \mathbf{u}_{t-h}
$$

onde as matrizes $\boldsymbol{\Phi}_h$ sao os **multiplicadores de impacto**:

$$
\boldsymbol{\Phi}_0 = \mathbf{I}_K, \qquad
\boldsymbol{\Phi}_h = \sum_{j=1}^{h} \boldsymbol{\Phi}_{h-j} \mathbf{A}_j
$$

O elemento $(\boldsymbol{\Phi}_h)_{ij}$ e a resposta da variavel $i$ no
horizonte $h$ a um choque unitario na variavel $j$.

### IRF Ortogonal (Cholesky)

Os choques $\mathbf{u}_t$ sao correlacionados ($\boldsymbol{\Sigma}_u$ nao diagonal).
Para obter choques **ortogonais** (nao correlacionados), usa-se a decomposicao
de Cholesky:

$$
\boldsymbol{\Sigma}_u = \mathbf{P}\mathbf{P}'
$$

onde $\mathbf{P}$ e triangular inferior. Os choques ortogonalizados sao
$\boldsymbol{\varepsilon}_t = \mathbf{P}^{-1}\mathbf{u}_t$, com
$\text{Cov}(\boldsymbol{\varepsilon}_t) = \mathbf{I}_K$.

A IRF ortogonal e:

$$
\boldsymbol{\Psi}_h = \boldsymbol{\Phi}_h \mathbf{P}
$$

O elemento $(\boldsymbol{\Psi}_h)_{ij}$ e a resposta da variavel $i$ no
horizonte $h$ a um choque de um desvio-padrao na variavel $j$.

!!! warning "Dependencia da ordenacao"
    A decomposicao de Cholesky **depende da ordenacao das variaveis**. A
    primeira variavel na ordenacao pode afetar todas as outras
    contemporaneamente, mas nao e afetada por nenhuma. A ultima variavel e
    afetada por todas contemporaneamente. Escolha a ordenacao com base em
    teoria economica.

### IRF Generalizada (Pesaran-Shin)

A IRF generalizada (GIRF) de Pesaran & Shin (1998) elimina a dependencia da
ordenacao. O choque e definido como a esperanca condicional:

$$
\text{GIRF}_i(h) = E[\mathbf{y}_{t+h} | u_{it} = \delta_i] - E[\mathbf{y}_{t+h}]
$$

A formula e:

$$
\boldsymbol{\Psi}_h^g = \sigma_{jj}^{-1/2} \boldsymbol{\Phi}_h \boldsymbol{\Sigma}_u \mathbf{e}_j
$$

onde:

- $\sigma_{jj}$ e o $(j,j)$-esimo elemento de $\boldsymbol{\Sigma}_u$
- $\mathbf{e}_j$ e o $j$-esimo vetor unitario

!!! tip "Quando usar qual?"
    - **IRF Ortogonal (Cholesky)**: quando voce tem uma teoria sobre a
      ordenacao causal contemporanea das variaveis
    - **IRF Generalizada**: quando nao ha base teorica para impor uma ordenacao,
      ou para verificar robustez

### IRF Acumulada

A IRF acumulada mede o **efeito total** do choque ate o horizonte $h$:

$$
\boldsymbol{\Psi}_h^{\text{cum}} = \sum_{s=0}^{h} \boldsymbol{\Psi}_s
$$

Util quando o interesse e no efeito permanente (de longo prazo) do choque,
especialmente para variaveis em diferencas.

---

## Quick Example

```python
from chronobox import VAR
from chronobox.datasets import load_macro

# Ajustar VAR
data = load_macro()
model = VAR(lags=2)
results = model.fit(data)

# IRF ortogonal (Cholesky), 20 periodos
irf = results.irf(steps=20, method='cholesky')

# IRF generalizada (Pesaran-Shin)
girf = results.irf(steps=20, method='generalized')

# Com intervalos de confianca (bootstrap)
irf_ci = results.irf(steps=20, method='cholesky', ci=0.95, n_boot=1000)

# Plotar
irf.plot()
```

---

## Guia Detalhado

### Parametros da IRF

```python
irf = results.irf(
    steps=20,            # Horizonte
    method='cholesky',   # 'cholesky' ou 'generalized'
    orth=True,           # Ortogonalizar (True para Cholesky)
    cumulative=False,    # Acumular respostas
    ci=0.95,             # Nivel de confianca (None para sem IC)
    n_boot=1000          # Numero de replicacoes bootstrap
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `steps` | `int` | `20` | Horizonte da IRF |
| `method` | `str` | `'cholesky'` | `'cholesky'` ou `'generalized'` |
| `orth` | `bool` | `True` | Ortogonalizar choques |
| `cumulative` | `bool` | `False` | Acumular respostas |
| `ci` | `float \| None` | `None` | Nivel de confianca para bootstrap |
| `n_boot` | `int` | `1000` | Replicacoes bootstrap |

### Exemplo: Choque Monetario

Considere um VAR com PIB ($y$), inflacao ($\pi$) e taxa de juros ($i$),
ordenados como $[\text{gdp}, \text{infl}, \text{rate}]$.

```python
from chronobox import VAR
from chronobox.datasets import load_macro

data = load_macro()  # gdp, infl, rate

# Ordenacao: gdp → infl → rate
# (gdp nao reage contemporaneamente a juros)
model = VAR(lags=4)
results = model.fit(data[["gdp", "infl", "rate"]])

# IRF do choque monetario (choque em 'rate')
irf = results.irf(steps=24, method='cholesky', ci=0.95, n_boot=1000)

# Plotar resposta do GDP a choque nos juros
irf.plot(impulse='rate', response='gdp')
```

A resposta esperada:

- **GDP**: queda temporaria apos aumento dos juros (efeito contracionista)
- **Inflacao**: queda defasada (price puzzle se subir primeiro)
- **Juros**: pico no impacto, decai gradualmente

### Bootstrap para Intervalos de Confianca

Os intervalos de confianca sao obtidos via bootstrap:

1. Re-amostrar os residuos $\hat{\mathbf{u}}_t$ com reposicao
2. Reconstruir as series a partir dos residuos bootstrap
3. Re-estimar o VAR e calcular a IRF
4. Repetir $B$ vezes e extrair os percentis

```python
# IRF com IC via bootstrap
irf_ci = results.irf(
    steps=20,
    method='cholesky',
    ci=0.95,
    n_boot=2000  # Mais replicacoes = IC mais preciso
)

# Acessar valores
print(irf_ci.irfs)       # Respostas pontuais (array 3D: steps x K x K)
print(irf_ci.lower)      # Limite inferior do IC
print(irf_ci.upper)      # Limite superior do IC
```

### Acessing IRF Values

```python
# Resposta de 'gdp' a choque em 'rate' no horizonte h=5
response = irf.irfs[5, 0, 2]  # [horizonte, var_resposta, var_choque]
print(f"Resposta do GDP no h=5: {response:.4f}")

# Todas as respostas como DataFrame
df_irf = irf.to_dataframe(impulse='rate', response='gdp')
print(df_irf)
```

```text
   step  response     lower     upper
0     0   0.0000   -0.0012    0.0015
1     1  -0.0023   -0.0058    0.0011
2     2  -0.0045   -0.0089   -0.0003
3     3  -0.0052   -0.0098   -0.0009
4     4  -0.0048   -0.0091   -0.0005
...
```

### IRF Acumulada

```python
# IRF acumulada: efeito total do choque
irf_cum = results.irf(steps=20, method='cholesky', cumulative=True)
irf_cum.plot(impulse='rate', response='gdp')
```

!!! tip "Variaveis em diferencas"
    Se o VAR e estimado em diferencas, a IRF acumulada recupera o efeito
    sobre o **nivel** da variavel. Por exemplo, se $\Delta \text{gdp}$ esta
    no VAR, a IRF acumulada mostra o efeito sobre $\text{gdp}$.

### Plotando a IRF

```python
# Plot completo: todas as combinacoes impulso-resposta
irf.plot()

# Plot seletivo: respostas a um choque especifico
irf.plot(impulse='rate')

# Plot de uma unica relacao
irf.plot(impulse='rate', response='gdp')
```

---

## Diagnosticos

### Verificar Convergencia

Uma IRF de um VAR estavel deve convergir a zero:

```python
# Verificar se a IRF converge
final_responses = irf.irfs[-1, :, :]  # Ultimo horizonte
print("Respostas no horizonte final:")
print(final_responses)
# Todos os valores devem ser proximos de zero
```

### Sensibilidade a Ordenacao

Compare a IRF Cholesky com diferentes ordenacoes e com a GIRF:

```python
# Ordenacao 1: gdp → infl → rate
results1 = model.fit(data[["gdp", "infl", "rate"]])
irf1 = results1.irf(steps=20, method='cholesky')

# Ordenacao 2: rate → infl → gdp
results2 = model.fit(data[["rate", "infl", "gdp"]])
irf2 = results2.irf(steps=20, method='cholesky')

# Generalizada (invariante a ordenacao)
girf = results1.irf(steps=20, method='generalized')
```

Se os resultados variam muito com a ordenacao, considere usar a IRF
generalizada ou impor restricoes estruturais (SVAR).

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import VAR

    model = VAR(lags=2)
    results = model.fit(data)

    # IRF ortogonal
    irf = results.irf(steps=20, method='cholesky', ci=0.95, n_boot=1000)
    irf.plot(impulse='rate', response='gdp')

    # IRF acumulada
    irf_cum = results.irf(steps=20, cumulative=True)

    # IRF generalizada
    girf = results.irf(steps=20, method='generalized')
    ```

=== "vars (R)"

    ```r
    library(vars)

    fit <- VAR(y, p = 2, type = "const")

    # IRF ortogonal
    ir <- irf(fit, impulse = "rate", response = "gdp",
              n.ahead = 20, ortho = TRUE, boot = TRUE,
              ci = 0.95, runs = 1000)
    plot(ir)

    # IRF acumulada
    ir_cum <- irf(fit, impulse = "rate", response = "gdp",
                  n.ahead = 20, cumulative = TRUE)

    # IRF generalizada (via rmgarch ou manual)
    ```

**Mapeamento de parametros**:

| chronobox | vars (R) | Descricao |
|---|---|---|
| `steps=20` | `n.ahead=20` | Horizonte |
| `method='cholesky'` | `ortho=TRUE` | IRF ortogonal |
| `ci=0.95` | `ci=0.95` | Nivel de confianca |
| `n_boot=1000` | `runs=1000` | Replicacoes bootstrap |
| `cumulative=True` | `cumulative=TRUE` | Acumular respostas |
| `irf.plot()` | `plot(ir)` | Plotar |

---

## Referencias

- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
  Capitulo 2.3 (IRF) e 3.7 (Bootstrap).
- Pesaran, M. H. & Shin, Y. (1998). Generalized Impulse Response Analysis in
  Linear Multivariate Models. *Economics Letters*, 58(1), 17--29.
- Sims, C. A. (1980). Macroeconomics and Reality. *Econometrica*, 48(1), 1--48.
- Kilian, L. & Lutkepohl, H. (2017). *Structural Vector Autoregressive Analysis*.
  Cambridge University Press.
