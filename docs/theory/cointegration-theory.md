# Cointegration Theory

## Definition

A set of $K$ I(1) variables are cointegrated if there exists a linear
combination that is I(0):

$$
\beta' y_t \sim I(0) \quad \text{where } y_t \sim I(1)
$$

The vector $\beta$ is called the cointegrating vector.

## Granger Representation Theorem

If $y_t$ is cointegrated with rank $r$, there exists a VECM representation:

$$
\Delta y_t = \alpha \beta' y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta y_{t-i} + u_t
$$

where:

- $\alpha$: $K \times r$ adjustment (loading) matrix
- $\beta$: $K \times r$ cointegrating matrix
- $\Pi = \alpha \beta'$: long-run impact matrix with rank $r$

## Johansen Test

Tests for the number of cointegrating relations using two statistics:

### Trace Test

$$
\lambda_{trace}(r) = -T \sum_{i=r+1}^{K} \ln(1 - \hat{\lambda}_i)
$$

$H_0$: rank $\leq r$ vs $H_1$: rank $> r$

### Maximum Eigenvalue Test

$$
\lambda_{max}(r, r+1) = -T \ln(1 - \hat{\lambda}_{r+1})
$$

$H_0$: rank $= r$ vs $H_1$: rank $= r+1$

## Engle-Granger Two-Step

1. Estimate the long-run regression: $y_t = \alpha + \beta x_t + e_t$
2. Test residuals $\hat{e}_t$ for stationarity (ADF with MacKinnon critical values)

## VECM Estimation

The VECM is estimated using the Johansen ML procedure:

1. Regress $\Delta y_t$ and $y_{t-1}$ on lagged differences
2. Form residual moment matrices
3. Solve the eigenvalue problem
4. Estimate $\hat{\beta}$ from eigenvectors, $\hat{\alpha}$ from regression

## Deterministic Terms

Five cases for deterministic terms (Johansen, 1995):

1. No intercept, no trend
2. Restricted intercept, no trend
3. Unrestricted intercept, no trend
4. Unrestricted intercept, restricted trend
5. Unrestricted intercept, unrestricted trend

## ARDL Bounds Test

The Pesaran-Shin-Smith bounds test works regardless of integration order:

$$
\Delta y_t = c + \pi_y y_{t-1} + \pi_x x_{t-1} + \sum_{i=1}^{p-1} \gamma_i \Delta y_{t-i} + \sum_{j=0}^{q-1} \delta_j \Delta x_{t-j} + \varepsilon_t
$$

Test $H_0: \pi_y = \pi_x = 0$ using F-statistic with non-standard bounds:

- F > upper bound $\Rightarrow$ cointegration
- F < lower bound $\Rightarrow$ no cointegration
- Between bounds $\Rightarrow$ inconclusive
