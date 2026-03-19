# VAR/SVAR Theory

## VAR Model

A VAR(p) model for $K$ variables:

$$
y_t = c + A_1 y_{t-1} + A_2 y_{t-2} + \cdots + A_p y_{t-p} + u_t
$$

where $y_t$ is a $K \times 1$ vector, $A_i$ are $K \times K$ coefficient matrices,
and $u_t \sim N(0, \Sigma_u)$.

## Estimation

VAR is estimated by OLS (equation by equation), which is efficient when
all equations have the same regressors.

## Stability

The VAR(p) is stable if all eigenvalues of the companion matrix lie
inside the unit circle:

$$
\det(I_K - A_1 z - A_2 z^2 - \cdots - A_p z^p) \neq 0 \quad \text{for } |z| \leq 1
$$

## Lag Selection

Select lag order $p$ by minimizing information criteria:

$$
AIC(p) = \ln|\hat{\Sigma}_u(p)| + \frac{2pK^2}{T}
$$

$$
BIC(p) = \ln|\hat{\Sigma}_u(p)| + \frac{pK^2 \ln T}{T}
$$

## Granger Causality

Variable $x$ Granger-causes $y$ if past values of $x$ help predict $y$
beyond past values of $y$ alone. Tested via F-test on coefficient restrictions.

## Impulse Response Functions

The VMA($\infty$) representation:

$$
y_t = \mu + \sum_{i=0}^{\infty} \Phi_i u_{t-i}
$$

where $\Phi_0 = I_K$ and $\Phi_i = \sum_{j=1}^{i} \Phi_{i-j} A_j$.

## FEVD

Forecast error variance decomposition measures how much of the $h$-step
forecast error variance of variable $i$ is explained by shocks to variable $j$.

## Structural VAR

The structural form:

$$
B_0 y_t = c^* + B_1 y_{t-1} + \cdots + B_p y_{t-p} + \varepsilon_t
$$

where $\varepsilon_t \sim N(0, I_K)$ are structural shocks.

The reduced form errors relate to structural shocks via:

$$
u_t = B_0^{-1} \varepsilon_t
$$

### Identification

- **Cholesky**: $B_0^{-1} = P$ where $\Sigma_u = PP'$ (lower triangular)
- **Short-run**: Zero restrictions on $B_0^{-1}$
- **Long-run** (Blanchard-Quah): Restrictions on long-run multiplier $C(1) = (I - A_1 - \cdots - A_p)^{-1} B_0^{-1}$

## Structural IRF

$$
y_t = \mu + \sum_{i=0}^{\infty} \Theta_i \varepsilon_{t-i}
$$

where $\Theta_i = \Phi_i B_0^{-1}$ gives the structural impulse responses.
