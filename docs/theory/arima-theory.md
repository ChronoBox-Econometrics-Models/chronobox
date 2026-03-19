# ARIMA Theory

## The ARMA Model

An ARMA(p,q) process is defined as:

$$
y_t = c + \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}
$$

where $\varepsilon_t \sim WN(0, \sigma^2)$.

Using the backshift operator $B$ (where $B y_t = y_{t-1}$):

$$
\phi(B) y_t = c + \theta(B) \varepsilon_t
$$

with $\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$ and $\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$.

## Stationarity and Invertibility

- **Stationarity**: All roots of $\phi(z) = 0$ lie outside the unit circle
- **Invertibility**: All roots of $\theta(z) = 0$ lie outside the unit circle

## ARIMA via Differencing

An ARIMA(p,d,q) model applies ARMA(p,q) to the d-th difference:

$$
\Delta^d y_t = (1-B)^d y_t
$$

So:

$$
\phi(B)(1-B)^d y_t = c + \theta(B) \varepsilon_t
$$

## Seasonal ARIMA

SARIMA extends ARIMA with seasonal components:

$$
\Phi_P(B^s) \phi_p(B) \Delta^d \Delta_s^D y_t = \Theta_Q(B^s) \theta_q(B) \varepsilon_t
$$

where:

- $\Phi_P(B^s) = 1 - \Phi_1 B^s - \cdots - \Phi_P B^{Ps}$
- $\Theta_Q(B^s) = 1 + \Theta_1 B^s + \cdots + \Theta_Q B^{Qs}$
- $\Delta_s = 1 - B^s$ (seasonal difference)

## State-Space Representation

ChronoBox casts ARIMA into state-space form for exact MLE:

$$
\begin{aligned}
\alpha_{t+1} &= T \alpha_t + R \eta_t, \quad \eta_t \sim N(0, Q) \\
y_t &= Z \alpha_t + \varepsilon_t
\end{aligned}
$$

The Kalman filter (via `kalmanbox`) recursively computes the log-likelihood.

## Estimation

ChronoBox uses Maximum Likelihood Estimation via the Kalman filter
(provided by kalmanbox). The exact log-likelihood is:

$$
\ln L = -\frac{T}{2} \ln(2\pi) - \frac{1}{2} \sum_{t=1}^{T} \left[ \ln |F_t| + v_t' F_t^{-1} v_t \right]
$$

where $v_t$ are the prediction errors and $F_t$ their variances from the Kalman filter.

## Information Criteria

$$
AIC = -2 \ln L + 2k
$$

$$
BIC = -2 \ln L + k \ln T
$$

$$
AICc = AIC + \frac{2k(k+1)}{T - k - 1}
$$

$$
HQIC = -2 \ln L + 2k \ln(\ln T)
$$

where $k$ is the number of parameters and $T$ is the sample size.

## Box-Jenkins Methodology

1. **Identification**: Use ACF/PACF to suggest orders
2. **Estimation**: Fit model via MLE
3. **Diagnostics**: Check residuals (Ljung-Box, normality)
4. **Forecasting**: Generate predictions with confidence intervals
