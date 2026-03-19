# Filters Theory

## Hodrick-Prescott Filter

The HP filter decomposes $y_t = \tau_t + c_t$ by minimizing:

$$
\min_{\tau} \left\{ \sum_{t=1}^{T}(y_t - \tau_t)^2 + \lambda \sum_{t=2}^{T-1}(\tau_{t+1} - 2\tau_t + \tau_{t-1})^2 \right\}
$$

The first term penalizes deviation from data, the second penalizes
curvature of the trend. $\lambda$ controls the tradeoff.

### Solution

The closed-form solution is:

$$
\hat{\tau} = (I_T + \lambda K'K)^{-1} y
$$

where $K$ is the second-difference matrix.

### Frequency Domain

The HP filter approximates a high-pass filter with gain:

$$
G(\omega) = \frac{4\lambda(1-\cos\omega)^2}{1 + 4\lambda(1-\cos\omega)^2}
$$

## Baxter-King Band-Pass Filter

The ideal band-pass filter isolates frequencies between $\underline{\omega}$
and $\bar{\omega}$:

$$
a_j = \begin{cases}
\frac{\bar{\omega} - \underline{\omega}}{\pi} & j = 0 \\
\frac{\sin(j\bar{\omega}) - \sin(j\underline{\omega})}{j\pi} & j \neq 0
\end{cases}
$$

The BK filter truncates at $K$ lags and adjusts weights so they sum to zero
(removing the zero-frequency component).

## Christiano-Fitzgerald Filter

The CF filter uses the full sample with asymmetric, time-varying weights.
Under the random walk assumption, the optimal filter weights are:

$$
\hat{b}_j = \frac{1}{2\pi} \int_{\underline{\omega}}^{\bar{\omega}} e^{ij\omega} d\omega
$$

Applied asymmetrically to use all $T$ observations.

## Hamilton Filter

Hamilton (2018) proposes a regression-based filter:

$$
y_{t+h} = \beta_0 + \beta_1 y_t + \beta_2 y_{t-1} + \cdots + \beta_p y_{t-p+1} + v_{t+h}
$$

- **Trend**: $\hat{y}_{t+h} = \hat{\beta}_0 + \hat{\beta}_1 y_t + \cdots$
- **Cycle**: $\hat{v}_{t+h} = y_{t+h} - \hat{y}_{t+h}$

### Advantages

- No spurious cyclicality
- No endpoint bias
- Standard regression inference
- No tuning parameters (just $h$ and $p$)

## Beveridge-Nelson Decomposition

For an ARIMA(p,1,q) process, the BN decomposition separates:

$$
y_t = y_t^P + y_t^T
$$

where $y_t^P$ (permanent) follows a random walk and $y_t^T$ (transitory)
is stationary. The permanent component is:

$$
y_t^P = y_t + \sum_{j=1}^{\infty} E_t[\Delta y_{t+j}]
$$

## Comparison

| Filter | Endpoint Bias | Data Loss | Spurious Cycles | Parameters |
|--------|:---:|:---:|:---:|:---:|
| HP | Yes | No | Yes | $\lambda$ |
| BK | No | Yes ($2K$) | No | $K$, band |
| CF | Mild | No | Mild | band |
| Hamilton | No | Yes ($h$) | No | $h$, $p$ |
