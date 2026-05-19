"""
Data generators for SVAR/BVAR examples.

Provides functions to generate structural VAR processes with various
identification schemes: short-run (Cholesky, AB model), long-run
(Blanchard-Quah), and sign restrictions.
"""

import numpy as np


def generate_svar_process(
    n: int = 200,
    B0_inv: np.ndarray | None = None,
    A_matrices: list[np.ndarray] | None = None,
    sigma: np.ndarray | None = None,
    seed: int = 42,
    const: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a Structural VAR process.

    The structural form is:
        B_0 y_t = const + B_1 y_{t-1} + ... + e_t

    where e_t ~ N(0, I).  The reduced form is:
        y_t = B_0^{-1} const + B_0^{-1} B_1 y_{t-1} + ... + u_t

    where u_t = B_0^{-1} e_t and Sigma_u = B_0^{-1} (B_0^{-1})'.

    Parameters
    ----------
    n : int
        Number of observations to generate.
    B0_inv : np.ndarray or None
        Inverse of the structural impact matrix (k x k). Maps structural
        shocks to reduced-form innovations: u_t = B0_inv @ e_t.
        If None, uses a lower-triangular matrix (Cholesky-type).
    A_matrices : list of np.ndarray or None
        Reduced-form VAR coefficient matrices [A_1, A_2, ...], each (k x k).
        Convention: y_t = const + A_1 y_{t-1} + A_2 y_{t-2} + ... + u_t.
        If None, uses a default stationary bivariate VAR(1).
    sigma : np.ndarray or None
        Diagonal covariance of structural shocks (k x k). Typically identity.
        If None, uses identity matrix.
    seed : int
        Random seed for reproducibility.
    const : np.ndarray or None
        Reduced-form constant vector (k,). If None, uses zeros.

    Returns
    -------
    y : np.ndarray
        Generated process in levels, shape (n, k).
    structural_shocks : np.ndarray
        Underlying structural shocks, shape (n, k).
    """
    rng = np.random.default_rng(seed)

    if A_matrices is None:
        A_matrices = [np.array([[0.5, 0.1], [0.2, 0.3]])]

    k = A_matrices[0].shape[0]
    p = len(A_matrices)

    if B0_inv is None:
        # Lower-triangular (Cholesky-type identification)
        B0_inv = np.array([[1.0, 0.0], [0.4, 0.8]]) if k == 2 else np.eye(k)

    if sigma is None:
        sigma = np.eye(k)

    if const is None:
        const = np.zeros(k)

    burn = max(200, 20 * p)
    total = n + burn

    # Structural shocks: e_t ~ N(0, sigma)
    structural_shocks = rng.multivariate_normal(np.zeros(k), sigma, total)

    # Reduced-form innovations: u_t = B0_inv @ e_t
    innovations = (B0_inv @ structural_shocks.T).T

    y = np.zeros((total, k))

    for t in range(total):
        y[t] = const + innovations[t]
        for lag_idx, A in enumerate(A_matrices):
            if t - lag_idx - 1 >= 0:
                y[t] += A @ y[t - lag_idx - 1]

    return y[burn:], structural_shocks[burn:]


def generate_demand_supply(
    n: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate a demand-supply system for structural identification.

    Two-equation system:
      - Supply: q_t^s = a_s p_t + supply_shock_t   (positive slope)
      - Demand: q_t^d = a_d p_t + demand_shock_t   (negative slope)

    Equilibrium: q_t^s = q_t^d => p and q are jointly determined.

    The structural shocks are independent, but price and quantity are
    correlated in the reduced form, requiring structural identification.

    Short-run restriction: supply does not respond contemporaneously to
    demand shocks (zero restriction on B0_inv).

    Parameters
    ----------
    n : int
        Number of observations to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data : np.ndarray
        Array of shape (n, 2), columns are [price, quantity].
    params : dict
        Dictionary with true structural parameters for validation.
    """
    rng = np.random.default_rng(seed)

    # Structural parameters
    a_s = 0.6    # supply elasticity (positive)
    a_d = -0.8   # demand elasticity (negative)

    # VAR dynamics (persistence)
    A1 = np.array([
        [0.4, 0.1],   # price depends on own lag and quantity lag
        [0.05, 0.3],   # quantity depends on price lag and own lag
    ])

    # Structural impact matrix B0_inv
    # Supply doesn't respond contemporaneously to demand shocks
    # [price]    = B0_inv @ [supply_shock]
    # [quantity]           [demand_shock]
    det = a_s - a_d  # = 1.4
    B0_inv = np.array([
        [1.0 / det, -1.0 / det],
        [a_s / det, -a_d / det],
    ])

    burn = 200
    total = n + burn

    sigma_structural = np.eye(2)
    e = rng.multivariate_normal(np.zeros(2), sigma_structural, total)
    u = (B0_inv @ e.T).T

    y = np.zeros((total, 2))
    for t in range(1, total):
        y[t] = A1 @ y[t - 1] + u[t]

    params = {
        "a_s": a_s,
        "a_d": a_d,
        "B0_inv": B0_inv,
        "A1": A1,
        "Sigma_u": B0_inv @ B0_inv.T,
        "variable_names": ["price", "quantity"],
    }

    return y[burn:], params


def generate_monetary_policy(
    n: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate a 3-variable monetary policy system.

    Three-equation model:
      - Output gap (y): responds to monetary policy shocks with a lag
      - Inflation (pi): responds to output gap and own dynamics
      - Interest rate (i): Taylor-rule-like, responds to inflation and output

    Identification via Cholesky ordering: [y, pi, i].
    The ordering assumes output and inflation are predetermined within the
    quarter; monetary policy can respond contemporaneously to both.

    This is the classic Christiano-Eichenbaum-Evans style identification.

    Parameters
    ----------
    n : int
        Number of observations to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data : np.ndarray
        Array of shape (n, 3), columns are [output_gap, inflation, interest_rate].
    params : dict
        Dictionary with true structural parameters.
    """
    rng = np.random.default_rng(seed)

    # Reduced-form VAR(2) for [output_gap, inflation, interest_rate]
    A1 = np.array([
        [0.50, -0.05, -0.10],   # output: persistence, mild Phillips, rate effect
        [0.10,  0.55,  0.03],   # inflation: output gap pass-through, persistence
        [0.15,  0.20,  0.60],   # rate: Taylor rule response, persistence
    ])

    A2 = np.array([
        [0.10,  0.02, -0.05],
        [0.05,  0.10,  0.01],
        [0.05,  0.08,  0.10],
    ])

    # Structural impact matrix (lower triangular = Cholesky)
    # Output doesn't respond contemporaneously to pi or i shocks
    # Inflation doesn't respond contemporaneously to i shocks
    # Interest rate responds contemporaneously to both
    B0_inv = np.array([
        [1.0, 0.0, 0.0],
        [0.3, 0.8, 0.0],
        [0.2, 0.3, 0.6],
    ])

    const = np.array([0.3, 0.2, 0.1])

    burn = 300
    total = n + burn

    e = rng.multivariate_normal(np.zeros(3), np.eye(3), total)
    u = (B0_inv @ e.T).T

    y = np.zeros((total, 3))
    for t in range(total):
        y[t] = const + u[t]
        if t >= 1:
            y[t] += A1 @ y[t - 1]
        if t >= 2:
            y[t] += A2 @ y[t - 2]

    params = {
        "B0_inv": B0_inv,
        "A1": A1,
        "A2": A2,
        "const": const,
        "Sigma_u": B0_inv @ B0_inv.T,
        "variable_names": ["output_gap", "inflation", "interest_rate"],
    }

    return y[burn:], params


def generate_blanchard_quah(
    n: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate a bivariate system for Blanchard-Quah long-run identification.

    Two structural shocks:
      - Supply shock: has permanent (long-run) effect on output
      - Demand shock: has only transitory effect on output (zero long-run effect)

    Both shocks can affect unemployment in the long run.
    This is the classic Blanchard-Quah (1989) setup.

    The DGP generates output growth and unemployment, where supply shocks
    drive the stochastic trend in output.

    Parameters
    ----------
    n : int
        Number of observations to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data : np.ndarray
        Array of shape (n, 2), columns are [output_growth, unemployment].
    params : dict
        Dictionary with true structural parameters including long-run matrix.
    """
    rng = np.random.default_rng(seed)

    # VAR(2) in [output_growth, unemployment]
    A1 = np.array([
        [0.30,  -0.10],
        [-0.15,  0.50],
    ])

    A2 = np.array([
        [0.10, -0.05],
        [-0.05, 0.15],
    ])

    # Structural impact matrix
    # Designed so that the long-run multiplier of demand shocks on output = 0
    # Long-run impact = (I - A1 - A2)^{-1} @ B0_inv
    # We need the (1,2) element of the long-run matrix to be zero.
    I_minus_A = np.eye(2) - A1 - A2
    I_minus_A_inv = np.linalg.inv(I_minus_A)

    # Choose B0_inv such that long-run restriction holds
    # Column 1 = supply shock (permanent effect on output)
    # Column 2 = demand shock (no long-run effect on output)
    # We need I_minus_A_inv @ B0_inv to have zero in position (0,1)
    # Let b = B0_inv column 2: I_minus_A_inv[0,:] @ b = 0
    # => I_minus_A_inv[0,0]*b[0] + I_minus_A_inv[0,1]*b[1] = 0
    # => b[0] = -I_minus_A_inv[0,1]/I_minus_A_inv[0,0] * b[1]
    b1_2 = 0.7
    b0_2 = -I_minus_A_inv[0, 1] / I_minus_A_inv[0, 0] * b1_2

    B0_inv = np.array([
        [0.8,  b0_2],
        [-0.3, b1_2],
    ])

    long_run = I_minus_A_inv @ B0_inv

    const = np.array([0.5, 0.0])

    burn = 300
    total = n + burn

    e = rng.multivariate_normal(np.zeros(2), np.eye(2), total)
    u = (B0_inv @ e.T).T

    y = np.zeros((total, 2))
    for t in range(total):
        y[t] = const + u[t]
        if t >= 1:
            y[t] += A1 @ y[t - 1]
        if t >= 2:
            y[t] += A2 @ y[t - 2]

    params = {
        "B0_inv": B0_inv,
        "A1": A1,
        "A2": A2,
        "const": const,
        "Sigma_u": B0_inv @ B0_inv.T,
        "long_run_matrix": long_run,
        "I_minus_A_inv": I_minus_A_inv,
        "variable_names": ["output_growth", "unemployment"],
    }

    return y[burn:], params


def generate_sign_restriction_dgp(
    n: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate a 3-variable system for sign restriction identification.

    Three structural shocks with economically motivated sign patterns:
      - Aggregate supply shock: output (+), inflation (-), rate (-)
      - Aggregate demand shock: output (+), inflation (+), rate (+)
      - Monetary policy shock: output (-), inflation (-), rate (+)

    Parameters
    ----------
    n : int
        Number of observations to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data : np.ndarray
        Array of shape (n, 3), columns are [output, inflation, rate].
    params : dict
        Dictionary with true structural parameters and sign pattern.
    """
    rng = np.random.default_rng(seed)

    A1 = np.array([
        [0.45, -0.05, -0.08],
        [0.08,  0.50,  0.02],
        [0.12,  0.18,  0.55],
    ])

    # B0_inv designed to produce the expected sign pattern in IRFs
    # Columns: [supply, demand, monetary_policy]
    B0_inv = np.array([
        [ 0.9,  0.5, -0.3],   # output
        [-0.4,  0.7, -0.2],   # inflation
        [-0.2,  0.3,  0.8],   # rate
    ])

    const = np.array([0.2, 0.15, 0.1])

    burn = 300
    total = n + burn

    e = rng.multivariate_normal(np.zeros(3), np.eye(3), total)
    u = (B0_inv @ e.T).T

    y = np.zeros((total, 3))
    for t in range(total):
        y[t] = const + u[t]
        if t >= 1:
            y[t] += A1 @ y[t - 1]

    sign_pattern = np.array([
        [+1, +1, -1],   # output
        [-1, +1, -1],   # inflation
        [-1, +1, +1],   # rate
    ])

    params = {
        "B0_inv": B0_inv,
        "A1": A1,
        "const": const,
        "Sigma_u": B0_inv @ B0_inv.T,
        "sign_pattern": sign_pattern,
        "shock_names": ["supply", "demand", "monetary_policy"],
        "variable_names": ["output", "inflation", "rate"],
    }

    return y[burn:], params
