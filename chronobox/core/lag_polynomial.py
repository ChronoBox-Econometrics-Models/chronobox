"""LagPolynomial - Lag operator polynomial representation for ARMA models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class LagPolynomial:
    """Lag operator polynomial.

    Represents polynomials of the form:
        p(L) = c_0 + c_1*L + c_2*L^2 + ... + c_n*L^n

    For AR polynomials: phi(L) = 1 - phi_1*L - ... - phi_p*L^p
        coeffs = [1, -phi_1, -phi_2, ..., -phi_p]

    For MA polynomials: theta(L) = 1 + theta_1*L + ... + theta_q*L^q
        coeffs = [1, theta_1, theta_2, ..., theta_q]

    Parameters
    ----------
    coeffs : array-like
        Polynomial coefficients, starting with the L^0 coefficient (always 1).
    """

    def __init__(self, coeffs: NDArray[np.float64] | list[float]) -> None:
        self.coeffs = np.asarray(coeffs, dtype=np.float64)
        if len(self.coeffs) == 0:
            msg = "coefficients must not be empty"
            raise ValueError(msg)

    @property
    def order(self) -> int:
        """Degree of the polynomial."""
        return len(self.coeffs) - 1

    def roots(self) -> NDArray[np.complex128]:
        """Compute roots of the polynomial.

        Returns
        -------
        ndarray of complex
            Roots of the polynomial.
        """
        if self.order == 0:
            return np.array([], dtype=np.complex128)
        # np.roots expects descending power order; our coeffs are ascending
        return np.roots(self.coeffs[::-1])

    def is_stationary(self) -> bool:
        """Check if all roots lie outside the unit circle.

        For AR polynomials, this is the stationarity condition.

        Returns
        -------
        bool
            True if all roots have |z| > 1.
        """
        r = self.roots()
        if len(r) == 0:
            return True
        return bool(np.all(np.abs(r) > 1.0))

    def is_invertible(self) -> bool:
        """Check if all roots lie outside the unit circle.

        For MA polynomials, this is the invertibility condition.
        Mathematically identical to is_stationary.

        Returns
        -------
        bool
            True if all roots have |z| > 1.
        """
        return self.is_stationary()

    def multiply(self, other: LagPolynomial) -> LagPolynomial:
        """Multiply two lag polynomials (convolution of coefficients).

        Parameters
        ----------
        other : LagPolynomial
            The other polynomial.

        Returns
        -------
        LagPolynomial
            Product polynomial.
        """
        return LagPolynomial(np.convolve(self.coeffs, other.coeffs))

    def to_companion(self) -> NDArray[np.float64]:
        """Convert to companion matrix form.

        For phi(L) = 1 - phi_1*L - ... - phi_p*L^p, the companion matrix is:

        T = [[phi_1, phi_2, ..., phi_p],
             [1,     0,     ..., 0    ],
             [0,     1,     ..., 0    ],
             [...                     ],
             [0,     0,     ..., 0    ]]

        Where phi_i = -coeffs[i] (sign flip because coeffs stores 1, -phi_1, ...).

        Returns
        -------
        ndarray of shape (p, p)
            Companion matrix.
        """
        p = self.order
        if p == 0:
            return np.array([[]], dtype=np.float64).reshape(0, 0)
        companion = np.zeros((p, p), dtype=np.float64)
        # First row: AR coefficients (flip sign from polynomial form)
        companion[0, :] = -self.coeffs[1:]
        # Sub-diagonal: identity
        if p > 1:
            companion[1:, :-1] += np.eye(p - 1)
        return companion

    def __repr__(self) -> str:
        terms = []
        for i, c in enumerate(self.coeffs):
            if i == 0:
                terms.append(f"{c:.4f}")
            elif c != 0:
                sign = "+" if c > 0 else "-"
                terms.append(f" {sign} {abs(c):.4f}*L^{i}")
        return "LagPolynomial(" + "".join(terms) + ")"

    def __len__(self) -> int:
        return len(self.coeffs)
