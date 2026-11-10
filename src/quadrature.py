"""Gauss-Legendre quadrature helpers with the kernel-singularity substitution.

Custom implementation (paper Sec. 2.3): the correction coefficients mu_P,
sigma_P, gamma_i require integrating K^u(t) and K^u(t)^2 over t in [0,T] and
u in [T, T+Delta]; both kernels considered in the paper (exponential and
power-law) are singular, or nearly so, at u = t = T. The paper resolves this
with the substitution t = T(1-(1-s)^4), u = T + Delta*s^4, s in [0,1], which
flattens the integrand near s = 0 (Sec. 2.3). numpy.polynomial.legendre.leggauss
supplies the underlying Gauss-Legendre nodes/weights.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy.special import roots_legendre


@lru_cache(maxsize=None)
def _leggauss_cached(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights on [-1, 1], memoized per order n.

    scipy.special.roots_legendre uses Newton iteration on the asymptotic
    initial guess (Trefethen, "Is Gauss Quadrature Better than
    Clenshaw-Curtis?"), which is markedly faster than the dense
    eigenvalue solve behind numpy.polynomial.legendre.leggauss for the
    large orders used here (n = 4000 for the Hermite-weight quadrature,
    Sec. 3.2.1); both agree to machine precision. The result is memoized
    because the same order n is requested many times (per strike, per
    Hermite order, per scenario/maturity).
    """
    return roots_legendre(n)


def gauss_legendre_unit(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """GL nodes/weights mapped from [-1, 1] to [0, 1]."""
    nodes, weights = _leggauss_cached(n)
    s = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    return s, w


def t_substitution_nodes(T: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Nodes/weights for integrating over t in [0, T] via t = T*(1-(1-s)^4).

    dt/ds = 4*T*(1-s)^3, so the quadrature weight on t is w_s * 4*T*(1-s)^3.
    """
    s, w_s = gauss_legendre_unit(n)
    t = T * (1.0 - (1.0 - s) ** 4)
    dt_ds = 4.0 * T * (1.0 - s) ** 3
    return t, w_s * dt_ds


def u_substitution_nodes(T: float, Delta: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Nodes/weights for integrating over u in [T, T+Delta] via u = T + Delta*s^4.

    du/ds = 4*Delta*s^3, so the quadrature weight on u is w_s * 4*Delta*s^3.
    All nodes satisfy u > T strictly (interior Gauss-Legendre nodes), so the
    kernel singularity at u = t = T is never sampled.
    """
    s, w_s = gauss_legendre_unit(n)
    u = T + Delta * s ** 4
    du_ds = 4.0 * Delta * s ** 3
    return u, w_s * du_ds


def gauss_legendre_interval(a: float, b: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """GL nodes/weights mapped from [-1, 1] to an arbitrary [a, b]."""
    nodes, weights = _leggauss_cached(n)
    x = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    w = 0.5 * (b - a) * weights
    return x, w
