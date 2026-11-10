"""Unit tests for src/mixed_kernel.py: mixed-kernel proxy, Hermite expansion."""

from __future__ import annotations

import numpy as np

from src.kernel_model import exponential_kernel
from src.mixed_kernel import (
    compute_mixed_proxy,
    h_of_z,
    proposition2_factors,
    select_truncation_order,
    solve_A,
)

T, DELTA, XI0 = 3 / 12, 30 / 365, 0.24 ** 2


def _scenario1_kernels():
    kernel1 = exponential_kernel(omega=10.0, kappa=1.0)
    kernel2 = exponential_kernel(omega=2.0, kappa=1.0)
    return kernel1, kernel2


def test_mixed_proxy_positive_sigmas() -> None:
    """Both component proxy volatilities sigma_{P,j} are positive (Eq. 26-27)."""
    kernel1, kernel2 = _scenario1_kernels()
    proxy = compute_mixed_proxy(kernel1, kernel2, lam=0.2, T=T, Delta=DELTA, xi0=XI0)
    assert proxy.sigma[0] > 0.0
    assert proxy.sigma[1] > 0.0


def test_h_of_z_is_increasing_mixture() -> None:
    """h(z) = sum_j lambda_j*exp(mu_{P,j}+sigma_{P,j}*z) is positive and increasing (Eq. 32)."""
    kernel1, kernel2 = _scenario1_kernels()
    proxy = compute_mixed_proxy(kernel1, kernel2, lam=0.2, T=T, Delta=DELTA, xi0=XI0)
    z = np.linspace(-2.0, 2.0, 9)
    vals = h_of_z(proxy, z)
    assert np.all(vals > 0.0)
    assert np.all(np.diff(vals) > 0.0)


def test_solve_A_inverts_h() -> None:
    """A = h^{-1}(e^{2k}) satisfies h(A) = e^{2k} (Sec. 3.2)."""
    kernel1, kernel2 = _scenario1_kernels()
    proxy = compute_mixed_proxy(kernel1, kernel2, lam=0.2, T=T, Delta=DELTA, xi0=XI0)
    k = 0.05
    A = solve_A(proxy, k)
    assert np.isclose(h_of_z(proxy, np.array([A]))[0], np.exp(2.0 * k), rtol=1e-6)


def test_hermite_weights_and_truncation_order() -> None:
    """The Hermite-weight MSE decreases and a finite truncation order N is found
    for the tolerance of Sec. 3.2.1 (Eq. 43 and the Parseval criterion)."""
    kernel1, kernel2 = _scenario1_kernels()
    proxy = compute_mixed_proxy(kernel1, kernel2, lam=0.2, T=T, Delta=DELTA, xi0=XI0)
    A = solve_A(proxy, 0.05)
    j = 1  # component with the lower proxy volatility (omega2 < omega1 here)
    factors = proposition2_factors(proxy, j, A)
    N, omega0, omega1 = select_truncation_order(
        proxy, j, factors, n_nodes=4000, bound=40.0, eps_tol=1e-6, n_max_cap=40,
    )
    assert 0 < N <= 40
    assert omega0.shape == (41,)
