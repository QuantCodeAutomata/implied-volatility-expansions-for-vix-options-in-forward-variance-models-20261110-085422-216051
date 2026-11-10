"""Forward-variance kernels and the single-kernel proxy parameters/corrections.

Custom implementation (paper Sec. 1.2, 2.1, Prop. 1 Eq. 12, Theorem 1 Eq. 16):
the standard and rough Bergomi kernels K^u(t), and the quadrature evaluation
of the log-normal proxy parameters (mu_P, sigma_P) and the weak-approximation
correction coefficients (gamma_1, gamma_2, gamma_3), using the substitution of
Sec. 2.3 to resolve the kernel singularity at u = t = T.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np

from src.quadrature import t_substitution_nodes, u_substitution_nodes

KernelFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]


def exponential_kernel(omega: float, kappa: float) -> KernelFunc:
    """Standard Bergomi kernel, Eq. (2): K^u(t) = omega*exp(-kappa*(u-t))."""

    def kernel(u: np.ndarray, t: np.ndarray) -> np.ndarray:
        return omega * np.exp(-kappa * (u - t))

    return kernel


def power_law_kernel(eta: float, H: float) -> KernelFunc:
    """Rough Bergomi kernel, Eq. (3): K^u(t) = eta*(u-t)^(H-1/2), H in (0, 1/2)."""
    assert 0.0 < H < 0.5, "H must lie in (0, 1/2) for the power-law kernel"

    def kernel(u: np.ndarray, t: np.ndarray) -> np.ndarray:
        return eta * (u - t) ** (H - 0.5)

    return kernel


class ProxyParams(NamedTuple):
    mu_P: float
    sigma_P: float
    gamma1: float
    gamma2: float
    gamma3: float
    E_vix2: float


def proxy_and_gamma(
    kernel: KernelFunc,
    T: float,
    Delta: float,
    xi0: float,
    n_t: int = 30,
    n_u: int = 30,
) -> ProxyParams:
    """Compute (mu_P, sigma_P, gamma_1, gamma_2, gamma_3) for a single kernel.

    Implements Proposition 1 Eq. (12) and Theorem 1 Eq. (16), via the
    Gauss-Legendre substitution of Sec. 2.3 with n_t nodes over t in [0, T]
    and n_u nodes over u in [T, T+Delta]. Assumes a flat forward variance
    curve xi_0^u = xi0 on [T, T+Delta], as used throughout the paper's
    numerical sections.
    """
    t_nodes, t_w = t_substitution_nodes(T, n_t)
    u_nodes, u_w = u_substitution_nodes(T, Delta, n_u)

    K_tu = kernel(u_nodes[None, :], t_nodes[:, None])  # shape (n_t, n_u)

    xi0_u = np.full_like(u_nodes, xi0)
    E_vix2 = float(np.sum(u_w * xi0_u) / Delta)
    w_u = xi0_u / E_vix2  # normalized weight; w_u == 1 for a flat curve

    # Weighted (over u) averages of K^u(t) and K^u(t)^2, as functions of t.
    A_t = (u_w * w_u) @ (K_tu ** 2).T / Delta  # shape (n_t,)
    B_t = (u_w * w_u) @ K_tu.T / Delta  # shape (n_t,)

    mu_P = float(np.log(E_vix2) - 0.5 * np.sum(t_w * A_t))
    sigma_P2 = float(np.sum(t_w * B_t ** 2))
    sigma_P = float(np.sqrt(sigma_P2))

    # Per-u inner t-integrals entering gamma_1, gamma_2, gamma_3 (Eq. 16).
    I1 = t_w @ (K_tu ** 2 - A_t[:, None])  # shape (n_u,)
    I2 = t_w @ (B_t[:, None] * (K_tu - B_t[:, None]))  # shape (n_u,)
    I3 = t_w @ (K_tu - B_t[:, None]) ** 2  # shape (n_u,)

    gamma1 = float(
        (1.0 / 8.0) * np.sum(u_w * w_u * I1 ** 2) / Delta
        + 0.5 * np.sum(u_w * w_u * I3) / Delta
    )
    gamma2 = float(-0.5 * np.sum(u_w * w_u * I1 * I2) / Delta)
    gamma3 = float(0.5 * np.sum(u_w * w_u * I2 ** 2) / Delta)

    return ProxyParams(mu_P, sigma_P, gamma1, gamma2, gamma3, E_vix2)


@dataclass(frozen=True)
class GaussianFactorModel:
    """Gaussian-process representation of the common driver of K^u(t).

    Both kernels used in this paper are *proportional* across the two mixed
    components (common kappa for the exponential kernel, common H for the
    power-law kernel); the shared shape K0^u(t) therefore drives a single
    Gaussian process G(u) = int_0^T K0^u(t) dW_t for all components. For the
    exponential kernel, K0 factorizes as f(u)*g(t), so G(u) collapses to a
    single scalar Gaussian factor (Sec. 2.3's "quadrature" reference); for the
    power-law kernel, G(u) is a genuine Gaussian process sampled jointly over
    a quadrature grid (Sec. 2.3's "randomized QMC" reference).
    """

    u_nodes: np.ndarray
    u_weights: np.ndarray
    cov: np.ndarray  # covariance matrix of G(u_i), G(u_j)
    is_scalar_factor: bool
    scalar_scale: np.ndarray  # if is_scalar_factor: G(u_i) = scalar_scale[i] * Z


def build_gaussian_factor_model(
    kernel0: KernelFunc, T: float, Delta: float, n_u: int, n_t_cov: int = 400
) -> GaussianFactorModel:
    """Build the covariance structure of G(u) = int_0^T K0^u(t) dW_t on a u-grid.

    cov[i, j] = int_0^T K0^{u_i}(t) * K0^{u_j}(t) dt, computed by a fine
    Gauss-Legendre quadrature over t in [0, T] (all u-nodes satisfy u > T, so
    the integrand is smooth and no substitution is required for this integral).
    """
    u_nodes, u_w = u_substitution_nodes(T, Delta, n_u)
    t_nodes, t_w = t_substitution_nodes(T, n_t_cov)

    K_tu = kernel0(u_nodes[None, :], t_nodes[:, None])  # (n_t_cov, n_u)
    cov = K_tu.T @ (K_tu * t_w[:, None])  # (n_u, n_u)

    # Detect separability K0^u(t) = f(u)*g(t) (exponential kernel): cov is then
    # rank-1, cov[i,j] = f(u_i)*f(u_j)*int g(t)^2 dt.
    rank1_ratio = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    is_scalar = np.allclose(rank1_ratio, 1.0, atol=1e-10)
    scalar_scale = np.sqrt(np.diag(cov)) if is_scalar else np.array([])
    return GaussianFactorModel(u_nodes, u_w, cov, is_scalar, scalar_scale)
