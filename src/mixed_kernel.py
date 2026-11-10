"""Mixed-kernel VIX option price approximation and implied volatility expansions.

Custom implementation (paper Sec. 3): the mixed log-normal proxy h(z) (Eq. 32),
the weak price approximation (Theorem 3, Eq. 36-41 via Proposition 2), its
Nth-order Hermite expansion (Proposition 3, Eq. 42-49), the fully explicit
implied volatility expansion with explicit zeroth-order term (Proposition 4,
Eq. 51-54, and Eq. 60 built by the Theorem-2 Vega-linearization technique
applied to Eq. 54), its "extended" variant absorbing the strike shift
(Remark 9, Eq. 61), and the implicit-proxy expansion (Theorem 4, Eq. 62-65).
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from src.bs import bs_call, d1 as bs_d1, hermite_prob, implied_vol_from_call, vega as bs_vega
from src.kernel_model import KernelFunc, ProxyParams, proxy_and_gamma
from src.quadrature import gauss_legendre_interval, gauss_legendre_unit


class MixedProxy(NamedTuple):
    """Per-component proxy parameters for a two-component mixed-kernel model."""
    lam: Tuple[float, float]  # (lambda_1, lambda_2)
    mu: Tuple[float, float]  # (mu_P1, mu_P2)
    sigma: Tuple[float, float]  # (sigma_P1, sigma_P2)
    gamma: Tuple[Tuple[float, float, float], Tuple[float, float, float]]  # (gamma_{i,1}, gamma_{i,2})


def compute_mixed_proxy(
    kernel1: KernelFunc, kernel2: KernelFunc, lam: float, T: float, Delta: float, xi0: float,
    n_t: int = 30, n_u: int = 30,
) -> MixedProxy:
    """Eq. (26)-(27): proxy parameters mu_{P,j}, sigma_{P,j}, and Theorem 1-type
    corrections gamma_{i,j}, computed per component j by replacing K with K_j in
    Proposition 1 / Eq. (16) (same flat-xi0 quadrature as the single-kernel case)."""
    p1 = proxy_and_gamma(kernel1, T, Delta, xi0, n_t, n_u)
    p2 = proxy_and_gamma(kernel2, T, Delta, xi0, n_t, n_u)
    return MixedProxy(
        lam=(lam, 1.0 - lam),
        mu=(p1.mu_P, p2.mu_P),
        sigma=(p1.sigma_P, p2.sigma_P),
        gamma=((p1.gamma1, p1.gamma2, p1.gamma3), (p2.gamma1, p2.gamma2, p2.gamma3)),
    )


def h_of_z(proxy: MixedProxy, z: np.ndarray) -> np.ndarray:
    """Eq. (32): h(z) = sum_j lambda_j * exp(mu_{P,j} + sigma_{P,j}*z)."""
    out = np.zeros_like(np.asarray(z, dtype=float))
    for j in range(2):
        out = out + proxy.lam[j] * np.exp(proxy.mu[j] + proxy.sigma[j] * z)
    return out


def q_j(proxy: MixedProxy, j: int, z: np.ndarray) -> np.ndarray:
    """Eq. (33): q_j(z) = gamma_{1,j} + gamma_{2,j}*z/sigma_{P,j} + gamma_{3,j}*(z^2-1)/sigma_{P,j}^2."""
    g1, g2, g3 = proxy.gamma[j]
    s = proxy.sigma[j]
    return g1 + g2 * z / s + g3 * (z ** 2 - 1.0) / s ** 2


def q_of_z(proxy: MixedProxy, z: np.ndarray) -> np.ndarray:
    """Eq. (33): q(z) = pi_1(z)*q_1(z) + pi_2(z)*q_2(z), pi_j(z) = lambda_j*exp(mu_j+sigma_j z)/h(z)."""
    h = h_of_z(proxy, z)
    total = np.zeros_like(np.asarray(z, dtype=float))
    for j in range(2):
        pi_j = proxy.lam[j] * np.exp(proxy.mu[j] + proxy.sigma[j] * z) / h
        total = total + pi_j * q_j(proxy, j, z)
    return total


def solve_A(proxy: MixedProxy, k: float, z_bound: float = 60.0) -> float:
    """A := h^{-1}(e^{2k}); h is a strictly increasing bijection R -> (0, infty) (Sec. 3.2)."""
    target = np.exp(2.0 * k)

    def obj(z: float) -> float:
        return h_of_z(proxy, np.array([z]))[0] - target

    lo, hi = -z_bound, z_bound
    while obj(lo) > 0:
        lo *= 2
    while obj(hi) < 0:
        hi *= 2
    return brentq(obj, lo, hi, xtol=1e-13, rtol=1e-13)


class Proposition2Factors(NamedTuple):
    """Factoring out component j (Proposition 2, Eq. 38)."""
    x_Pj: float
    Delta_sigma_Pj: float
    C_j: float
    A_j: float


def proposition2_factors(proxy: MixedProxy, j: int, A: float) -> Proposition2Factors:
    """Proposition 2 (Eq. 38): j in {0, 1} is the 0-indexed counterpart of the
    paper's j in {1, 2}. All formulas use the literal components 1 and 2
    (proxy.lam/mu/sigma[0] and [1]) with the sign s := (-1)^j_paper, i.e.
    s = -1 for j=0 (paper j=1) and s = +1 for j=1 (paper j=2)."""
    s = -1.0 if j == 0 else 1.0  # (-1)^j in the paper's 1-indexed convention
    lam1, lam2 = proxy.lam[0], proxy.lam[1]
    mu1, mu2 = proxy.mu[0], proxy.mu[1]
    sig1, sig2 = proxy.sigma[0], proxy.sigma[1]
    sig_j = proxy.sigma[j]
    x_Pj = proxy.mu[j] / 2.0 + sig_j ** 2 / 8.0
    Delta_sigma_Pj = s * (sig1 - sig2)
    C_j = (lam1 / lam2) ** s * np.exp(s * (mu1 - mu2) + 0.5 * sig_j * Delta_sigma_Pj)
    A_j = A - sig_j / 2.0
    return Proposition2Factors(x_Pj, Delta_sigma_Pj, C_j, A_j)


def g0_g1(proxy: MixedProxy, j: int, y: np.ndarray, factors: Proposition2Factors) -> Tuple[np.ndarray, np.ndarray]:
    """Eq. (38): g_0^j(y) and g_1^j(y) = 0.5*q(y + sigma_{P,j}/2)*g_0^j(y)."""
    sig_j = proxy.sigma[j]
    g0 = np.sqrt(1.0 + factors.C_j * np.exp(factors.Delta_sigma_Pj * y))
    g1 = 0.5 * q_of_z(proxy, y + sig_j / 2.0) * g0
    return g0, g1


def hermite_weights(
    proxy: MixedProxy, j: int, factors: Proposition2Factors,
    n_max: int, n_nodes: int, bound: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Eq. (43): omega_{n,j}^{(i)} for i in {0,1}, n = 0..n_max, via Gauss-Legendre
    quadrature on [-bound, bound] (Sec. 3.2.1)."""
    y, w = gauss_legendre_interval(-bound, bound, n_nodes)
    g0, g1 = g0_g1(proxy, j, y, factors)
    phi_y = norm.pdf(y)
    omega0 = np.empty(n_max + 1)
    omega1 = np.empty(n_max + 1)
    fact = 1.0
    for n in range(n_max + 1):
        if n > 0:
            fact *= n
        he_n = hermite_prob(n, y)
        omega0[n] = np.sum(w * g0 * he_n * phi_y) / fact
        omega1[n] = np.sum(w * g1 * he_n * phi_y) / fact
    return omega0, omega1


def select_truncation_order(
    proxy: MixedProxy, j: int, factors: Proposition2Factors,
    n_nodes: int, bound: float, eps_tol: float, n_max_cap: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Select the smallest N with MSE_j(N) < eps_tol (Eq. 44, Sec. 3.2.1), using
    Parseval's identity on the i=0 factor g_0^j."""
    omega0, omega1 = hermite_weights(proxy, j, factors, n_max_cap, n_nodes, bound)
    y, w = gauss_legendre_interval(-bound, bound, n_nodes)
    g0, _ = g0_g1(proxy, j, y, factors)
    norm_sq = np.sum(w * g0 ** 2 * norm.pdf(y))  # ||g_0^j||^2_{L^2(phi)}
    fact = 1.0
    partial = 0.0
    for n in range(n_max_cap + 1):
        if n > 0:
            fact *= n
        partial += fact * omega0[n] ** 2
        mse = norm_sq - partial
        if mse < eps_tol:
            return n, omega0, omega1
    return n_max_cap, omega0, omega1


def I_iN(x: float, omega: np.ndarray, N: int) -> float:
    """Eq. (45)-(46): I_{i,N}^j(x) = omega_0*Phi(-x) + sum_{n=1}^N omega_n*He_{n-1}(x)*phi(x)."""
    total = omega[0] * norm.cdf(-x)
    phi_x = norm.pdf(x)
    for n in range(1, N + 1):
        total += omega[n] * hermite_prob(n - 1, np.array([x]))[0] * phi_x
    return total


def hermite_call_price(
    proxy: MixedProxy, j: int, k: float, A: float, factors: Proposition2Factors,
    omega0: np.ndarray, omega1: np.ndarray, N: int,
) -> float:
    """Proposition 3, Eq. (47): Nth-order Hermite approximation of the VIX call price."""
    lam_j = proxy.lam[j]
    I0 = I_iN(factors.A_j, omega0, N)
    I1 = I_iN(factors.A_j, omega1, N)
    return np.sqrt(lam_j) * np.exp(factors.x_Pj) * (I0 + I1) - np.exp(k) * norm.cdf(-A)


def approx_gaussian_integrals(
    proxy: MixedProxy, k: np.ndarray, A: np.ndarray, n_nodes: int, b: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Theorem 3, Eq. (36)-(37): the weak approximation C_P = C_{P,0}+C_{P,1},
    F_P = F_{P,0}+F_{P,1}, as 1-D Gaussian integrals computed by Gauss-Legendre
    quadrature on [A, b] (calls) and [-b, b] (futures), Sec. 3.3.1."""
    k = np.atleast_1d(np.asarray(k, dtype=float))
    A = np.atleast_1d(np.asarray(A, dtype=float))
    C0 = np.empty_like(k)
    C1 = np.empty_like(k)
    for idx in range(k.size):
        z, w = gauss_legendre_interval(A[idx], b, n_nodes)
        sqrt_h = np.sqrt(h_of_z(proxy, z))
        phi_z = norm.pdf(z)
        C0[idx] = np.sum(w * sqrt_h * phi_z) - np.exp(k[idx]) * norm.cdf(-A[idx])
        C1[idx] = 0.5 * np.sum(w * q_of_z(proxy, z) * sqrt_h * phi_z)
    z, w = gauss_legendre_interval(-b, b, n_nodes)
    sqrt_h = np.sqrt(h_of_z(proxy, z))
    phi_z = norm.pdf(z)
    F0 = float(np.sum(w * sqrt_h * phi_z))
    F1 = float(0.5 * np.sum(w * q_of_z(proxy, z) * sqrt_h * phi_z))
    return C0, C1, F0, F1


def approx_implied_vol(proxy: MixedProxy, k: np.ndarray, T: float, n_nodes: int, b: float) -> np.ndarray:
    """*Approx*: BS-inversion implied volatility of the mixed weak approximation
    C_P = C_{P,0}(k) + C_{P,1}(k) (Theorem 3, Eq. 36)."""
    k = np.atleast_1d(np.asarray(k, dtype=float))
    A = np.array([solve_A(proxy, kk) for kk in k])
    C0, C1, F0, F1 = approx_gaussian_integrals(proxy, k, A, n_nodes, b)
    call_price = C0 + C1
    F = F0 + F1
    return implied_vol_from_call(call_price, np.log(F), k, T)


def geometric_proxy_price_functions(
    proxy: MixedProxy, n_nodes: int, b: float,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[], float]]:
    """Exact price/futures of the mixed log-normal proxy VIX_{T,P} = sqrt(h(Z))
    (no q(z) correction), used as the control variate in the randomized QMC
    reference for mixed rough Bergomi (Sec. 2.3, analogue of the single-kernel
    geometric proxy): C_{P,0}(k) = E[(sqrt(h(Z))-e^k)^+] and F_{P,0} = E[sqrt(h(Z))]
    from Eq. (37), computed by the same Gauss-Legendre Gaussian integral as the
    weak approximation's zeroth-order term."""
    dummy_A = solve_A(proxy, 0.0)
    _, _, F0, _ = approx_gaussian_integrals(proxy, np.array([0.0]), np.array([dummy_A]), n_nodes, b)
    F0 = float(F0)
    ln_F0 = np.log(F0)

    def futures_fn() -> float:
        return F0

    def call_fn(log_moneyness: np.ndarray) -> np.ndarray:
        log_moneyness = np.atleast_1d(np.asarray(log_moneyness, dtype=float))
        k_abs = ln_F0 + log_moneyness
        A = np.array([solve_A(proxy, kk) for kk in k_abs])
        C0, _, _, _ = approx_gaussian_integrals(proxy, k_abs, A, n_nodes, b)
        return C0

    return call_fn, futures_fn


def explicit_expansion_iv(
    proxy: MixedProxy, k: np.ndarray, T: float, j: int,
    n_nodes_hermite: int, bound_hermite: float, eps_tol: float, n_max_cap: int,
    extended: bool = False, N_override: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Eq. (60): explicit implied volatility expansion with explicit zeroth-order
    term (standard form), built by applying the Theorem-2 Vega-linearization
    technique to the exact BS representation (Eq. 54) of the Nth-order Hermite
    call price. Eq. (61) (Remark 9) is the 'extended' variant that absorbs the
    strike part of the shift residual via a Taylor expansion in log-strike,
    instead of discarding it as in (60).

    Returns (implied_vol, N_used) with N_used the selected Hermite order.
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    sigma_tilde_Pj = proxy.sigma[j] / np.sqrt(T)
    iv = np.empty_like(k)
    N_used = np.empty_like(k, dtype=int)

    # g_0^j, g_1^j (Eq. 38) depend on (C_j, Delta_sigma_Pj), fixed by (proxy, j);
    # they do not depend on the strike A, so the Hermite weights omega_{n,j}^{(i)}
    # (Eq. 43) and the truncation order N (Eq. 44) are computed once per smile,
    # not once per strike, and reused for every log-moneyness point.
    factors0 = proposition2_factors(proxy, j, 0.0)
    if N_override is not None:
        N = N_override
        omega0, omega1 = hermite_weights(proxy, j, factors0, N, n_nodes_hermite, bound_hermite)
    else:
        N, omega0, omega1 = select_truncation_order(
            proxy, j, factors0, n_nodes_hermite, bound_hermite, eps_tol, n_max_cap
        )

    for idx, kk in enumerate(k):
        A = solve_A(proxy, kk)
        factors = proposition2_factors(proxy, j, A)
        N_used[idx] = N

        # F_P, F_{P,0} via the Gaussian-integral futures approximation (Eq. 36/40)
        C0_vec, C1_vec, F0, F1 = approx_gaussian_integrals(proxy, np.array([kk]), np.array([A]), n_nodes_hermite, bound_hermite)
        F_P = F0 + F1

        ln_F_P = np.log(F_P)
        delta_j = kk - ln_F_P - A * sigma_tilde_Pj * np.sqrt(T) / 2.0 + sigma_tilde_Pj ** 2 * T / 8.0
        theta = 1.0 if delta_j > 0 else 0.0  # Remark 8

        x_Pj = factors.x_Pj
        A_j = factors.A_j
        phi_Aj = norm.pdf(A_j)
        hermite_sum = sum(
            omega0[n] * hermite_prob(n - 1, np.array([A_j]))[0]
            + omega1[n] * hermite_prob(n - 1, np.array([A_j]))[0]
            for n in range(1, N + 1)
        ) if N >= 1 else 0.0

        if not extended:
            # Eq. (60): Delta_sigma_{theta,j,N}(k) = sqrt(lambda_j)*e^{x_Pj} /
            # (e^{x_theta,j} * sqrt(T)) * sum_n sum_i omega_{n,j}^{(i)} He_{n-1}(A_j),
            # equivalently sqrt(lambda_j)*e^{x_Pj}*phi(A_j) / Vega(x_theta,j, k_theta,j,
            # sigma_tilde_Pj/2) since Vega(x_theta,j,k_theta,j,sigma_tilde_Pj/2) =
            # e^{x_theta,j}*sqrt(T)*phi(A_j) (d_1(x_theta,j,k_theta,j,sigma_tilde_Pj/2)=-A_j).
            # The shift residual R_{theta,j}^{shift} in (54)/(59) is explicitly neglected
            # when forming (60), as stated just above (60); it must not be added here.
            x_theta_j = ln_F_P + theta * delta_j
            k_theta_j = kk - (1.0 - theta) * delta_j
            vg = bs_vega(x_theta_j, k_theta_j, sigma_tilde_Pj / 2.0, T)
            delta_sigma = (np.sqrt(proxy.lam[j]) * np.exp(x_Pj) * phi_Aj * hermite_sum) / vg
        else:
            vg = bs_vega(ln_F_P, kk, sigma_tilde_Pj / 2.0, T)
            delta_sigma = (
                np.sqrt(proxy.lam[j]) * np.exp(x_Pj) * phi_Aj / vg * hermite_sum
            )
        iv[idx] = sigma_tilde_Pj / 2.0 + delta_sigma
    return iv, N_used


def implicit_proxy_expansion_iv(
    proxy: MixedProxy, k: np.ndarray, T: float, n_nodes: int, b: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Theorem 4, Eq. (62)-(65): implied vol expansion with implicit (numerical)
    zeroth-order term Sigma_0 (the implied vol of the mixed log-normal proxy,
    Eq. 64) and explicit first-order correction Sigma_1 (Eq. 65).

    Returns (sigma_hat, sigma_0_only, futures_F_P0).
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    A = np.array([solve_A(proxy, kk) for kk in k])
    C0, C1, F0, F1 = approx_gaussian_integrals(proxy, k, A, n_nodes, b)
    ln_F0 = np.log(F0)
    sigma0 = implied_vol_from_call(C0, ln_F0, k, T)
    sigma0 = np.atleast_1d(sigma0)
    d0 = bs_d1(ln_F0, k, sigma0, T)  # d_0(k) := d_1(ln(F_{P,0}), k, Sigma_0(k))
    vg = bs_vega(ln_F0, k, sigma0, T)
    sigma1 = (C1 - norm.cdf(d0) * F1) / vg
    return sigma0 + sigma1, sigma0, F0
