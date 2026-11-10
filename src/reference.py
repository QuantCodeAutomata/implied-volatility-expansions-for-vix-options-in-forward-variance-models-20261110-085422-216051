"""Reference VIX option prices: adaptive quadrature and randomized QMC.

Custom implementation (paper Sec. 2.3): both reference methods exploit that
every kernel used in the paper's numerical experiments is *proportional*
across mixed components (common kappa for the exponential kernel, common H
for the power-law kernel), so a single Gaussian driver G(u) = int_0^T
K0^u(t) dW_t governs every component.

- Standard/mixed-standard Bergomi: the exponential kernel factorizes as
  K0^u(t) = f(u)*g(t), so G(u) collapses to a single scalar Gaussian factor Z;
  the paper's "256 Gauss-Legendre nodes over the delivery dates and an
  adaptive Gauss-Kronrod quadrature in the Gaussian factor" is implemented by
  quadrature in z of the u-quadrature VIX_T^2(z).
- Rough/mixed-rough Bergomi: the power-law kernel does not factorize, so G(u)
  is simulated jointly over the u-quadrature grid via the Cholesky factor of
  its covariance matrix, driven by 16 independently scrambled Sobol sequences
  of 2^18 draws, with the geometric (log-normal mixture) proxy as a control
  variate sampled from the same underlying draws.
"""

from __future__ import annotations

from typing import Callable, List, NamedTuple, Sequence, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, qmc

from src.kernel_model import GaussianFactorModel

Component = Tuple[float, float]  # (amplitude_j, lambda_j)


class ReferenceResult(NamedTuple):
    call_prices: np.ndarray
    futures: float
    iv: np.ndarray
    iv_halfwidth: np.ndarray  # 95% CI half-width on the implied vol (QMC only; 0 for quadrature)


def _bs_implied_vol_safe(call_price: float, log_moneyness: float, T: float, F: float) -> float:
    from src.bs import implied_vol_from_call

    x = np.log(F)
    k = x + log_moneyness
    return float(implied_vol_from_call(call_price, x, np.array([k]), T)[0])


def _invert_vix_of_z(vix2_of_z, strike: float, lo: float, hi: float):
    """Locate z* with sqrt(vix2_of_z(z*)) == strike (monotone increasing in z)."""
    from scipy.optimize import brentq

    f = lambda z: np.sqrt(vix2_of_z(z)) - strike
    if f(lo) >= 0.0 or f(hi) <= 0.0:
        return None
    return brentq(f, lo, hi, xtol=1e-13, rtol=1e-13)


def reference_price_quadrature(
    components: Sequence[Component],
    gfm: GaussianFactorModel,
    xi0: float,
    Delta: float,
    T: float,
    log_moneyness: np.ndarray,
    z_bound: float = 12.0,
    quad_limit: int = 400,
) -> ReferenceResult:
    """Reference VIX call prices/futures for scalar-factor (proportional,
    separable-kernel) models via adaptive Gauss-Kronrod quadrature in the
    single Gaussian factor z (Sec. 2.3)."""
    assert gfm.is_scalar_factor, "quadrature reference requires a scalar Gaussian factor"
    s = gfm.scalar_scale  # shape (n_u,)
    u_w = gfm.u_weights

    def vix2_of_z(z: float) -> float:
        total = 0.0
        for amplitude, lam in components:
            Y = -0.5 * amplitude ** 2 * s ** 2 + amplitude * s * z
            total += lam * np.sum(u_w * xi0 * np.exp(Y)) / Delta
        return total

    futures_val, _ = quad(
        lambda z: np.sqrt(vix2_of_z(z)) * norm.pdf(z), -z_bound, z_bound,
        limit=quad_limit, epsabs=1e-14, epsrel=1e-13,
    )
    F = float(futures_val)
    x_fwd = np.log(F)

    call_prices = np.empty_like(log_moneyness, dtype=float)
    for idx, k_rel in enumerate(log_moneyness):
        k_abs = x_fwd + k_rel
        strike = np.exp(k_abs)

        def integrand(z: float, _strike=strike) -> float:
            return max(np.sqrt(vix2_of_z(z)) - _strike, 0.0) * norm.pdf(z)

        # Split at the payoff kink (where sqrt(vix2_of_z(z)) == strike) so the
        # adaptive quadrature does not lose precision across the non-smooth point.
        kink = _invert_vix_of_z(vix2_of_z, strike, -z_bound, z_bound)
        val = 0.0
        for a, b in ([(-z_bound, kink), (kink, z_bound)] if kink is not None else [(-z_bound, z_bound)]):
            part, _ = quad(integrand, a, b, limit=quad_limit, epsabs=1e-14, epsrel=1e-13)
            val += part
        call_prices[idx] = val

    iv = np.array([
        _bs_implied_vol_safe(call_prices[i], log_moneyness[i], T, F)
        for i in range(len(log_moneyness))
    ])
    return ReferenceResult(call_prices, F, iv, np.zeros_like(iv))


def reference_price_qmc(
    components: Sequence[Component],
    gfm: GaussianFactorModel,
    xi0: float,
    Delta: float,
    T: float,
    log_moneyness: np.ndarray,
    proxy_call_price: Callable[[np.ndarray], np.ndarray],
    proxy_futures_price: Callable[[], float],
    n_scrambles: int,
    n_paths_log2: int,
    seed: int,
) -> ReferenceResult:
    """Reference VIX call prices/futures for models whose Gaussian driver G(u)
    is a genuine process (power-law kernel): randomized QMC over the
    Cholesky-correlated u-grid, with the geometric log-normal-mixture proxy
    (same underlying draws) as a control variate (Sec. 2.3)."""
    cov = gfm.cov
    L = np.linalg.cholesky(cov + 1e-14 * np.eye(cov.shape[0]))
    u_w = gfm.u_weights
    n_u = cov.shape[0]
    n_paths = 2 ** n_paths_log2
    c_j = {amp: float(np.sum(u_w * np.diag(cov))) / Delta for amp, _ in components}

    exact_futures = float(proxy_futures_price())
    exact_calls = np.asarray(proxy_call_price(log_moneyness))

    n_k = len(log_moneyness)
    call_scramble_means = np.zeros((n_scrambles, n_k))
    futures_scramble_means = np.zeros(n_scrambles)
    x_fwd_guess = np.log(exact_futures)  # strikes are defined relative to the futures level

    for s_idx in range(n_scrambles):
        sampler = qmc.Sobol(d=n_u, scramble=True, seed=seed + s_idx)
        u01 = sampler.random(n_paths)
        z_std = norm.ppf(np.clip(u01, 1e-12, 1 - 1e-12))  # (n_paths, n_u)
        G = z_std @ L.T  # correlated Gaussian factor realizations, (n_paths, n_u)
        g_bar = (G @ u_w) / Delta  # scalar linear functional per path

        vix2_target = np.zeros(n_paths)
        vix2_proxy = np.zeros(n_paths)
        for amplitude, lam in components:
            Y = -0.5 * amplitude ** 2 * (np.diag(cov))[None, :] + amplitude * G
            vix2_target += lam * (np.exp(Y) @ u_w) * xi0 / Delta
            y_geo = -0.5 * amplitude ** 2 * c_j[amplitude] + amplitude * g_bar
            vix2_proxy += lam * xi0 * np.exp(y_geo)
        vix_target = np.sqrt(vix2_target)
        vix_proxy = np.sqrt(vix2_proxy)

        futures_scramble_means[s_idx] = np.mean(vix_target - vix_proxy)
        for i_k, k_rel in enumerate(log_moneyness):
            k_abs = x_fwd_guess + k_rel
            strike = np.exp(k_abs)
            payoff_target = np.maximum(vix_target - strike, 0.0)
            payoff_proxy = np.maximum(vix_proxy - strike, 0.0)
            call_scramble_means[s_idx, i_k] = np.mean(payoff_target - payoff_proxy)
        del G, z_std, u01, vix2_target, vix2_proxy, vix_target, vix_proxy

    futures_est = float(np.mean(futures_scramble_means)) + exact_futures
    futures_se = float(np.std(futures_scramble_means, ddof=1) / np.sqrt(n_scrambles))
    call_est = call_scramble_means.mean(axis=0) + exact_calls
    call_se = call_scramble_means.std(axis=0, ddof=1) / np.sqrt(n_scrambles)

    # call_est[i] is the price of the option with ABSOLUTE log-strike
    # k_abs = x_fwd_guess + log_moneyness[i] (the strike actually simulated
    # above, anchored at the proxy forward x_fwd_guess). Its log-moneyness
    # relative to the simulated futures futures_est is therefore
    # k_abs - ln(futures_est), not the nominal log_moneyness[i]: the proxy
    # forward and the true futures differ (that is the object of study), so
    # reusing log_moneyness[i] directly would invert the wrong strike.
    k_abs = x_fwd_guess + log_moneyness
    log_moneyness_actual = k_abs - np.log(futures_est)
    iv = np.array([
        _bs_implied_vol_safe(call_est[i], log_moneyness_actual[i], T, futures_est)
        for i in range(n_k)
    ])
    # propagate the call-price 95% CI to implied vol via the BS Vega (delta method)
    from src.bs import vega as bs_vega

    x_fwd = np.log(futures_est)
    vegas = np.array([
        bs_vega(x_fwd, x_fwd + log_moneyness_actual[i], max(iv[i], 1e-6), T) for i in range(n_k)
    ])
    iv_halfwidth = 1.96 * call_se / np.maximum(vegas, 1e-12)
    return ReferenceResult(call_est, futures_est, iv, iv_halfwidth)
