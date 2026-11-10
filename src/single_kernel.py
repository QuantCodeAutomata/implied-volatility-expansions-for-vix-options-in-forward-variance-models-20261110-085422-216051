"""Single-kernel VIX option price approximation and implied volatility expansion.

Custom implementation (paper Sec. 2.2, Theorem 1 Eq. 15-16, Theorem 2 Eq. 21):
the weak price approximation C_P (Eq. 18) built from Black-Scholes prices and
their spatial derivatives, its BS-inversion implied volatility (*Approx*), and
the fully explicit implied volatility expansion sigma_hat_P (*Expansion*).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from src.bs import bs_call, dx_call, implied_vol_from_call
from src.kernel_model import KernelFunc, ProxyParams, proxy_and_gamma


class SingleKernelModel(NamedTuple):
    kernel: KernelFunc
    T: float
    Delta: float
    xi0: float


def compute_proxy(model: SingleKernelModel, n_t: int = 30, n_u: int = 30) -> ProxyParams:
    """Proposition 1 (Eq. 12) and Theorem 1 (Eq. 16) proxy and corrections."""
    return proxy_and_gamma(model.kernel, model.T, model.Delta, model.xi0, n_t, n_u)


def approx_call_price(proxy: ProxyParams, k: np.ndarray, T: float) -> np.ndarray:
    """Eq. (18): C_P = C^BS(x_P, k, sigma_tilde_P/2) + sum_i 2^-i*gamma_i*dx^i C^BS."""
    sigma_tilde_P = proxy.sigma_P / np.sqrt(T)
    x_P = proxy.mu_P / 2.0 + proxy.sigma_P ** 2 / 8.0
    sigma_half = sigma_tilde_P / 2.0
    price = bs_call(x_P, k, sigma_half, T)
    gammas = (proxy.gamma1, proxy.gamma2, proxy.gamma3)
    for i, gamma_i in zip((1, 2, 3), gammas):
        price = price + 2.0 ** (-i) * gamma_i * dx_call(i, x_P, k, sigma_half, T)
    return price


def approx_futures_price(proxy: ProxyParams) -> float:
    """Eq. (20): F_P = e^{x_P}*(1 + gamma_1/2 + gamma_2/4 + gamma_3/8)."""
    x_P = proxy.mu_P / 2.0 + proxy.sigma_P ** 2 / 8.0
    delta = proxy.gamma1 / 2.0 + proxy.gamma2 / 4.0 + proxy.gamma3 / 8.0
    return float(np.exp(x_P) * (1.0 + delta))


def approx_implied_vol(proxy: ProxyParams, k: np.ndarray, T: float) -> np.ndarray:
    """*Approx*: BS-inversion implied volatility of the weak price C_P (Eq. 18-20)."""
    k = np.atleast_1d(np.asarray(k, dtype=float))
    call_price = approx_call_price(proxy, k, T)
    F_P = approx_futures_price(proxy)
    return implied_vol_from_call(call_price, np.log(F_P), k, T)


def geometric_proxy_call_price(proxy: ProxyParams, log_moneyness: np.ndarray, T: float) -> np.ndarray:
    """Exact (log-normal) price of the single-kernel geometric proxy
    VIX_{T,P} = exp(x_P + (sigma_tilde_P/2)*Z), used as the control variate in
    the randomized QMC reference for the rough Bergomi model (Sec. 2.3): since
    the single-kernel proxy is exactly log-normal, its call price is the exact
    BS formula C^BS(x_P, k, sigma_tilde_P/2), with no weak-approximation error."""
    log_moneyness = np.atleast_1d(np.asarray(log_moneyness, dtype=float))
    sigma_tilde_P = proxy.sigma_P / np.sqrt(T)
    x_P = proxy.mu_P / 2.0 + proxy.sigma_P ** 2 / 8.0
    k_abs = x_P + log_moneyness
    return bs_call(x_P, k_abs, sigma_tilde_P / 2.0, T)


def geometric_proxy_futures_price(proxy: ProxyParams) -> float:
    """Exact futures price of the single-kernel geometric proxy, F_{T,P} = exp(x_P)."""
    x_P = proxy.mu_P / 2.0 + proxy.sigma_P ** 2 / 8.0
    return float(np.exp(x_P))


def expansion_implied_vol(proxy: ProxyParams, k: np.ndarray, T: float) -> np.ndarray:
    """Theorem 2, Eq. (21): explicit VIX implied volatility expansion sigma_hat_P.

    sigma_hat_P = sigma_tilde_P/2 + gamma_2/(2*sigma_tilde_P*T)
                  + 3*gamma_3/(8*sigma_tilde_P*T)
                  - gamma_3/(sigma_tilde_P^3*T^2) * (x_P - k)
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    sigma_tilde_P = proxy.sigma_P / np.sqrt(T)
    x_P = proxy.mu_P / 2.0 + proxy.sigma_P ** 2 / 8.0
    term0 = sigma_tilde_P / 2.0
    term1 = proxy.gamma2 / (2.0 * sigma_tilde_P * T)
    term2 = 3.0 * proxy.gamma3 / (8.0 * sigma_tilde_P * T)
    term3 = -proxy.gamma3 / (sigma_tilde_P ** 3 * T ** 2) * (x_P - k)
    return term0 + term1 + term2 + term3
