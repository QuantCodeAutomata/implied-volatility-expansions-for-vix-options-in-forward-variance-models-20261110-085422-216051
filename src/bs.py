"""Black-Scholes price formula, Greeks, and spatial derivatives.

Implements Eq. (4)-(9) of Liao, Agarwal, Bourgey, "Implied Volatility
Expansions for VIX Options in Forward Variance Models": the BS call/put price
as functions of log-spot x, log-strike k, volatility sigma and maturity T, the
d1/d2 auxiliary variables, Vega/Vomma, and the spatial-derivative identities
used to build the VIX option price approximations in Sec. 2.2.
"""

from __future__ import annotations

import numpy as np
from scipy.special import eval_hermitenorm
from scipy.stats import norm
from scipy.optimize import brentq


def hermite_prob(n: int, x: np.ndarray) -> np.ndarray:
    """Probabilists' Hermite polynomial He_n(x) (scipy.special.eval_hermitenorm)."""
    return eval_hermitenorm(n, x)


def d1(x: np.ndarray, k: np.ndarray, sigma: np.ndarray, T: float) -> np.ndarray:
    """Eq. (6): d1(x, k, sigma) = (x-k)/(sigma*sqrt(T)) + sigma*sqrt(T)/2."""
    sqrt_t = np.sqrt(T)
    return (x - k) / (sigma * sqrt_t) + sigma * sqrt_t / 2.0


def d2(x: np.ndarray, k: np.ndarray, sigma: np.ndarray, T: float) -> np.ndarray:
    """Eq. (6): d2 = d1 - sigma*sqrt(T)."""
    return d1(x, k, sigma, T) - sigma * np.sqrt(T)


def bs_call(x: np.ndarray, k: np.ndarray, sigma: np.ndarray, T: float) -> np.ndarray:
    """Eq. (4): C^BS(x,k,sigma) = e^x*Phi(d1) - e^k*Phi(d2)."""
    d_1 = d1(x, k, sigma, T)
    d_2 = d_1 - sigma * np.sqrt(T)
    return np.exp(x) * norm.cdf(d_1) - np.exp(k) * norm.cdf(d_2)


def bs_put(x: np.ndarray, k: np.ndarray, sigma: np.ndarray, T: float) -> np.ndarray:
    """Eq. (5): P^BS(x,k,sigma) = e^k*Phi(-d2) - e^x*Phi(-d1)."""
    d_1 = d1(x, k, sigma, T)
    d_2 = d_1 - sigma * np.sqrt(T)
    return np.exp(k) * norm.cdf(-d_2) - np.exp(x) * norm.cdf(-d_1)


def vega(x: np.ndarray, k: np.ndarray, sigma: np.ndarray, T: float) -> np.ndarray:
    """Eq. (7): Vega = e^x*sqrt(T)*phi(d1)."""
    d_1 = d1(x, k, sigma, T)
    return np.exp(x) * np.sqrt(T) * norm.pdf(d_1)


def vomma(x: np.ndarray, k: np.ndarray, sigma: np.ndarray, T: float) -> np.ndarray:
    """Eq. (8): Vomma = e^x*sqrt(T)*phi(d1)*((x-k)^2/(sigma^3*T) - sigma*T/4)."""
    d_1 = d1(x, k, sigma, T)
    return np.exp(x) * np.sqrt(T) * norm.pdf(d_1) * ((x - k) ** 2 / (sigma ** 3 * T) - sigma * T / 4.0)


def dx_call(i: int, x: np.ndarray, k: np.ndarray, sigma: np.ndarray, T: float) -> np.ndarray:
    """Eq. (17): i-th order spatial derivative of C^BS w.r.t. x, i in {1,2,3}."""
    d_1 = d1(x, k, sigma, T)
    out = np.exp(x) * norm.cdf(d_1)
    if i >= 2:
        extra = np.zeros_like(out)
        for j in range(1, i):
            binom = np.math.comb(i - 1, j)
            extra += binom * (-1.0) ** (j - 1) * hermite_prob(j - 1, d_1) / (sigma ** j * T ** (j / 2.0))
        out = out + np.exp(x) * norm.pdf(d_1) * extra
    return out


def dx_put(i: int, x: np.ndarray, k: np.ndarray, sigma: np.ndarray, T: float) -> np.ndarray:
    """Spatial derivative of P^BS w.r.t. x (below Eq. 19)."""
    d_1 = d1(x, k, sigma, T)
    out = -np.exp(x) * norm.cdf(-d_1)
    if i >= 2:
        extra = np.zeros_like(out)
        for j in range(1, i):
            binom = np.math.comb(i - 1, j)
            extra += binom * (-1.0) ** (j - 1) * hermite_prob(j - 1, d_1) / (sigma ** j * T ** (j / 2.0))
        out = out + np.exp(x) * norm.pdf(d_1) * extra
    return out


def implied_vol_from_call(
    price: np.ndarray, x: float, k: np.ndarray, T: float,
    lo: float = 1e-6, hi: float = 5.0,
) -> np.ndarray:
    """Invert C^BS(x, k, sigma) = price for sigma > 0 (Eq. 10), via Brent's method."""
    price = np.atleast_1d(np.asarray(price, dtype=float))
    k = np.atleast_1d(np.asarray(k, dtype=float))
    out = np.empty_like(price)
    for idx in range(price.size):

        def obj(sigma: float, _p=price[idx], _k=k[idx]) -> float:
            return bs_call(x, _k, sigma, T) - _p

        f_lo, f_hi = obj(lo), obj(hi)
        if f_lo * f_hi > 0:
            # widen the bracket once; a persistent sign mismatch is a genuine error
            f_hi = obj(hi := 20.0)
            if f_lo * f_hi > 0:
                raise ValueError(f"implied_vol_from_call: no sign change on [{lo},{hi}]")
        out[idx] = brentq(obj, lo, hi, xtol=1e-12, rtol=1e-12)
    return out
