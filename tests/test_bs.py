"""Unit tests for src/bs.py: Black-Scholes price formula, Greeks, inversion."""

from __future__ import annotations

import numpy as np

from src.bs import bs_call, bs_put, d1, implied_vol_from_call, vega


def test_put_call_parity() -> None:
    """C^BS(x,k,sigma) - P^BS(x,k,sigma) = e^x - e^k (Eq. 4-5)."""
    x, k, sigma, T = 0.1, -0.05, 0.3, 0.5
    call = bs_call(x, np.array([k]), sigma, T)[0]
    put = bs_put(x, np.array([k]), sigma, T)[0]
    assert np.isclose(call - put, np.exp(x) - np.exp(k), atol=1e-10)


def test_call_price_bounds() -> None:
    """A call price lies in (0, e^x) for finite volatility and strictly positive spot."""
    x, sigma, T = 0.2, 0.25, 1.0
    k = np.linspace(-0.3, 0.3, 11)
    call = bs_call(x, k, sigma, T)
    assert np.all(call > 0.0)
    assert np.all(call < np.exp(x))


def test_call_price_monotone_in_vol() -> None:
    """Vega > 0: the BS call price is increasing in volatility."""
    x, k, T = 0.0, 0.05, 0.25
    sigmas = np.array([0.1, 0.2, 0.3, 0.4])
    prices = np.array([bs_call(x, np.array([k]), s, T)[0] for s in sigmas])
    assert np.all(np.diff(prices) > 0.0)
    assert np.all(vega(x, np.array([k]), sigmas, T) > 0.0)


def test_implied_vol_round_trip() -> None:
    """Inverting a BS price recovers the input volatility (Eq. 10)."""
    x, T = 0.0, 0.5
    k = np.linspace(-0.2, 0.2, 5)
    sigma_true = 0.35
    price = bs_call(x, k, sigma_true, T)
    sigma_hat = implied_vol_from_call(price, x, k, T)
    assert np.allclose(sigma_hat, sigma_true, atol=1e-8)


def test_d1_shift() -> None:
    """d1 - d2 = sigma*sqrt(T) (Eq. 6)."""
    from src.bs import d2

    x, k, sigma, T = 0.0, 0.0, 0.2, 2.0
    assert np.isclose(d1(x, k, sigma, T) - d2(x, k, sigma, T), sigma * np.sqrt(T))
