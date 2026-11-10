"""Unit tests for src/kernel_model.py: kernels, proxy parameters, corrections."""

from __future__ import annotations

import numpy as np

from src.kernel_model import (
    build_gaussian_factor_model,
    exponential_kernel,
    power_law_kernel,
    proxy_and_gamma,
)

T, DELTA, XI0 = 3 / 12, 30 / 365, 0.24 ** 2


def test_exponential_kernel_shape() -> None:
    """K^u(t) = omega*exp(-kappa*(u-t)) is positive and decays with (u-t) (Sec. 2.3)."""
    kernel = exponential_kernel(omega=2.0, kappa=0.25)
    t = np.array([0.0, 0.1, 0.2])
    u = 0.5
    vals = kernel(u, t)  # kernel(u, t), Eq. (2)
    assert np.all(vals > 0.0)
    assert np.all(np.diff(vals) > 0.0)  # increasing t -> smaller (u-t) -> less decay -> larger K


def test_power_law_kernel_singularity() -> None:
    """K^u(t) = eta*(u-t)^(H-1/2) blows up as t -> u for H < 1/2 (Sec. 2.3)."""
    kernel = power_law_kernel(eta=1.0, H=0.1)
    u = 0.5
    t_near = np.array([u - 1e-6, u - 1e-3, u - 1e-1])
    vals = kernel(u, t_near)  # kernel(u, t), Eq. (3)
    assert np.all(np.diff(vals) < 0.0)  # closer to u -> larger value


def test_proxy_sigma_positive_and_scales_with_omega() -> None:
    """sigma_P > 0 (Prop. 1, Eq. 12); scaling omega up should scale sigma_P up."""
    kernel_small = exponential_kernel(omega=2.0, kappa=0.25)
    kernel_large = exponential_kernel(omega=8.0, kappa=0.25)
    proxy_small = proxy_and_gamma(kernel_small, T, DELTA, XI0, 30, 30)
    proxy_large = proxy_and_gamma(kernel_large, T, DELTA, XI0, 30, 30)
    assert proxy_small.sigma_P > 0.0
    assert proxy_large.sigma_P > proxy_small.sigma_P


def test_gaussian_factor_model_scalar_for_exponential() -> None:
    """The exponential kernel factorizes into a scalar Gaussian factor (Sec. 2.3)."""
    kernel = exponential_kernel(omega=2.0, kappa=0.25)
    gfm = build_gaussian_factor_model(kernel, T, DELTA, n_u=64)
    assert gfm.is_scalar_factor


def test_gaussian_factor_model_not_scalar_for_power_law() -> None:
    """The power-law kernel does not factorize into a scalar factor (Sec. 2.3)."""
    kernel = power_law_kernel(eta=1.0, H=0.1)
    gfm = build_gaussian_factor_model(kernel, T, DELTA, n_u=32)
    assert not gfm.is_scalar_factor
