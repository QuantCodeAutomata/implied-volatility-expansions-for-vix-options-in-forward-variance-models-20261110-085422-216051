"""Shared scenario setup for the mixed-kernel experiments (Sec. 3, Table 3).

Builds, once per scenario, the two-component mixed forward-variance model,
the mixed log-normal proxy of Eq. (26)-(27), the Gaussian-factor model used by
the quadrature/QMC reference of Sec. 2.3 (extended to the mixed case by the
component amplitudes), and the reference VIX option prices. Shared by
Experiments 3-6 so every one of them uses the same proxy, reference, and
"lower proxy volatility" component selection.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from src import config
from src.kernel_model import (
    GaussianFactorModel,
    KernelFunc,
    build_gaussian_factor_model,
    exponential_kernel,
    power_law_kernel,
)
from src.mixed_kernel import MixedProxy, compute_mixed_proxy, geometric_proxy_price_functions
from src.reference import Component, ReferenceResult, reference_price_qmc, reference_price_quadrature


class ScenarioKernels(NamedTuple):
    """Kernels of a mixed scenario: two amplitude-scaled copies of a shared shape."""
    kernel1: KernelFunc
    kernel2: KernelFunc
    kernel0: KernelFunc  # unit-amplitude shape kernel shared by both components
    amplitude1: float
    amplitude2: float
    lam: float
    is_rough: bool


def build_scenario_kernels(scenario: dict) -> ScenarioKernels:
    """Build the per-component kernels of a mixed scenario (Table 3).

    Mixed standard Bergomi components are proportional exponential kernels
    with common kappa: K_j^u(t) = omega_j*exp(-kappa(u-t)). Mixed rough
    Bergomi components are proportional power-law kernels with common H:
    K_j^u(t) = eta_j*(u-t)^(H-1/2).
    """
    if scenario["model"] == "standard":
        kappa = scenario["kappa"]
        return ScenarioKernels(
            kernel1=exponential_kernel(scenario["omega1"], kappa),
            kernel2=exponential_kernel(scenario["omega2"], kappa),
            kernel0=exponential_kernel(1.0, kappa),
            amplitude1=scenario["omega1"], amplitude2=scenario["omega2"],
            lam=scenario["lam"], is_rough=False,
        )
    if scenario["model"] == "rough":
        H = scenario["H"]
        return ScenarioKernels(
            kernel1=power_law_kernel(scenario["eta1"], H),
            kernel2=power_law_kernel(scenario["eta2"], H),
            kernel0=power_law_kernel(1.0, H),
            amplitude1=scenario["eta1"], amplitude2=scenario["eta2"],
            lam=scenario["lam"], is_rough=True,
        )
    raise ValueError(f"unknown mixed-scenario model type {scenario['model']!r}")


def mixed_proxy_for_scenario(scenario: dict, T: float) -> MixedProxy:
    """Mixed proxy parameters mu_{P,j}, sigma_{P,j}, gamma_{i,j} (Eq. 26-27)."""
    kernels = build_scenario_kernels(scenario)
    return compute_mixed_proxy(
        kernels.kernel1, kernels.kernel2, kernels.lam, T, config.DELTA, config.XI0,
        n_t=config.N_GL_PROXY, n_u=config.N_GL_PROXY,
    )


def lower_vol_component(proxy: MixedProxy) -> int:
    """0-indexed component j with the lower of sigma_{P,1}, sigma_{P,2} (Sec. 3.2):
    unless stated otherwise, the paper's explicit expansion is built around this
    component."""
    return 0 if proxy.sigma[0] <= proxy.sigma[1] else 1


def _gfm_for_scenario(kernels: ScenarioKernels, T: float) -> GaussianFactorModel:
    n_u = config.N_GL_REF_U_STANDARD if not kernels.is_rough else config.N_GL_REF_U_ROUGH
    gfm = build_gaussian_factor_model(kernels.kernel0, T, config.DELTA, n_u)
    assert gfm.is_scalar_factor != kernels.is_rough
    return gfm


def reference_for_scenario(
    scenario: dict, proxy: MixedProxy, T: float, log_moneyness: np.ndarray,
) -> ReferenceResult:
    """Reference VIX call prices/implied vols for a mixed scenario (Sec. 2.3,
    extended to two components): adaptive quadrature for mixed standard
    Bergomi (proportional exponential kernels collapse to a scalar Gaussian
    factor), randomized QMC with the mixed log-normal proxy as control
    variate for mixed rough Bergomi (power-law kernels drive a genuine
    Gaussian process)."""
    kernels = build_scenario_kernels(scenario)
    components: list[Component] = [
        (kernels.amplitude1, kernels.lam),
        (kernels.amplitude2, 1.0 - kernels.lam),
    ]
    gfm = _gfm_for_scenario(kernels, T)
    if not kernels.is_rough:
        return reference_price_quadrature(components, gfm, config.XI0, config.DELTA, T, log_moneyness)
    proxy_call_fn, proxy_futures_fn = geometric_proxy_price_functions(
        proxy, config.N_GL_MIXED_APPROX, config.GAUSS_INTEGRAL_B
    )
    return reference_price_qmc(
        components, gfm, config.XI0, config.DELTA, T, log_moneyness,
        proxy_call_fn, proxy_futures_fn,
        config.N_QMC_SCRAMBLES, config.N_QMC_PATHS_LOG2, config.RANDOM_SEED,
    )
