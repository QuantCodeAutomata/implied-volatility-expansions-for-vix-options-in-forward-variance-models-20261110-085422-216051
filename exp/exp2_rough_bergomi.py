"""Experiment 2: single-kernel VIX implied volatility expansion in the rough
Bergomi model with power-law kernel K^u(t) = eta*(u-t)^(H-1/2).

Reproduces the paper's Sec. 2.3 numerical test for the rough-kernel case: the
explicit implied volatility expansion (Theorem 2, Eq. 21) and the BS-inversion
*Approx* (Theorem 1) are compared against a randomized quasi-Monte Carlo
reference that samples the Gaussian driver jointly over a quadrature grid in
the delivery date u, using the geometric (log-normal) proxy as a control
variate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import config
from src.kernel_model import build_gaussian_factor_model, power_law_kernel
from src.reference import reference_price_qmc
from src.runtime import median_runtime_ms
from src.single_kernel import (
    SingleKernelModel,
    approx_implied_vol,
    compute_proxy,
    expansion_implied_vol,
    geometric_proxy_call_price,
    geometric_proxy_futures_price,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_one_parameter_set(H: float, eta: float, reference_futures: tuple) -> dict:
    kernel = power_law_kernel(eta, H)
    rows = []
    futures_by_T = {}
    halfwidth_by_T = {}
    runtimes = {"Expansion": [], "Approx": [], "Reference": []}

    for T_idx, T in enumerate(config.MATURITIES):
        model = SingleKernelModel(kernel, T, config.DELTA, config.XI0)
        proxy = compute_proxy(model, n_t=config.N_GL_PROXY, n_u=config.N_GL_PROXY)

        log_moneyness = config.LOG_MONEYNESS_GRID
        gfm = build_gaussian_factor_model(kernel, T, config.DELTA, config.N_GL_REF_U_ROUGH)
        assert not gfm.is_scalar_factor, "power-law kernel must drive a genuine Gaussian process"

        proxy_call_fn = lambda lm: geometric_proxy_call_price(proxy, lm, T)
        proxy_futures_fn = lambda: geometric_proxy_futures_price(proxy)

        ref = reference_price_qmc(
            [(1.0, 1.0)], gfm, config.XI0, config.DELTA, T, log_moneyness,
            proxy_call_fn, proxy_futures_fn,
            config.N_QMC_SCRAMBLES, config.N_QMC_PATHS_LOG2, config.RANDOM_SEED,
        )
        futures_by_T[T] = ref.futures
        halfwidth_by_T[T] = float(np.max(ref.iv_halfwidth))

        k = np.log(ref.futures) + log_moneyness
        expansion_iv = expansion_implied_vol(proxy, k, T)
        approx_iv = approx_implied_vol(proxy, k, T)

        rel_err_expansion = (expansion_iv - ref.iv) / ref.iv
        rel_err_approx = (approx_iv - ref.iv) / ref.iv

        for i, k_rel in enumerate(log_moneyness):
            rows.append({
                "T_months": round(T * 12),
                "log_moneyness": float(k_rel),
                "iv_expansion": float(expansion_iv[i]),
                "iv_approx": float(approx_iv[i]),
                "iv_reference": float(ref.iv[i]),
                "iv_reference_halfwidth_pct": float(ref.iv_halfwidth[i] * 100),
                "rel_err_expansion_pct": float(rel_err_expansion[i] * 100),
                "rel_err_approx_pct": float(rel_err_approx[i] * 100),
            })

        if T_idx == 1:  # T = 3 months, per the paper's runtime protocol (Sec. 2.3)
            runtimes["Expansion"].append(median_runtime_ms(
                lambda: expansion_implied_vol(proxy, k, T), config.N_RUNTIME_RUNS))
            runtimes["Approx"].append(median_runtime_ms(
                lambda: approx_implied_vol(proxy, k, T), config.N_RUNTIME_RUNS))
            runtimes["Reference"].append(median_runtime_ms(
                lambda: reference_price_qmc(
                    [(1.0, 1.0)], gfm, config.XI0, config.DELTA, T, k,
                    proxy_call_fn, proxy_futures_fn,
                    config.N_QMC_SCRAMBLES, config.N_QMC_PATHS_LOG2, config.RANDOM_SEED,
                ), config.N_RUNTIME_RUNS_REFERENCE))

    return {
        "H": H, "eta": eta,
        "rows": rows,
        "futures": {str(round(T * 12)): v for T, v in futures_by_T.items()},
        "futures_paper_reference": {
            str(m): f for m, f in zip(config.MATURITIES_MONTHS, reference_futures)
        },
        "iv_reference_max_halfwidth_pct": {
            str(round(T * 12)): v for T, v in halfwidth_by_T.items()
        },
        "runtime_ms": {name: (vals[0] if vals else None) for name, vals in runtimes.items()},
        "max_abs_rel_err_expansion_pct": max(abs(r["rel_err_expansion_pct"]) for r in rows),
        "max_abs_rel_err_approx_pct": max(abs(r["rel_err_approx_pct"]) for r in rows),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = [
        run_one_parameter_set(p["H"], p["eta"], config.EXP2_REFERENCE_FUTURES[i])
        for i, p in enumerate(config.EXP2_PARAMS)
    ]
    out_path = RESULTS_DIR / "exp2_rough_bergomi.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    for res in results:
        print(
            f"H={res['H']}, eta={res['eta']}: "
            f"max |rel err| Expansion = {res['max_abs_rel_err_expansion_pct']:.6f}%, "
            f"Approx = {res['max_abs_rel_err_approx_pct']:.6f}%, "
            f"futures = {res['futures']}"
        )


if __name__ == "__main__":
    main()
