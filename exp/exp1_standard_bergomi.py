"""Experiment 1: single-kernel VIX implied volatility expansion in the
standard Bergomi model with exponential kernel K^u(t) = omega*exp(-kappa*(u-t)).

Reproduces the paper's Sec. 2.3 numerical test: the explicit implied
volatility expansion (Theorem 2, Eq. 21) is compared against the BS-inversion
of the weak price approximation (*Approx*, Theorem 1) and a high-precision
quadrature reference, for two parameter sets, three VIX maturities, and ten
log-moneyness points.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src import config
from src.kernel_model import build_gaussian_factor_model, exponential_kernel
from src.reference import reference_price_quadrature
from src.runtime import median_runtime_ms
from src.single_kernel import (
    SingleKernelModel,
    approx_implied_vol,
    compute_proxy,
    expansion_implied_vol,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_one_parameter_set(omega: float, kappa: float) -> dict:
    kernel = exponential_kernel(omega, kappa)
    rows = []
    futures_by_T = {}
    runtimes = {"Expansion": [], "Approx": [], "Reference": []}

    for T in config.MATURITIES:
        model = SingleKernelModel(kernel, T, config.DELTA, config.XI0)
        proxy = compute_proxy(model)

        log_moneyness = config.LOG_MONEYNESS_GRID
        gfm = build_gaussian_factor_model(kernel, T, config.DELTA, config.N_GL_REF_U_STANDARD)
        ref = reference_price_quadrature(
            [(1.0, 1.0)], gfm, config.XI0, config.DELTA, T, log_moneyness
        )
        futures_by_T[T] = ref.futures

        # Log-moneyness ln(e^k/F) is defined against the reference futures F
        # (Sec. 2.3); absolute log-strikes k are recovered as ln(F) + log-moneyness
        # before evaluating the Expansion (Eq. 21) and Approx (Eq. 18-20).
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
                "rel_err_expansion_pct": float(rel_err_expansion[i] * 100),
                "rel_err_approx_pct": float(rel_err_approx[i] * 100),
            })

        if T == config.MATURITIES[1]:  # T = 3 months, per the paper's runtime protocol
            runtimes["Expansion"].append(median_runtime_ms(
                lambda: expansion_implied_vol(proxy, k, T), config.N_RUNTIME_RUNS))
            runtimes["Approx"].append(median_runtime_ms(
                lambda: approx_implied_vol(proxy, k, T), config.N_RUNTIME_RUNS))
            runtimes["Reference"].append(median_runtime_ms(
                lambda: reference_price_quadrature([(1.0, 1.0)], gfm, config.XI0, config.DELTA, T, k),
                config.N_RUNTIME_RUNS_REFERENCE))

    return {
        "omega": omega, "kappa": kappa,
        "rows": rows,
        "futures": {str(round(T * 12)): v for T, v in futures_by_T.items()},
        "runtime_ms": {name: (vals[0] if vals else None) for name, vals in runtimes.items()},
        "max_abs_rel_err_expansion_pct": max(abs(r["rel_err_expansion_pct"]) for r in rows),
        "max_abs_rel_err_approx_pct": max(abs(r["rel_err_approx_pct"]) for r in rows),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = [run_one_parameter_set(**p) for p in config.EXP1_PARAMS]
    out_path = RESULTS_DIR / "exp1_standard_bergomi.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    for res in results:
        print(
            f"omega={res['omega']}, kappa={res['kappa']}: "
            f"max |rel err| Expansion = {res['max_abs_rel_err_expansion_pct']:.6f}%, "
            f"Approx = {res['max_abs_rel_err_approx_pct']:.6f}%"
        )


if __name__ == "__main__":
    main()
