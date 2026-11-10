"""Experiment 4: mixed-kernel implied volatility expansion with implicit
zeroth-order term (Theorem 4, Eq. 62), where the zeroth-order smile Sigma_0(k)
is the Black-Scholes implied volatility of the mixed log-normal proxy
(Eq. 64) and the first-order correction Sigma_1(k) is explicit (Eq. 65).

Reproduces the paper's Sec. 3.3.2 numerical test across the four mixed-kernel
scenarios of Table 3: the full expansion Sigma_0+Sigma_1, the proxy alone
Sigma_0, and the weak-approximation BS-inversion *Approx* (Theorem 3) are
compared against the same references used in Experiment 3 (adaptive
quadrature for mixed standard Bergomi, randomized QMC for mixed rough
Bergomi, Sec. 2.3).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import config
from src.mixed_kernel import approx_implied_vol, implicit_proxy_expansion_iv
from src.mixed_scenarios import mixed_proxy_for_scenario, reference_for_scenario
from src.runtime import median_runtime_ms

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_one_scenario(scenario: dict) -> dict:
    sid = scenario["id"]
    log_moneyness = config.LOG_MONEYNESS_GRID
    rows = []
    futures_by_T = {}
    runtimes = {"Expansion": [], "Approx": [], "Reference": []}

    for T_idx, T in enumerate(config.MATURITIES):
        proxy = mixed_proxy_for_scenario(scenario, T)

        ref = reference_for_scenario(scenario, proxy, T, log_moneyness)
        futures_by_T[T] = ref.futures
        k = np.log(ref.futures) + log_moneyness

        approx_iv = approx_implied_vol(proxy, k, T, config.N_GL_MIXED_APPROX, config.GAUSS_INTEGRAL_B)
        expansion_iv, proxy_iv, F_P0 = implicit_proxy_expansion_iv(
            proxy, k, T, config.N_GL_MIXED_APPROX, config.GAUSS_INTEGRAL_B
        )

        rel_err_expansion = (expansion_iv - ref.iv) / ref.iv
        rel_err_approx = (approx_iv - ref.iv) / ref.iv
        rel_err_proxy = (proxy_iv - ref.iv) / ref.iv

        for i, k_rel in enumerate(log_moneyness):
            rows.append({
                "T_months": round(T * 12),
                "log_moneyness": float(k_rel),
                "iv_expansion": float(expansion_iv[i]),
                "iv_proxy_sigma0": float(proxy_iv[i]),
                "iv_approx": float(approx_iv[i]),
                "iv_reference": float(ref.iv[i]),
                "rel_err_expansion_pct": float(rel_err_expansion[i] * 100),
                "rel_err_approx_pct": float(rel_err_approx[i] * 100),
                "rel_err_proxy_pct": float(rel_err_proxy[i] * 100),
            })

        if T_idx == 1:  # T = 3 months, per the paper's runtime protocol (Sec. 2.3)
            runtimes["Expansion"].append(median_runtime_ms(
                lambda: implicit_proxy_expansion_iv(
                    proxy, k, T, config.N_GL_MIXED_APPROX, config.GAUSS_INTEGRAL_B
                ), config.N_RUNTIME_RUNS))
            runtimes["Approx"].append(median_runtime_ms(
                lambda: approx_implied_vol(proxy, k, T, config.N_GL_MIXED_APPROX, config.GAUSS_INTEGRAL_B),
                config.N_RUNTIME_RUNS))
            runtimes["Reference"].append(median_runtime_ms(
                lambda: reference_for_scenario(scenario, proxy, T, k),
                config.N_RUNTIME_RUNS_REFERENCE))

    return {
        "scenario_id": sid,
        "model": scenario["model"],
        "params": {kk: v for kk, v in scenario.items() if kk not in ("id", "model")},
        "rows": rows,
        "futures": {str(round(T * 12)): v for T, v in futures_by_T.items()},
        "futures_paper_reference": {
            str(m): f for m, f in zip(config.MATURITIES_MONTHS, config.MIXED_REFERENCE_FUTURES[sid])
        },
        "runtime_ms": {name: (vals[0] if vals else None) for name, vals in runtimes.items()},
        "max_abs_rel_err_expansion_pct": max(abs(r["rel_err_expansion_pct"]) for r in rows),
        "max_abs_rel_err_approx_pct": max(abs(r["rel_err_approx_pct"]) for r in rows),
        "max_abs_rel_err_proxy_pct": max(abs(r["rel_err_proxy_pct"]) for r in rows),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = [run_one_scenario(scenario) for scenario in config.MIXED_SCENARIOS]

    out_path = RESULTS_DIR / "exp4_mixed_implicit_proxy.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    for res in results:
        print(
            f"Scenario {res['scenario_id']} ({res['model']}): "
            f"max |rel err| Expansion = {res['max_abs_rel_err_expansion_pct']:.4f}%, "
            f"Approx = {res['max_abs_rel_err_approx_pct']:.4f}%, "
            f"Proxy alone = {res['max_abs_rel_err_proxy_pct']:.4f}%"
        )


if __name__ == "__main__":
    main()
