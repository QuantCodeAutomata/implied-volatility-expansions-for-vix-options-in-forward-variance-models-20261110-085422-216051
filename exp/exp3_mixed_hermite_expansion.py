"""Experiment 3: mixed-kernel implied volatility expansion with explicit
zeroth-order term (Eq. 60), built from the Hermite expansion of the
non-Gaussian factors in the mixed-kernel price approximation (Sec. 3.2).

Reproduces the paper's four mixed-kernel scenarios (Table 3): mixed standard
Bergomi (Scenarios 1-2) and mixed rough Bergomi (Scenarios 3-4). For each
scenario, maturity, and choice of component j in {lower, higher} proxy
volatility, the explicit expansion sigma_hat_{theta,N}^j (Eq. 60) is compared
against the reference implied volatility (adaptive quadrature for mixed
standard Bergomi, randomized QMC for mixed rough Bergomi, Sec. 2.3) and the
weak-approximation BS-inversion *Approx* (Theorem 3).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import config
from src.mixed_kernel import approx_implied_vol, explicit_expansion_iv
from src.mixed_scenarios import (
    build_scenario_kernels,
    lower_vol_component,
    mixed_proxy_for_scenario,
    reference_for_scenario,
)
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
        j_lower = lower_vol_component(proxy)

        ref = reference_for_scenario(scenario, proxy, T, log_moneyness)
        futures_by_T[T] = ref.futures
        k = np.log(ref.futures) + log_moneyness

        approx_iv = approx_implied_vol(proxy, k, T, config.N_GL_MIXED_APPROX, config.GAUSS_INTEGRAL_B)

        for j in (0, 1):
            label = "lower" if j == j_lower else "higher"
            N_table = config.MIXED_HERMITE_N[sid][j + 1][T_idx]
            expansion_iv, N_used = explicit_expansion_iv(
                proxy, k, T, j,
                config.N_GL_HERMITE, config.HERMITE_NODE_BOUND,
                config.HERMITE_EPS_TOL, config.HERMITE_N_MAX,
                extended=False, N_override=N_table,
            )
            rel_err_expansion = (expansion_iv - ref.iv) / ref.iv
            rel_err_approx = (approx_iv - ref.iv) / ref.iv

            for i, k_rel in enumerate(log_moneyness):
                rows.append({
                    "T_months": round(T * 12),
                    "component_j": j + 1,
                    "component_label": label,
                    "N_hermite": int(N_used[i]),
                    "log_moneyness": float(k_rel),
                    "iv_expansion": float(expansion_iv[i]),
                    "iv_approx": float(approx_iv[i]),
                    "iv_reference": float(ref.iv[i]),
                    "rel_err_expansion_pct": float(rel_err_expansion[i] * 100),
                    "rel_err_approx_pct": float(rel_err_approx[i] * 100),
                })

        if T_idx == 1:  # T = 3 months, per the paper's runtime protocol (Sec. 2.3)
            N_table_lower = config.MIXED_HERMITE_N[sid][j_lower + 1][T_idx]
            runtimes["Expansion"].append(median_runtime_ms(
                lambda: explicit_expansion_iv(
                    proxy, k, T, j_lower, config.N_GL_HERMITE, config.HERMITE_NODE_BOUND,
                    config.HERMITE_EPS_TOL, config.HERMITE_N_MAX,
                    extended=False, N_override=N_table_lower,
                ), config.N_RUNTIME_RUNS))
            runtimes["Approx"].append(median_runtime_ms(
                lambda: approx_implied_vol(proxy, k, T, config.N_GL_MIXED_APPROX, config.GAUSS_INTEGRAL_B),
                config.N_RUNTIME_RUNS))
            runtimes["Reference"].append(median_runtime_ms(
                lambda: reference_for_scenario(scenario, proxy, T, k),
                config.N_RUNTIME_RUNS_REFERENCE))

    max_err_by_label = {}
    for label in ("lower", "higher"):
        errs = [abs(r["rel_err_expansion_pct"]) for r in rows if r["component_label"] == label]
        max_err_by_label[label] = max(errs) if errs else None

    return {
        "scenario_id": sid,
        "model": scenario["model"],
        "params": {k: v for k, v in scenario.items() if k not in ("id", "model")},
        "sigma_ratio": None,  # filled below once the T=3M proxy is available
        "rows": rows,
        "futures": {str(round(T * 12)): v for T, v in futures_by_T.items()},
        "runtime_ms": {name: (vals[0] if vals else None) for name, vals in runtimes.items()},
        "max_abs_rel_err_expansion_pct": max_err_by_label,
        "max_abs_rel_err_approx_pct": max(abs(r["rel_err_approx_pct"]) for r in rows),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = []
    for scenario in config.MIXED_SCENARIOS:
        res = run_one_scenario(scenario)
        proxy_3m = mixed_proxy_for_scenario(scenario, config.MATURITIES[1])
        res["sigma_ratio"] = float(max(proxy_3m.sigma) / min(proxy_3m.sigma))
        results.append(res)

    out_path = RESULTS_DIR / "exp3_mixed_hermite_expansion.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    for res in results:
        print(
            f"Scenario {res['scenario_id']} ({res['model']}), sigma ratio="
            f"{res['sigma_ratio']:.2f}: max |rel err| Expansion = "
            f"{res['max_abs_rel_err_expansion_pct']}, Approx = "
            f"{res['max_abs_rel_err_approx_pct']:.4f}%"
        )


if __name__ == "__main__":
    main()
