"""Experiment 5: accuracy of the Nth-order Hermite expansion of VIX call
prices (Proposition 3, Eq. 47), the intermediate price-level object on which
the implied-volatility expansion of Experiment 3 is built.

Reproduces the paper's Sec. 3.2.1 validation: the Hermite call price
C_{P,N}^j is compared against the reference VIX call price (Sec. 2.3) and the
weak-approximation price C_P=C_{P,0}+C_{P,1} (Theorem 3, Eq. 36) built from
the same Gaussian integrals as *Approx*, for the component j with the lower
proxy volatility, across the four mixed-kernel scenarios of Table 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import config
from src.mixed_kernel import (
    approx_gaussian_integrals,
    hermite_call_price,
    hermite_weights,
    proposition2_factors,
    solve_A,
)
from src.mixed_scenarios import lower_vol_component, mixed_proxy_for_scenario, reference_for_scenario

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_one_scenario(scenario: dict) -> dict:
    sid = scenario["id"]
    log_moneyness = config.LOG_MONEYNESS_GRID
    rows = []

    for T_idx, T in enumerate(config.MATURITIES):
        proxy = mixed_proxy_for_scenario(scenario, T)
        j = lower_vol_component(proxy)

        ref = reference_for_scenario(scenario, proxy, T, log_moneyness)
        k = np.log(ref.futures) + log_moneyness

        A = np.array([solve_A(proxy, kk) for kk in k])
        # Theorem 3 weak-approximation price C_P = C_{P,0} + C_{P,1} (Eq. 36),
        # the same price-level object whose BS inversion gives *Approx*.
        C0, C1, F0, F1 = approx_gaussian_integrals(proxy, k, A, config.N_GL_MIXED_APPROX, config.GAUSS_INTEGRAL_B)
        approx_price = C0 + C1

        N_table = config.MIXED_HERMITE_N[sid][j + 1][T_idx]
        factors0 = proposition2_factors(proxy, j, 0.0)
        omega0, omega1 = hermite_weights(
            proxy, j, factors0, N_table, config.N_GL_HERMITE, config.HERMITE_NODE_BOUND
        )

        hermite_price = np.empty_like(k)
        for idx, (kk, a) in enumerate(zip(k, A)):
            factors = proposition2_factors(proxy, j, a)
            hermite_price[idx] = hermite_call_price(proxy, j, kk, a, factors, omega0, omega1, N_table)

        rel_err_hermite = (hermite_price - ref.call_prices) / ref.call_prices
        rel_err_approx = (approx_price - ref.call_prices) / ref.call_prices
        hermite_vs_approx = (hermite_price - approx_price) / approx_price

        for i, k_rel in enumerate(log_moneyness):
            rows.append({
                "T_months": round(T * 12),
                "component_j": j + 1,
                "N_hermite": N_table,
                "log_moneyness": float(k_rel),
                "price_hermite": float(hermite_price[i]),
                "price_approx": float(approx_price[i]),
                "price_reference": float(ref.call_prices[i]),
                "rel_err_hermite_pct": float(rel_err_hermite[i] * 100),
                "rel_err_approx_pct": float(rel_err_approx[i] * 100),
                "hermite_vs_approx_pct": float(hermite_vs_approx[i] * 100),
            })

    return {
        "scenario_id": sid,
        "model": scenario["model"],
        "rows": rows,
        "max_abs_rel_err_hermite_pct": max(abs(r["rel_err_hermite_pct"]) for r in rows),
        "max_abs_rel_err_approx_pct": max(abs(r["rel_err_approx_pct"]) for r in rows),
        "max_abs_hermite_vs_approx_pct": max(abs(r["hermite_vs_approx_pct"]) for r in rows),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = [run_one_scenario(scenario) for scenario in config.MIXED_SCENARIOS]

    out_path = RESULTS_DIR / "exp5_hermite_call_price_accuracy.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    for res in results:
        print(
            f"Scenario {res['scenario_id']} ({res['model']}): "
            f"max |rel err| Hermite = {res['max_abs_rel_err_hermite_pct']:.4f}%, "
            f"Approx = {res['max_abs_rel_err_approx_pct']:.4f}%, "
            f"Hermite-vs-Approx = {res['max_abs_hermite_vs_approx_pct']:.4f}%"
        )


if __name__ == "__main__":
    main()
