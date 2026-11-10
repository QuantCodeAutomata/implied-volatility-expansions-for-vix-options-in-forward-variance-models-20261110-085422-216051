"""Experiment 6: extended implied volatility expansion (Remark 9, Eq. 61) that
absorbs the strike part of the shift residual via a Taylor expansion in
log-strike, instead of discarding it as in the standard expansion (Eq. 60).

Reproduces the paper's Sec. 3.2.2 comparison between the standard expansion
sigma_hat_{theta,N}^j (Eq. 60, Experiment 3) and its extended variant
sigma_hat_N^{j,ext} (Eq. 61) against the same references (Sec. 2.3), for the
component j with the lower proxy volatility, across the four mixed-kernel
scenarios of Table 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import config
from src.mixed_kernel import explicit_expansion_iv
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

        N_table = config.MIXED_HERMITE_N[sid][j + 1][T_idx]
        standard_iv, _ = explicit_expansion_iv(
            proxy, k, T, j, config.N_GL_HERMITE, config.HERMITE_NODE_BOUND,
            config.HERMITE_EPS_TOL, config.HERMITE_N_MAX, extended=False, N_override=N_table,
        )
        extended_iv, _ = explicit_expansion_iv(
            proxy, k, T, j, config.N_GL_HERMITE, config.HERMITE_NODE_BOUND,
            config.HERMITE_EPS_TOL, config.HERMITE_N_MAX, extended=True, N_override=N_table,
        )

        rel_err_standard = (standard_iv - ref.iv) / ref.iv
        rel_err_extended = (extended_iv - ref.iv) / ref.iv

        for i, k_rel in enumerate(log_moneyness):
            rows.append({
                "T_months": round(T * 12),
                "component_j": j + 1,
                "N_hermite": N_table,
                "log_moneyness": float(k_rel),
                "iv_standard_eq60": float(standard_iv[i]),
                "iv_extended_eq61": float(extended_iv[i]),
                "iv_reference": float(ref.iv[i]),
                "rel_err_standard_pct": float(rel_err_standard[i] * 100),
                "rel_err_extended_pct": float(rel_err_extended[i] * 100),
            })

    return {
        "scenario_id": sid,
        "model": scenario["model"],
        "rows": rows,
        "max_abs_rel_err_standard_pct": max(abs(r["rel_err_standard_pct"]) for r in rows),
        "max_abs_rel_err_extended_pct": max(abs(r["rel_err_extended_pct"]) for r in rows),
        "n_points_above_100pct_extended": sum(1 for r in rows if abs(r["rel_err_extended_pct"]) > 100.0),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = [run_one_scenario(scenario) for scenario in config.MIXED_SCENARIOS]

    out_path = RESULTS_DIR / "exp6_mixed_extended_expansion.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    for res in results:
        print(
            f"Scenario {res['scenario_id']} ({res['model']}): "
            f"max |rel err| standard(Eq.60) = {res['max_abs_rel_err_standard_pct']:.2f}%, "
            f"extended(Eq.61) = {res['max_abs_rel_err_extended_pct']:.2f}% "
            f"({res['n_points_above_100pct_extended']} points > 100%)"
        )


if __name__ == "__main__":
    main()
