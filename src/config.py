"""Parameters for the VIX implied-volatility expansion experiments.

All values are module-level constants; each is annotated with its meaning,
unit, and source (paper section/equation, or "not specified in the paper").
Reference: Liao, Agarwal, Bourgey, "Implied Volatility Expansions for VIX
Options in Forward Variance Models".
"""

import numpy as np

# ---------------------------------------------------------------------------
# Common VIX / smile grid (Sec. 2.3, used throughout Sec. 2-3)
# ---------------------------------------------------------------------------
XI0 = 0.24 ** 2  # flat initial forward variance xi_0^u, Sec. 2.3
DELTA = 30 / 365  # VIX tenor Delta (30-day window), Sec. 2.3
MATURITIES_MONTHS = (1, 3, 6)  # VIX maturities T in months, Sec. 2.3
MATURITIES = tuple(m / 12 for m in MATURITIES_MONTHS)  # T in years
N_STRIKES = 10  # number of log-moneyness points, Sec. 2.3
LOG_MONEYNESS_GRID = np.linspace(-0.1, 0.4, N_STRIKES)  # ln(e^k/F) range, Sec. 2.3

# ---------------------------------------------------------------------------
# Quadrature settings for the correction coefficients mu_P, sigma_P, gamma_i
# (Prop. 1 Eq. 12, Theorem 1 Eq. 16); same substitution used for mixed kernels.
# ---------------------------------------------------------------------------
N_GL_PROXY = 30  # Gauss-Legendre nodes for (mu_P, sigma_P, gamma_i), Sec. 2.3
# substitution t = T(1-(1-s)^4), u = T + Delta*s^4, s in [0,1], resolves the
# kernel singularity at u = t = T (Sec. 2.3).

# ---------------------------------------------------------------------------
# Reference methods (Sec. 2.3)
# ---------------------------------------------------------------------------
N_GL_REF_U_STANDARD = 256  # GL nodes over delivery dates for quadrature reference, Sec. 2.3
N_QMC_SCRAMBLES = 16  # independent scrambled Sobol sequences, Sec. 2.3
N_QMC_PATHS_LOG2 = 18  # 2^18 paths per scramble, Sec. 2.3
N_GL_REF_U_ROUGH = 64  # GL nodes over delivery dates for the QMC Gaussian factor;
# the paper specifies the quartic substitution but not the node count for the
# rough-kernel reference grid (not specified in the paper); chosen for an
# accurate trade-off between discretization bias and runtime.

# ---------------------------------------------------------------------------
# Hermite expansion (Sec. 3.2), mixed-kernel models
# ---------------------------------------------------------------------------
N_GL_HERMITE = 4000  # GL nodes for Hermite weights omega_{n,j}^{(i)}, Eq. 43
HERMITE_NODE_BOUND = 40.0  # integration range [-40, 40] for Hermite weights, Sec. 3.2.1
HERMITE_EPS_TOL = 1e-6  # MSE tolerance for selecting truncation order N, Sec. 3.2.1
HERMITE_N_MAX = 80  # safety cap on the Hermite truncation order search (not specified in the paper)

# ---------------------------------------------------------------------------
# Mixed-kernel Gaussian integrals for Approx / Theorem 4 (Sec. 3.3.1, Eq. 36-37)
# ---------------------------------------------------------------------------
N_GL_MIXED_APPROX = 128  # GL nodes for the one-dimensional Gaussian integrals, Sec. 3.3.1
GAUSS_INTEGRAL_B = 20.0  # upper truncation bound b for [A,b] / [-b,b]; paper requires b >= 14, Sec. 3.3.1

# ---------------------------------------------------------------------------
# Runtime benchmarking protocol (Sec. 2.3)
# ---------------------------------------------------------------------------
N_RUNTIME_RUNS = 51  # median of 51 runs for Expansion/Approx, Sec. 2.3
N_RUNTIME_RUNS_REFERENCE = 5  # median of 5 runs for reference methods, Sec. 2.3

RANDOM_SEED = 2026  # seed for Sobol scrambling (not specified in the paper)

# ---------------------------------------------------------------------------
# Experiment 1: standard Bergomi, single exponential kernel K^u(t)=w*exp(-k(u-t))
# ---------------------------------------------------------------------------
EXP1_PARAMS = (
    {"omega": 2.0, "kappa": 0.25},  # [13, Figure 3], Sec. 2.3
    {"omega": 8.0, "kappa": 10.0},  # [13, Figure 3], Sec. 2.3
)

# ---------------------------------------------------------------------------
# Experiment 2: rough Bergomi, single power-law kernel K^u(t)=eta*(u-t)^(H-1/2)
# ---------------------------------------------------------------------------
EXP2_PARAMS = (
    {"H": 0.1, "eta": 1.0},  # [10, Section 2], Sec. 2.3
    {"H": 0.23, "eta": 1.02},  # [9], eta*sqrt(2H) convention used in [9], Sec. 2.3
)
# Reference VIX futures reported in the paper for the H=0.1 / H=0.23 cases,
# for the 1-, 3-, 6-month maturities (Sec. 2.3):
EXP2_REFERENCE_FUTURES = {
    0: (0.2194, 0.2032, 0.1906),  # H=0.1, eta=1
    1: (0.2293, 0.2176, 0.2064),  # H=0.23, eta=1.02
}

# ---------------------------------------------------------------------------
# Experiments 3-6: mixed-kernel models, four scenarios (Table 3, Sec. 3.2-3.3)
# Mixed standard Bergomi: proportional exponential kernels, common kappa.
# Mixed rough Bergomi: proportional power-law kernels, common H.
# ---------------------------------------------------------------------------
MIXED_SCENARIOS = (
    {"id": 1, "model": "standard", "omega1": 10.0, "omega2": 2.0, "lam": 0.2, "kappa": 1.0},
    {"id": 2, "model": "standard", "omega1": 0.5, "omega2": 6.0, "lam": 0.3, "kappa": 1.0},
    {"id": 3, "model": "rough", "eta1": 1.4, "eta2": 0.7, "lam": 0.3, "H": 0.1},
    {"id": 4, "model": "rough", "eta1": 2.0, "eta2": 0.2, "lam": 0.4, "H": 0.1},
)

# Hermite truncation orders N selected offline by the paper's MSE criterion,
# Tables 1-2, reported per scenario / maturity / component j in {1, 2}.
MIXED_HERMITE_N = {
    1: {1: (10, 22, 36), 2: (10, 14, 17)},
    2: {1: (7, 12, 17), 2: (7, 13, 19)},
    3: {1: (4, 5, 6), 2: (4, 4, 4)},
    4: {1: (7, 11, 14), 2: (8, 9, 12)},
}

# Reference VIX futures for the 1-, 3-, 6-month maturities, Sec. 3.2.1/3.3.1
MIXED_REFERENCE_FUTURES = {
    1: (0.2178, 0.1986, 0.1862),
    2: (0.2072, 0.1752, 0.1566),
    3: (0.2233, 0.2106, 0.2010),
    4: (0.2236, 0.2130, 0.2061),
}

# ---------------------------------------------------------------------------
# Execution-time reduction for the full-grid robustness pass of experiments
# 3-6 (quarterly rerun across all N truncation orders and theta choices): the
# main first-order runs always use the paper's full grid above; only optional
# `--full` sensitivity passes are scaled down by default (not a reduction of
# any paper-specified quantity).
# ---------------------------------------------------------------------------
RUN_FULL_QMC_PAPER = True  # use the paper's 2^18 paths x 16 scrambles when True
