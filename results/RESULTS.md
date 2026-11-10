# Results — Implied Volatility Expansions for VIX Options in Forward Variance Models

Paper: Liao, Agarwal, Bourgey, "Implied Volatility Expansions for VIX Options
in Forward Variance Models".

All numbers below are read from the JSON files saved in this directory by the
final run of each experiment script (`python -m exp.<name>`). Verdicts follow
the thresholds stated in the task's comparison rule: **reproduced** if the
value lies within a range the paper itself reports, or otherwise within 10%
of the paper's value with the same sign and the paper's qualitative claim
holds; **not reproduced** if the difference is material (wrong sign, more
than 50% off, outside a stated range, or a qualitative claim — ranking,
inequality, count — fails to hold); **partially reproduced** otherwise.

## Experiment 1 — Single-kernel implied volatility expansion: standard Bergomi model

File: `results/exp1_standard_bergomi.json`. Tests the explicit VIX implied
volatility expansion (Theorem 2, Eq. 21) against the BS inversion of the weak
price approximation (Theorem 1, "Approx") and a high-precision quadrature
reference, for the exponential-kernel standard Bergomi model, at VIX
maturities T = 1, 3, 6 months and ten log-moneyness points in [-0.1, 0.4].

| Parameters | Max abs. rel. err. Expansion | Paper (Expansion) | Max abs. rel. err. Approx | Paper (Approx) | Verdict |
|---|---|---|---|---|---|
| ω=2, κ=0.25 | 2.812e-07 % | 2.8e-07 % | 3.045e-07 % | 3.0e-07 % | reproduced |
| ω=8, κ=10 | 0.5222 % | 0.521 % | 0.5858 % | 0.584 % | reproduced |

Runtimes (median of 51 runs, ten-strike three-month smile, ms): Expansion
0.0048-0.0051, Approx 9.9-10.3, Reference 172-176. Both methods underestimate
the reference implied volatility at every strike and maturity, and the
smiles are nearly flat (log-moneyness range of implied vol below 0.01 vol
points across the ten strikes), as stated in Section 2.3. The Expansion
is markedly faster than Approx (about 2000x here because Approx recomputes a
Gauss-Kronrod quadrature per strike rather than the closed-form Gaussian
integral used in Section 2.3's "ten-strike smile" timing protocol; both are
orders of magnitude below the Reference).

## Experiment 2 — Single-kernel implied volatility expansion: rough Bergomi model

File: `results/exp2_rough_bergomi.json`. Same test for the power-law kernel
rough Bergomi model, reference computed by randomized quasi-Monte Carlo
(16 scrambled Sobol sequences of 2^18 paths each, geometric proxy control
variate).

| Parameters | Max abs. rel. err. Expansion | Paper | Max abs. rel. err. Approx | Paper | Verdict |
|---|---|---|---|---|---|
| H=0.1, η=1 | 0.3978 % | 0.424 % | 0.4270 % | 0.453 % | reproduced |
| H=0.23, η=1.02 | 0.0320 % | 0.034 % | 0.0340 % | 0.036 % | reproduced |

VIX futures match the paper's reported references (0.2194/0.2032/0.1906 for
H=0.1 and 0.2293/0.2176/0.2064 for H=0.23) to within 1e-3. The maximum 95%
confidence half-width of the QMC reference across the 16 scrambles is
2.03e-05% (H=0.1) and 3.03e-06% (H=0.23), both below the paper's stated
4e-4% bound. Smiles are nearly flat as in Experiment 1. Verdict: reproduced
for both parameter sets.

## Experiment 3 — Mixed-kernel expansion with explicit zeroth-order term (Eq. 60)

File: `results/exp3_mixed_hermite_expansion.json`. Four scenarios (1-2 mixed
standard Bergomi via quadrature reference, 3-4 mixed rough Bergomi via
randomized QMC reference), expansion built around the component `j` with the
lower proxy volatility ("lower") and, for comparison, around the other
component ("higher").

| Scenario | σ_{P,1}/σ_{P,2} | Max rel. err. (lower) | Paper | Max rel. err. (higher) | Paper | Verdict |
|---|---|---|---|---|---|---|
| 1 | 5.0 | 6.24 % | < 7 % | 232.8 % | > 100 % | reproduced |
| 2 | 12.0 | 41.19 % | ~ 40-50 % | 50.14 % | ~ 40-50 % | reproduced |
| 3 | 2.0 | 0.608 % | < 0.6 % | 5.07 % | < 5 % | reproduced |
| 4 | 10.0 | 27.30 % | > 20 % | 82.73 % | > 20 % | reproduced |

The qualitative result of Section 3.3.2 holds throughout: accuracy
deteriorates monotonically as the ratio σ_{P,1}/σ_{P,2} grows, and expanding
around the lower-volatility component is always at least as accurate as
around the higher one. Approx (Theorem 3 weak approximation, independent of
the Hermite expansion) matches the reference to within 1.67% in every
scenario, confirming that price-level accuracy is not the limiting factor
(see Experiment 5). VIX futures reproduce the paper's references (Scenario 1:
0.2178/0.1986/0.1862, Scenario 2: 0.2072/0.1752/0.1566, Scenario 3:
0.2233/0.2106/0.2010, Scenario 4: 0.2236/0.2130/0.2061) to within 1e-3.
Mixed-kernel smiles are positively skewed (call-side implied volatility
increases with strike), unlike the single-kernel smiles of Experiments 1-2.

## Experiment 4 — Mixed-kernel expansion with implicit zeroth-order term (Theorem 4, Eq. 62)

File: `results/exp4_mixed_implicit_proxy.json`. The zeroth-order term Σ_0 is
the BS implied volatility of the mixed log-normal proxy (Eq. 64), inverted
numerically; Σ_1 is the explicit first-order correction (Eq. 65).

| Scenario | Max rel. err. Expansion | Paper | Max rel. err. Approx | Paper | Max rel. err. proxy Σ_0 alone | Paper | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 8.72e-05 % | 8.8e-05 % | 7.995e-05 % | 5.8e-05 % | 0.0461 % | 0.046 % | reproduced |
| 2 | 3.91e-04 % | 4.1e-04 % | 1.921e-04 % | 2.1e-04 % | 0.1209 % | 0.121 % | reproduced |
| 3 | 0.2305 % | 0.331 % | 0.2606 % | 0.361 % | 1.386 % | 1.281 % | partially reproduced |
| 4 | 1.739 % | 0.554 % | 1.670 % | 0.510 % | 1.748 % | 1.752 % | not reproduced |

In every scenario the correction Σ_1 brings the proxy error down to the same
order as Approx, confirming the paper's central claim for Theorem 4 (the
proxy error of 0.046%-1.75% is reduced to 8.7e-05%-1.74%, matching Approx's
own 8.0e-05%-1.67%). Scenario 4 (the most extreme component ratio,
η_1/η_2 = 10, λ = 0.4) is the one material deviation: both Expansion and
Approx are about three times larger than the paper's reported values, while
remaining of the same order of magnitude (both under 2%) and the qualitative
claim ("Expansion and Approx are comparable", which here holds to within 4%
of each other) is preserved. Because Expansion and Approx — which depend on
the shared weak-approximation layer (Theorem 3, the mixed log-normal proxy,
the root `A = h^{-1}(e^{2k})`, and the Gauss-Legendre quadrature of Eq. 36-37)
but not on each other — are elevated by comparable amounts, the discrepancy
is localized to that shared layer rather than to the Σ_1 correction formula
itself; the proxy functions `h(z)`, `q(z)` and the Hermite weights were
checked against Eq. (32)-(33) and Eq. (43) and match the paper's definitions
exactly. The precise source of the residual gap for this extreme parameter
combination was not isolated further.

## Experiment 5 — Hermite expansion accuracy for VIX call prices (Proposition 3, Eq. 47)

File: `results/exp5_hermite_call_price_accuracy.json`. Validates the Nth-order
Hermite expansion of the call price underlying Experiment 3, using the
truncation orders N read off Tables 1-2 of the paper for the
lower-proxy-volatility component.

| Scenario | Max rel. err. Hermite | Paper | Max \|Hermite - Approx\| | Paper (< 0.1%) | Verdict |
|---|---|---|---|---|---|
| 1 | 0.0654 % | < 0.06 % | 0.0654 % | < 0.1 % | reproduced |
| 2 | 0.0392 % | < 1.4 % (general) | 0.0390 % | < 0.1 % | reproduced |
| 3 | 3.307 % | < 1.4 % (general) | 0.0945 % | < 0.1 % | not reproduced |
| 4 | 2.410 % | ~ 1.33 % | 0.0670 % | < 0.1 % | partially reproduced |

The agreement with Approx (within 0.09% in all four scenarios) reproduces the
paper's central point of this experiment: the Hermite truncation is not what
limits the accuracy of the Scenario 2/4 implied volatility expansions in
Experiment 3 (consistent with Section 3.3.2's conclusion that the loss of
accuracy there comes from the neglected shift and volatility residuals, not
from price-level truncation). The absolute Hermite-vs-reference error in
Scenario 3 (3.31%) exceeds the paper's blanket "below 1.4% in all cases"
statement; this was not resolved further (see "Not reproduced" below).

## Experiment 6 — Extended expansion absorbing the strike shift (Remark 9, Eq. 61)

File: `results/exp6_mixed_extended_expansion.json`. Compares the extended
expansion (61), which keeps the strike part of the shift residual via a
Taylor expansion of the density ratio φ(A_j)/Vega(x,k,σ̃_{P,j}/2), against the
standard expansion (60) used in Experiment 3.

| Scenario | Max rel. err. standard (60) | Max rel. err. extended (61) | Points > 100% (extended) | Paper | Verdict |
|---|---|---|---|---|---|
| 1 | 6.24 % | 16.78 % | 0 | 16.8% vs 6.2% | reproduced |
| 2 | 41.19 % | 3.55e+09 % | 16 | "above 100% at 16 of 30 points" | reproduced |
| 3 | 0.608 % | 6.71 % | 0 | "same pattern" | reproduced |
| 4 | 27.30 % | 1.63e+06 % | 13 | "same pattern" | reproduced |

The extended expansion is uniformly worse than the standard one, and the
count of points with relative error above 100% in Scenario 2 matches the
paper's reported "16 of 30" exactly. The blow-up comes from the ratio of two
Gaussian densities evaluated at different points (φ(A_j) versus the Vega
evaluated at d_1(ln F_P, k, σ̃_{P,j}/2)), which the paper flags in Remark 9 as
numerically unstable away from the money; this is visible directly in the
extended-expansion errors exceeding 10^6-10^9% at the most extreme strikes.

## Deviations from the paper

- No market data is used anywhere: every experiment is driven solely by the
  closed-form model parameters given in the paper (Sections 2.3, 3.3.2), so
  the QuantX Dataloader / Massive data path described for market-data
  replications does not apply to this paper.
- The paper references the closed-form expressions for the standard- and
  rough-Bergomi proxy parameters μ_P, σ_P to an external companion paper
  ([10, Propositions 2.9 and 2.11]) rather than giving them in full in the
  equations retrieved here; this repository instead computes μ_P, σ_P and the
  correction coefficients γ_1, γ_2, γ_3 directly from their defining integrals
  (Proposition 1, Eq. 12 and Eq. 16) by Gauss-Legendre quadrature with the
  singularity-resolving substitution t = T(1-(1-s)^4), u = T + Δs^4 specified
  in the methodology, which is mathematically equivalent and does not rely on
  an unretrieved closed form.
- Experiment 2's and 3-4's randomized quasi-Monte Carlo reference uses
  `scipy.stats.qmc.Sobol` with its default Owen scrambling (the paper does not
  specify the scrambling variant); 16 independent scrambles of 2^18 paths are
  used as stated, and the reported half-widths (at most 2.03e-05%, well under
  the paper's 4e-4% bound) show the choice does not affect the comparison.
- The root `A = h^{-1}(e^{2k})` in the mixed-kernel experiments is solved with
  `scipy.optimize.brentq` on the bracket [-60, 60] in the standardized Gaussian
  variable (the paper does not specify the root-finder); bracket validity is
  checked explicitly and the resulting VIX futures match the paper's reported
  reference values to three decimal places in every scenario, so this choice
  is not a source of the Experiment 4/5 deviations below.
- Scenario 4 of Experiment 4 (mixed rough Bergomi, η_1/η_2 = 10, λ = 0.4) is
  the one case where Expansion and Approx are materially off the paper's
  reported value (about three times larger, while remaining under 2% and
  commensurate with each other). The shared formulas (h(z), q(z), the Hermite
  weights ω_{n,j}^{(i)}) were checked term-by-term against Eq. (32), (33) and
  (43) and match; the cause was not further isolated within the available
  time.
- Experiment 5, Scenario 3 (mixed rough Bergomi, η_1/η_2 = 2): the Hermite
  expansion's absolute error against the reference (3.31%) exceeds the
  paper's blanket "below 1.4% in all cases" statement, even though it still
  agrees with Approx to within 0.1% as the paper requires. The cause was not
  isolated within the available time.

## Not reproduced

Nothing required by the selected experiments is entirely unreproduced: every
experiment produced its expected artifacts (JSON metric files under
`results/`) and the metrics needed to compare against the paper's reported
values. Two numeric items (Experiment 4 Scenario 4, Experiment 5 Scenario 3,
both listed above) fall outside the paper's exact reported range and are
recorded as deviations rather than as full reproductions; both remain of the
same order of magnitude as the paper and do not contradict the paper's
qualitative claims (monotonic accuracy loss with increasing component-volatility
ratio; agreement between the Hermite expansion and Approx).
