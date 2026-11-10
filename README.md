# Implied Volatility Expansions for VIX Options in Forward Variance Models

Reproduction of selected numerical experiments from Liao, Agarwal, Bourgey,
"Implied Volatility Expansions for VIX Options in Forward Variance Models".

The paper derives explicit small-time/weak-error expansions for VIX option
implied volatilities in forward variance models driven by a single Gaussian
kernel (standard and rough Bergomi) and by a two-component mixture of such
kernels. This repository implements the model, the proxy and correction
quantities, the Hermite-expansion machinery, the explicit implied-volatility
formulas, the weak price approximation, and the reference pricing methods
(adaptive quadrature and randomized quasi-Monte Carlo), and reproduces six
numerical experiments from the paper. No market data is used: every
experiment is driven purely by the model parameters given in the paper.

## Experiments reproduced

1. **exp1** — single-kernel explicit implied volatility expansion (Theorem 2,
   Eq. 21) in the standard Bergomi model, two parameter sets.
2. **exp2** — same expansion in the rough Bergomi model, two parameter sets.
3. **exp3** — mixed-kernel explicit-zeroth-order expansion (Eq. 60) built from
   the Hermite expansion of the mixed-kernel price approximation, four
   scenarios (mixed standard and mixed rough Bergomi).
4. **exp4** — mixed-kernel expansion with implicit (numerically inverted)
   zeroth-order term (Theorem 4, Eq. 62-65), same four scenarios.
5. **exp5** — accuracy of the Nth-order Hermite expansion of VIX call prices
   (Proposition 3, Eq. 47) underlying Experiment 3.
6. **exp6** — extended expansion absorbing the strike part of the shift
   residual (Remark 9, Eq. 61), compared against the standard expansion of
   Experiment 3.

## Layout

```
src/     model, proxy, Hermite-expansion and reference-pricing library code
exp/     one script per experiment, calling only the src/ API
tests/   pytest unit tests of the src/ components
results/ saved metrics, tables and RESULTS.md with the paper comparison
data/    empty package: the experiments use no external or generated dataset
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys or external data are required: every quantity is computed from
the closed-form kernels and parameters of the paper.

## Running the tests

```bash
python -m pytest
```

## Running the experiments

```bash
python -m exp.exp1_standard_bergomi
python -m exp.exp2_rough_bergomi
python -m exp.exp3_mixed_hermite_expansion
python -m exp.exp4_mixed_implicit_proxy
python -m exp.exp5_hermite_call_price_accuracy
python -m exp.exp6_mixed_extended_expansion
```

or

```bash
bash reproduce.sh
```

Each script writes a JSON file with full per-strike rows, maximum relative
errors, runtimes and VIX futures to `results/`, overwriting any previous run.
All experiments use the paper's full configuration: 30-node Gauss-Legendre
quadrature (with the quartic substitution of Sec. 2.3) for the proxy and
correction coefficients, 256-node quadrature / 16 scrambled Sobol sequences of
2^18 paths for the references, 4000-node Gauss-Legendre quadrature for the
Hermite weights, and the Hermite truncation orders of Tables 1-2. No
experiment configuration was reduced to fit a time limit.

## Results

See `results/RESULTS.md` for the full comparison against the paper's reported
values, and the "Deviations from the paper" / "Not reproduced" sections for
every unresolved or substituted detail.
