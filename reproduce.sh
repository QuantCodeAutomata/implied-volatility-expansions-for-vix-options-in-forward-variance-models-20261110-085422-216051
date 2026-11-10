#!/usr/bin/env bash
# Runs the test suite and every experiment of the VIX implied-volatility
# expansion replication, in the same configuration as README.md.
set -uo pipefail
cd "$(dirname "$0")"

python -m pytest || exit 1

python -m exp.exp1_standard_bergomi
python -m exp.exp2_rough_bergomi
python -m exp.exp3_mixed_hermite_expansion
python -m exp.exp4_mixed_implicit_proxy
python -m exp.exp5_hermite_call_price_accuracy
python -m exp.exp6_mixed_extended_expansion
