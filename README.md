# TEMPSPECGEN Artifact

## Data provenance

- `data/benchmark/` contains the labeled benchmark instances.
- `data/ground_truth/` contains the retained TEMPSPECGEN reference behaviors.
- `data/ltl/` contains the policy-processing and verified LTL artifacts.


## Code organization

- `construction/`: reference-trace and instruction construction.
- `evaluation/`: evaluator, model-run protocol, and saved-output reevaluation utilities.
- `analysis/rq1` through `analysis/rq4`: result analyses.
- `k_trace_convergence/`: k-trace metric and multi-seed generation utilities.
- `tests/`: evaluator regression tests.
