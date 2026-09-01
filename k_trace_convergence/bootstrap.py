"""Permutation-bootstrap analysis for trace-order sensitivity."""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import median
from typing import Sequence

from .accumulation import cumulative_trace_points
from .config import BootstrapConfig, RetrospectiveConfig
from .convergence import assess_sequences
from k_trace_coverage.metrics import contiguous_kgrams


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def final_window_new_kgrams(sequences: Sequence[Sequence[str]], k: int, window: int) -> tuple[int, float]:
    if not sequences:
        return 0, 0.0
    start = max(0, len(sequences) - window)
    before = set()
    for sequence in sequences[:start]:
        before.update(contiguous_kgrams(sequence, k))
    after = set(before)
    for sequence in sequences[start:]:
        after.update(contiguous_kgrams(sequence, k))
    new = len(after) - len(before)
    return new, new / max(len(before), 1)


def permutation_bootstrap(
    sequences: Sequence[Sequence[str]],
    domain: str,
    ks: Sequence[int],
    convergence_config: RetrospectiveConfig,
    bootstrap_config: BootstrapConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    convergence_config.validate()
    bootstrap_config.validate()
    rng = random.Random(bootstrap_config.random_seed)
    n = len(sequences)
    curve_values: dict[tuple[int, int], list[int]] = defaultdict(list)
    converged_by_perm: dict[int, list[bool]] = {int(k): [] for k in ks}
    convergence_indices: dict[int, list[int]] = {int(k): [] for k in ks}
    final_rows: list[dict[str, object]] = []

    for permutation_index in range(bootstrap_config.num_permutations):
        order = list(range(n))
        rng.shuffle(order)
        permuted = [sequences[i] for i in order]
        points = cumulative_trace_points(permuted, ks)
        for point in points:
            curve_values[(point.trace_index, point.k)].append(point.cumulative_unique_kgrams)

        for k in ks:
            one_k_config = RetrospectiveConfig(
                batch_size=convergence_config.batch_size,
                minimum_valid_traces=convergence_config.minimum_valid_traces,
                patience_batches=convergence_config.patience_batches,
                relative_gain_threshold=convergence_config.relative_gain_threshold,
                absolute_gain_threshold=convergence_config.absolute_gain_threshold,
                k_values=(int(k),),
                maximum_valid_traces=convergence_config.maximum_valid_traces,
            )
            assessment, _ = assess_sequences(permuted, one_k_config)
            converged_by_perm[int(k)].append(assessment.converged)
            if assessment.trace_count_at_convergence is not None:
                convergence_indices[int(k)].append(assessment.trace_count_at_convergence)
            new10, rel10 = final_window_new_kgrams(permuted, int(k), 10)
            new20, rel20 = final_window_new_kgrams(permuted, int(k), 20)
            final_rows.append(
                {
                    "domain": domain,
                    "permutation_index": permutation_index,
                    "k": int(k),
                    "converged_by_trace_80": assessment.converged,
                    "trace_count_at_convergence": assessment.trace_count_at_convergence,
                    "new_kgrams_final_10_traces": new10,
                    "relative_gain_final_10_traces": rel10,
                    "new_kgrams_final_20_traces": new20,
                    "relative_gain_final_20_traces": rel20,
                }
            )

    summary_rows: list[dict[str, object]] = []
    for trace_index in range(1, n + 1):
        for k in ks:
            values = curve_values[(trace_index, int(k))]
            summary_rows.append(
                {
                    "domain": domain,
                    "trace_index": trace_index,
                    "k": int(k),
                    "mean_cumulative_unique_kgrams": None if not values else sum(values) / len(values),
                    "p2_5_cumulative_unique_kgrams": percentile([float(v) for v in values], 0.025),
                    "p97_5_cumulative_unique_kgrams": percentile([float(v) for v in values], 0.975),
                }
            )

    convergence_rows: list[dict[str, object]] = []
    for k in ks:
        converged_flags = converged_by_perm[int(k)]
        indices = convergence_indices[int(k)]
        final_10 = [row["new_kgrams_final_10_traces"] for row in final_rows if row["k"] == int(k)]
        final_10_rel = [row["relative_gain_final_10_traces"] for row in final_rows if row["k"] == int(k)]
        convergence_rows.append(
            {
                "domain": domain,
                "k": int(k),
                "num_permutations": bootstrap_config.num_permutations,
                "proportion_converged_by_trace_80": sum(converged_flags) / len(converged_flags),
                "median_convergence_trace_index": None if not indices else median(indices),
                "median_new_kgrams_final_10_traces": None if not final_10 else median(final_10),
                "median_relative_gain_final_10_traces": None if not final_10_rel else median(final_10_rel),
            }
        )
    return summary_rows, convergence_rows, final_rows
