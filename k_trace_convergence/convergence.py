"""Convergence-state logic for empirical k-trace saturation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from .accumulation import BatchPoint, batch_accumulation_points
from .config import ConvergenceConfig


@dataclass(frozen=True)
class ConvergenceAssessment:
    status: str
    converged: bool
    trace_count_at_convergence: int | None
    batches_processed: int
    consecutive_saturated_batches: int
    stop_reason: str
    slowest_k: int | None


def assess_batch_points(points: Sequence[BatchPoint], config: ConvergenceConfig) -> ConvergenceAssessment:
    config.validate()
    by_batch: dict[int, list[BatchPoint]] = defaultdict(list)
    for point in points:
        by_batch[point.batch_index].append(point)

    consecutive = 0
    trace_count_at_convergence: int | None = None
    batches_processed = 0
    slowest_k: int | None = None
    for batch_index in sorted(by_batch):
        batch_points = by_batch[batch_index]
        batches_processed = batch_index
        end_trace = max(point.end_trace_index for point in batch_points)
        unsaturated = [point for point in batch_points if not point.saturated]
        if unsaturated:
            slowest_k = max(unsaturated, key=lambda p: (p.relative_gain, p.new_unique_kgrams)).k
        all_saturated = len(batch_points) > 0 and all(point.saturated for point in batch_points)
        if end_trace >= config.minimum_valid_traces and all_saturated:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= config.patience_batches:
            trace_count_at_convergence = end_trace
            return ConvergenceAssessment(
                status="converged",
                converged=True,
                trace_count_at_convergence=trace_count_at_convergence,
                batches_processed=batches_processed,
                consecutive_saturated_batches=consecutive,
                stop_reason="convergence",
                slowest_k=slowest_k,
            )

    final_trace_count = max((point.end_trace_index for point in points), default=0)
    stop_reason = (
        "max_limit_reached_without_convergence"
        if final_trace_count >= config.maximum_valid_traces
        else "available_traces_exhausted_without_convergence"
    )
    return ConvergenceAssessment(
        status=stop_reason,
        converged=False,
        trace_count_at_convergence=None,
        batches_processed=batches_processed,
        consecutive_saturated_batches=consecutive,
        stop_reason=stop_reason,
        slowest_k=slowest_k,
    )


def assess_sequences(sequences: Sequence[Sequence[str]], config: ConvergenceConfig) -> tuple[ConvergenceAssessment, list[BatchPoint]]:
    config.validate()
    limited = list(sequences[: config.maximum_valid_traces])
    points = batch_accumulation_points(
        limited,
        config.k_values,
        config.batch_size,
        config.absolute_gain_threshold,
        config.relative_gain_threshold,
    )
    return assess_batch_points(points, config), points


def conservative_decision(native_converged: bool | None, permutation_props: dict[int, float]) -> str:
    if native_converged is True and permutation_props and all(value >= 0.95 for value in permutation_props.values()):
        return "converged"
    if native_converged is False and permutation_props and any(value < 0.50 for value in permutation_props.values()):
        return "not_converged"
    return "inconclusive"
