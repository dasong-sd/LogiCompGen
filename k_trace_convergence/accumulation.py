"""Cumulative k-gram discovery utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from k_trace_coverage.metrics import contiguous_kgrams


@dataclass(frozen=True)
class TracePoint:
    trace_index: int
    k: int
    cumulative_unique_kgrams: int
    new_unique_kgrams: int
    relative_gain: float


@dataclass(frozen=True)
class BatchPoint:
    batch_index: int
    start_trace_index: int
    end_trace_index: int
    k: int
    previous_total_unique_kgrams: int
    cumulative_unique_kgrams: int
    new_unique_kgrams: int
    relative_gain: float
    saturated: bool


def cumulative_trace_points(sequences: Sequence[Sequence[str]], ks: Iterable[int]) -> list[TracePoint]:
    seen = {int(k): set() for k in ks}
    rows: list[TracePoint] = []
    for trace_index, sequence in enumerate(sequences, start=1):
        for k in seen:
            before = len(seen[k])
            seen[k].update(contiguous_kgrams(sequence, k))
            new = len(seen[k]) - before
            rows.append(
                TracePoint(
                    trace_index=trace_index,
                    k=k,
                    cumulative_unique_kgrams=len(seen[k]),
                    new_unique_kgrams=new,
                    relative_gain=new / max(before, 1),
                )
            )
    return rows


def batch_accumulation_points(
    sequences: Sequence[Sequence[str]],
    ks: Iterable[int],
    batch_size: int,
    absolute_gain_threshold: int,
    relative_gain_threshold: float,
) -> list[BatchPoint]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    seen = {int(k): set() for k in ks}
    rows: list[BatchPoint] = []
    batch_index = 0
    for start in range(0, len(sequences), batch_size):
        batch_index += 1
        batch = sequences[start : start + batch_size]
        end = start + len(batch)
        for k in seen:
            before = len(seen[k])
            for sequence in batch:
                seen[k].update(contiguous_kgrams(sequence, k))
            new = len(seen[k]) - before
            relative = new / max(before, 1)
            rows.append(
                BatchPoint(
                    batch_index=batch_index,
                    start_trace_index=start + 1,
                    end_trace_index=end,
                    k=k,
                    previous_total_unique_kgrams=before,
                    cumulative_unique_kgrams=len(seen[k]),
                    new_unique_kgrams=new,
                    relative_gain=relative,
                    saturated=new <= absolute_gain_threshold and relative <= relative_gain_threshold,
                )
            )
    return rows


def final_unique_kgrams(sequences: Sequence[Sequence[str]], ks: Iterable[int]) -> dict[int, set[tuple[str, ...]]]:
    result = {int(k): set() for k in ks}
    for sequence in sequences:
        for k in result:
            result[k].update(contiguous_kgrams(sequence, k))
    return result


def kgram_frequency(sequences: Sequence[Sequence[str]], k: int) -> Counter[tuple[str, ...]]:
    counter: Counter[tuple[str, ...]] = Counter()
    for sequence in sequences:
        counter.update(contiguous_kgrams(sequence, k))
    return counter


def full_sequence_duplicate_summary(sequences: Sequence[Sequence[str]]) -> dict[str, object]:
    counts = Counter(tuple(sequence) for sequence in sequences)
    total = len(sequences)
    unique = len(counts)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "total_valid_traces": total,
        "unique_full_traces": unique,
        "duplicate_full_traces": duplicates,
        "duplicate_ratio": None if total == 0 else duplicates / total,
    }
