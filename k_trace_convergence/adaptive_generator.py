"""Adaptive wrapper around the existing LOGICOMPGEN fuzzer."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

from k_trace_coverage.metrics import contiguous_kgrams

from .accumulation import full_sequence_duplicate_summary
from .checkpoints import read_json, write_atomic_json
from .config import ConvergenceConfig
from .export import append_jsonl, write_csv, write_json, write_text
from .repository_adapter import ExistingFuzzerAdapter
from .seeding import derive_seed, seed_derivation_metadata


def run_domain_until_converged(
    repo_root: Path,
    domain: str,
    output_dir: Path,
    config: ConvergenceConfig,
    seed: int,
    resume: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    config.validate()
    domain_dir = output_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = domain_dir / "generated_trace_manifest.jsonl"
    failures_path = domain_dir / "generation_failures.csv"
    batch_log_path = domain_dir / "batch_convergence_log.csv"
    kcurve_path = domain_dir / "ktrace_accumulation_curves.csv"
    new_kgrams_path = domain_dir / "new_kgrams_by_batch.jsonl"
    checkpoint_path = domain_dir / "checkpoint.json"

    if not resume and not overwrite and any(path.exists() for path in [manifest_path, batch_log_path, checkpoint_path]):
        raise FileExistsError(f"Existing adaptive generation artifacts found for {domain}; use --resume or --overwrite")
    if overwrite and not resume:
        for path in [manifest_path, failures_path, batch_log_path, kcurve_path, new_kgrams_path, checkpoint_path]:
            if path.exists():
                path.unlink()

    adapter = ExistingFuzzerAdapter(repo_root, domain, seed)
    sequences: list[tuple[str, ...]] = []
    seen_kgrams: dict[int, set[tuple[str, ...]]] = {int(k): set() for k in config.k_values}
    batch_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    guiding_ltl_counter: Counter[str] = Counter()
    attempt_index = 0
    consecutive_saturated_batches = 0
    status = "running"
    stop_reason = ""
    start_time = time.time()

    checkpoint = read_json(checkpoint_path) if resume else None
    if checkpoint:
        attempt_index = int(checkpoint.get("attempt_index", 0))
        consecutive_saturated_batches = int(checkpoint.get("consecutive_saturated_batches", 0))
        for key, values in checkpoint.get("seen_kgrams", {}).items():
            seen_kgrams[int(key)] = {tuple(item) for item in values}
        sequences = [tuple(item) for item in checkpoint.get("sequences", [])]
        adapter.trace_generator.occurence_book = {
            tuple(item[0]): item[1] for item in checkpoint.get("occurence_book", [])
        }
        guiding_ltl_counter.update(checkpoint.get("guiding_ltl_counts", {}))

    if dry_run:
        return {
            "domain": domain,
            "status": "dry_run",
            "valid_trace_count_at_stop": len(sequences),
            "stop_reason": "dry_run",
        }

    fieldnames_failure = [
        "domain",
        "trace_id",
        "attempt_index",
        "seed",
        "master_seed",
        "domain_seed",
        "batch_seed",
        "benchmark_instance_seed",
        "mutation_seed",
        "failure_reason",
        "trace_length",
    ]
    if not failures_path.exists():
        with failures_path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames_failure).writeheader()

    batch_index = len(sequences) // config.batch_size
    while len(sequences) < config.maximum_valid_traces:
        batch_index += 1
        batch_valid: list[tuple[str, ...]] = []
        batch_start_valid_count = len(sequences)
        batch_start_time = time.time()
        while len(batch_valid) < config.batch_size and len(sequences) < config.maximum_valid_traces:
            attempt_index += 1
            candidate = adapter.generate_candidate(batch_index, attempt_index)
            if candidate.success:
                sequences.append(candidate.api_sequence)
                batch_valid.append(candidate.api_sequence)
                for ltl in candidate.guiding_ltls:
                    guiding_ltl_counter[ltl] += 1
                append_jsonl(manifest_path, candidate.manifest_record())
            else:
                row = {
                    "domain": domain,
                    "trace_id": candidate.trace_id,
                    "attempt_index": candidate.attempt_index,
                    "seed": candidate.seed,
                    "master_seed": candidate.master_seed,
                    "domain_seed": candidate.domain_seed,
                    "batch_seed": candidate.batch_seed,
                    "benchmark_instance_seed": candidate.benchmark_instance_seed,
                    "mutation_seed": candidate.mutation_seed,
                    "failure_reason": candidate.failure_reason,
                    "trace_length": len(candidate.api_sequence),
                }
                failure_rows.append(row)
                with failures_path.open("a", encoding="utf-8", newline="") as handle:
                    csv.DictWriter(handle, fieldnames=fieldnames_failure).writerow(row)

        if not batch_valid:
            status = "generation_stalled"
            stop_reason = "no_valid_traces_in_batch"
            break

        batch_saturated = True
        batch_new_payload: dict[str, object] = {"domain": domain, "batch_index": batch_index, "new_kgrams": {}}
        for k in config.k_values:
            k = int(k)
            before = len(seen_kgrams[k])
            before_set = set(seen_kgrams[k])
            for sequence in batch_valid:
                seen_kgrams[k].update(contiguous_kgrams(sequence, k))
            new_items = sorted(seen_kgrams[k] - before_set)
            new_count = len(seen_kgrams[k]) - before
            relative_gain = new_count / max(before, 1)
            saturated = new_count <= config.absolute_gain_threshold and relative_gain <= config.relative_gain_threshold
            batch_saturated = batch_saturated and saturated
            row = {
                "domain": domain,
                "batch_index": batch_index,
                "k": k,
                "start_valid_trace_count": batch_start_valid_count + 1,
                "end_valid_trace_count": len(sequences),
                "previous_total_unique_kgrams": before,
                "cumulative_unique_kgrams": len(seen_kgrams[k]),
                "new_unique_kgrams": new_count,
                "relative_gain": relative_gain,
                "saturated": saturated,
                "batch_runtime_seconds": time.time() - batch_start_time,
            }
            batch_rows.append(row)
            curve_rows.append(row.copy())
            batch_new_payload["new_kgrams"][str(k)] = [list(item) for item in new_items]
        append_jsonl(new_kgrams_path, batch_new_payload)

        if len(sequences) >= config.minimum_valid_traces and batch_saturated:
            consecutive_saturated_batches += 1
        else:
            consecutive_saturated_batches = 0

        checkpoint_payload = {
            "domain": domain,
            "attempt_index": attempt_index,
            "valid_trace_count": len(sequences),
            "unique_full_trace_count": len(set(sequences)),
            "seen_kgrams": {str(k): [list(item) for item in sorted(values)] for k, values in seen_kgrams.items()},
            "sequences": [list(sequence) for sequence in sequences],
            "consecutive_saturated_batches": consecutive_saturated_batches,
            "status": status,
            "master_seed": seed,
            "domain_seed": derive_seed(seed, "domain", domain),
            "seed_derivation": seed_derivation_metadata(),
            "occurence_book": [[list(key), value] for key, value in adapter.trace_generator.occurence_book.items()],
            "guiding_ltl_counts": dict(guiding_ltl_counter),
            "runtime_seconds": time.time() - start_time,
        }
        write_atomic_json(checkpoint_path, checkpoint_payload)

        if consecutive_saturated_batches >= config.patience_batches:
            status = "converged"
            stop_reason = "convergence"
            break

    if status == "running":
        status = "max_limit_reached_without_convergence"
        stop_reason = "max_limit_reached_without_convergence"

    duplicate = full_sequence_duplicate_summary(sequences)
    summary = {
        "domain": domain,
        "master_seed": seed,
        "status": status,
        "valid_trace_count_at_stop": len(sequences),
        "trace_count_at_convergence": len(sequences) if status == "converged" else None,
        "unique_full_trace_count": duplicate["unique_full_traces"],
        "duplicate_ratio": duplicate["duplicate_ratio"],
        "unique_2grams": len(seen_kgrams.get(2, set())),
        "unique_3grams": len(seen_kgrams.get(3, set())),
        "unique_4grams": len(seen_kgrams.get(4, set())),
        "number_of_batches": batch_index,
        "consecutive_saturated_batches": consecutive_saturated_batches,
        "runtime_seconds": time.time() - start_time,
        "stop_reason": stop_reason,
        "attempt_count": attempt_index,
    }

    write_csv(batch_log_path, batch_rows)
    write_csv(kcurve_path, curve_rows)
    write_json(domain_dir / "duplicate_trace_summary.json", duplicate)
    write_json(domain_dir / "final_unique_kgrams.json", {str(k): [list(item) for item in sorted(values)] for k, values in seen_kgrams.items()})
    write_csv(
        domain_dir / "instance_generation_distribution.csv",
        [
            {"domain": domain, "allocation_unit": "guiding_ltl", "unit_id": ltl, "valid_trace_count": count}
            for ltl, count in sorted(guiding_ltl_counter.items())
        ],
    )
    write_text(
        domain_dir / "generation_report.md",
        f"# LOGICOMPGEN Adaptive Generation Report: {domain}\n\n"
        f"Master seed: {seed}\n\n"
        f"Status: {status}\n\n"
        f"Valid traces at stop: {len(sequences)}\n\n"
        f"Trace count at convergence: {len(sequences) if status == 'converged' else 'N/A'}\n\n"
        f"Stop reason: {stop_reason}\n\n"
        "Seed derivation: child seeds are derived from the master seed and typed components "
        "for domain, batch, benchmark-instance/allocation, and mutation/fuzzing. The mutation "
        "seed is applied immediately before each candidate generation attempt.\n\n"
        "This run wraps the existing LOGICOMPGEN fuzzer and uses empirical k-trace saturation, "
        "not complete coverage.\n",
    )
    return summary
