"""Offline sensitivity analysis over stored batch convergence logs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from .config import ConvergenceConfig, resolve_output_dir
from .convergence import assess_batch_points
from .accumulation import BatchPoint
from .export import write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate convergence threshold sensitivity from a stored log.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--logicompgen-dir", type=Path, default=Path("results/k_trace_convergence/logicompgen"))
    parser.add_argument("--output", type=Path, default=Path("results/k_trace_convergence/convergence_sensitivity.csv"))
    parser.add_argument("--relative-gain-thresholds", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--patience-batches", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--absolute-gain-threshold", type=int, default=2)
    parser.add_argument("--min-valid-traces", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-valid-traces", type=int, default=5000)
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    log_dir = resolve_output_dir(repo_root, args.logicompgen_dir)
    output = resolve_output_dir(repo_root, args.output)
    rows = run_sensitivity(
        log_dir / "batch_convergence_log.csv",
        args.relative_gain_thresholds,
        args.patience_batches,
        args.absolute_gain_threshold,
        args.min_valid_traces,
        args.batch_size,
        args.max_valid_traces,
        tuple(args.ks),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows)
    return 0


def run_sensitivity(
    batch_log_path: Path,
    relative_thresholds,
    patience_values,
    absolute_threshold: int,
    minimum_valid_traces: int,
    batch_size: int,
    maximum_valid_traces: int,
    ks,
) -> list[dict[str, object]]:
    by_domain: dict[str, list[dict[str, str]]] = defaultdict(list)
    if not batch_log_path.exists() or batch_log_path.stat().st_size == 0:
        return []
    with batch_log_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row:
                by_domain[row["domain"]].append(row)
    output_rows: list[dict[str, object]] = []
    for domain, raw_rows in sorted(by_domain.items()):
        for rel in relative_thresholds:
            for patience in patience_values:
                points = [
                    BatchPoint(
                        batch_index=int(row["batch_index"]),
                        start_trace_index=int(row["start_valid_trace_count"]),
                        end_trace_index=int(row["end_valid_trace_count"]),
                        k=int(row["k"]),
                        previous_total_unique_kgrams=int(row["previous_total_unique_kgrams"]),
                        cumulative_unique_kgrams=int(row["cumulative_unique_kgrams"]),
                        new_unique_kgrams=int(row["new_unique_kgrams"]),
                        relative_gain=float(row["relative_gain"]),
                        saturated=int(row["new_unique_kgrams"]) <= absolute_threshold and float(row["relative_gain"]) <= rel,
                    )
                    for row in raw_rows
                ]
                config = ConvergenceConfig(
                    batch_size=batch_size,
                    minimum_valid_traces=minimum_valid_traces,
                    patience_batches=patience,
                    relative_gain_threshold=rel,
                    absolute_gain_threshold=absolute_threshold,
                    k_values=tuple(ks),
                    maximum_valid_traces=maximum_valid_traces,
                )
                assessment = assess_batch_points(points, config)
                output_rows.append(
                    {
                        "domain": domain,
                        "relative_gain_threshold": rel,
                        "patience_batches": patience,
                        "absolute_gain_threshold": absolute_threshold,
                        "status": assessment.status,
                        "trace_count_at_convergence": assessment.trace_count_at_convergence,
                        "batches_processed": assessment.batches_processed,
                        "stop_reason": assessment.stop_reason,
                    }
                )
    return output_rows


if __name__ == "__main__":
    raise SystemExit(main())
