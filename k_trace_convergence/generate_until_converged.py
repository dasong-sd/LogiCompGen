"""CLI for adaptive LOGICOMPGEN generation until empirical k-trace saturation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .adaptive_generator import run_domain_until_converged
from .config import ConvergenceConfig, resolve_output_dir
from .export import ensure_output_dir, write_csv, write_json, write_text
from .seeding import seed_derivation_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LOGICOMPGEN until empirical k-trace saturation.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--domains", nargs="+", default=["bank_manager", "teladoc", "smart_lock"])
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--min-valid-traces", type=int, default=100)
    parser.add_argument("--patience-batches", type=int, default=5)
    parser.add_argument("--relative-gain-threshold", type=float, default=0.01)
    parser.add_argument("--absolute-gain-threshold", type=int, default=2)
    parser.add_argument("--max-valid-traces", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/k_trace_convergence/logicompgen"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_dir = resolve_output_dir(repo_root, args.output_dir)
    ensure_output_dir(output_dir)
    config = ConvergenceConfig(
        batch_size=args.batch_size,
        minimum_valid_traces=args.min_valid_traces,
        patience_batches=args.patience_batches,
        relative_gain_threshold=args.relative_gain_threshold,
        absolute_gain_threshold=args.absolute_gain_threshold,
        k_values=tuple(args.ks),
        maximum_valid_traces=args.max_valid_traces,
    )
    config.validate()
    summaries: list[dict[str, object]] = []
    for domain in args.domains:
        summaries.append(
            run_domain_until_converged(
                repo_root,
                domain,
                output_dir,
                config,
                seed=args.seed,
                resume=args.resume,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        )
    if not args.dry_run:
        _aggregate_outputs(output_dir, args.domains, summaries, config, args.seed)
    return 0


def _aggregate_outputs(output_dir: Path, domains: list[str], summaries: list[dict[str, object]], config: ConvergenceConfig, seed: int) -> None:
    write_csv(output_dir / "domain_convergence_summary.csv", summaries)
    write_json(output_dir / "generation_config.json", {"master_seed": seed, "seed": seed, "seed_derivation": seed_derivation_metadata(), **config.to_dict()})
    _concat_text_files(output_dir, domains, "generated_trace_manifest.jsonl")
    _concat_text_files(output_dir, domains, "new_kgrams_by_batch.jsonl")
    _concat_csv_files(output_dir, domains, "generation_failures.csv")
    _concat_csv_files(output_dir, domains, "batch_convergence_log.csv")
    _concat_csv_files(output_dir, domains, "ktrace_accumulation_curves.csv")
    _concat_csv_files(output_dir, domains, "instance_generation_distribution.csv")
    duplicate_rows = []
    final_unique = {}
    for domain in domains:
        duplicate_path = output_dir / domain / "duplicate_trace_summary.json"
        if duplicate_path.exists():
            row = json.loads(duplicate_path.read_text(encoding="utf-8"))
            row["domain"] = domain
            duplicate_rows.append(row)
        unique_path = output_dir / domain / "final_unique_kgrams.json"
        if unique_path.exists():
            final_unique[domain] = json.loads(unique_path.read_text(encoding="utf-8"))
    write_csv(output_dir / "duplicate_trace_summary.csv", duplicate_rows)
    write_json(output_dir / "final_unique_kgrams.json", final_unique)
    write_text(
        output_dir / "generation_report.md",
        "# LOGICOMPGEN Adaptive Generation Report\n\n"
        f"Master seed: {seed}\n\n"
        "This report summarizes adaptive empirical k-trace saturation runs that wrap the existing fuzzer.\n\n"
        "Seed derivation: child seeds are derived from the master seed and typed components "
        "for domain, batch, benchmark-instance/allocation, and mutation/fuzzing. The mutation "
        "seed is applied immediately before each candidate generation attempt.\n\n"
        + "\n".join(
            f"- {row['domain']}: {row['status']} at {row['valid_trace_count_at_stop']} valid traces "
            f"({row['stop_reason']})"
            for row in summaries
        )
        + "\n",
    )


def _concat_text_files(output_dir: Path, domains: list[str], filename: str) -> None:
    target = output_dir / filename
    with target.open("w", encoding="utf-8") as out:
        for domain in domains:
            path = output_dir / domain / filename
            if path.exists():
                out.write(path.read_text(encoding="utf-8"))


def _concat_csv_files(output_dir: Path, domains: list[str], filename: str) -> None:
    target = output_dir / filename
    wrote_header = False
    with target.open("w", encoding="utf-8", newline="") as out:
        writer = None
        for domain in domains:
            path = output_dir / domain / filename
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if writer is None:
                    writer = csv.DictWriter(out, fieldnames=reader.fieldnames or [])
                    writer.writeheader()
                    wrote_header = True
                for row in reader:
                    writer.writerow(row)
        if not wrote_header:
            out.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
