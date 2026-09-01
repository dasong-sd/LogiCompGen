"""Run LOGICOMPGEN adaptive convergence experiments for multiple master seeds."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .adaptive_generator import run_domain_until_converged
from .config import ConvergenceConfig, resolve_output_dir
from .export import ensure_output_dir
from .generate_until_converged import _aggregate_outputs


DEFAULT_DOMAINS = ["bank_manager", "teladoc", "smart_lock"]
DEFAULT_SEEDS = [42, 123, 2026]
REQUIRED_SEED_FILES = [
    "generated_trace_manifest.jsonl",
    "generation_failures.csv",
    "batch_convergence_log.csv",
    "domain_convergence_summary.csv",
    "ktrace_accumulation_curves.csv",
    "duplicate_trace_summary.csv",
    "instance_generation_distribution.csv",
    "final_unique_kgrams.json",
    "generation_config.json",
    "generation_report.md",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-master-seed LOGICOMPGEN convergence experiments.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output-root", type=Path, default=Path("results/k_trace_convergence/multi_seed"))
    parser.add_argument("--seed-42-source", type=Path, default=Path("results/k_trace_convergence/logicompgen"))
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--min-valid-traces", type=int, default=100)
    parser.add_argument("--patience-batches", type=int, default=5)
    parser.add_argument("--relative-gain-threshold", type=float, default=0.01)
    parser.add_argument("--absolute-gain-threshold", type=int, default=2)
    parser.add_argument("--max-valid-traces", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_root = resolve_output_dir(repo_root, args.output_root)
    seed_42_source = resolve_output_dir(repo_root, args.seed_42_source)
    ensure_output_dir(output_root)
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

    for seed in args.seeds:
        seed_dir = output_root / f"seed_{seed}"
        if seed == 42 and _can_reuse_seed_42(seed_dir, seed_42_source, args.domains, config):
            print(f"Reusing existing seed 42 results at {seed_dir}")
            continue
        if seed_dir.exists() and not args.resume and not args.overwrite:
            validate_seed_output(seed_dir, seed, args.domains)
            print(f"Existing complete seed {seed} results found at {seed_dir}; reusing without overwrite")
            continue
        if args.overwrite and seed_dir.exists() and not args.resume:
            shutil.rmtree(seed_dir)
        ensure_output_dir(seed_dir)
        if args.dry_run:
            print(f"Dry run: would generate seed {seed} into {seed_dir}")
            continue
        summaries = []
        for domain in args.domains:
            summaries.append(
                run_domain_until_converged(
                    repo_root=repo_root,
                    domain=domain,
                    output_dir=seed_dir,
                    config=config,
                    seed=seed,
                    resume=args.resume,
                    overwrite=args.overwrite,
                    dry_run=False,
                )
            )
        _aggregate_outputs(seed_dir, args.domains, summaries, config, seed)
        validate_seed_output(seed_dir, seed, args.domains)
        print(f"Completed seed {seed}: {seed_dir}")
    return 0


def _can_reuse_seed_42(seed_dir: Path, source_dir: Path, domains: list[str], config: ConvergenceConfig) -> bool:
    if seed_dir.exists():
        validate_seed_output(seed_dir, 42, domains)
        return True
    validate_seed_output(source_dir, 42, domains)
    _validate_config(source_dir, 42, config)
    shutil.copytree(source_dir, seed_dir)
    return True


def validate_seed_output(seed_dir: Path, seed: int, domains: list[str]) -> None:
    if not seed_dir.exists():
        raise FileNotFoundError(f"Missing output directory for seed {seed}: {seed_dir}")
    missing = [name for name in REQUIRED_SEED_FILES if not (seed_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Seed {seed} output is incomplete at {seed_dir}; missing: {', '.join(missing)}")
    config_path = seed_dir / "generation_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    recorded_seed = config.get("master_seed", config.get("seed"))
    if int(recorded_seed) != int(seed):
        raise ValueError(f"Seed directory {seed_dir} records seed {recorded_seed}, expected {seed}")
    for domain in domains:
        domain_dir = seed_dir / domain
        if not domain_dir.exists():
            raise FileNotFoundError(f"Seed {seed} is missing domain directory: {domain_dir}")
        for name in [
            "generated_trace_manifest.jsonl",
            "generation_failures.csv",
            "batch_convergence_log.csv",
            "ktrace_accumulation_curves.csv",
            "duplicate_trace_summary.json",
            "final_unique_kgrams.json",
            "instance_generation_distribution.csv",
            "generation_report.md",
        ]:
            if not (domain_dir / name).exists():
                raise FileNotFoundError(f"Seed {seed} domain {domain} missing {name}")


def _validate_config(seed_dir: Path, seed: int, config: ConvergenceConfig) -> None:
    path = seed_dir / "generation_config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    recorded_seed = data.get("master_seed", data.get("seed"))
    if int(recorded_seed) != int(seed):
        raise ValueError(f"Cannot reuse {seed_dir}: expected seed {seed}, found {recorded_seed}")
    expected = config.to_dict()
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"Cannot reuse {seed_dir}: config mismatch for {key}: {data.get(key)} != {value}")


if __name__ == "__main__":
    raise SystemExit(main())
