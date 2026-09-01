"""Plot accumulation curves for empirical k-trace saturation."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from .config import resolve_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create empirical k-trace saturation plots.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--gpt-dir", type=Path, default=Path("results/k_trace_convergence/gpt5_mini"))
    parser.add_argument("--logicompgen-dir", type=Path, default=Path("results/k_trace_convergence/logicompgen"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/k_trace_convergence/figures"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_dir = resolve_output_dir(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "PLOTTING_SKIPPED.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return 0

    gpt_dir = resolve_output_dir(repo_root, args.gpt_dir)
    logicompgen_dir = resolve_output_dir(repo_root, args.logicompgen_dir)
    _plot_gpt_native(plt, gpt_dir / "native_order_accumulation.csv", output_dir)
    _plot_gpt_bootstrap(plt, gpt_dir / "permutation_accumulation_summary.csv", output_dir)
    _plot_logicompgen(plt, logicompgen_dir / "ktrace_accumulation_curves.csv", logicompgen_dir / "domain_convergence_summary.csv", output_dir)
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_gpt_native(plt, path: Path, output_dir: Path) -> None:
    rows = _read_csv(path)
    by_domain: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    for domain, items in by_domain.items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for k in sorted({row["k"] for row in items}, key=int):
            series = [row for row in items if row["k"] == k]
            ax.plot([int(r["trace_index"]) for r in series], [int(r["cumulative_unique_kgrams"]) for r in series], label=f"k={k}")
        ax.set_title(f"GPT-5-Mini native accumulation: {domain}")
        ax.set_xlabel("Valid trace index")
        ax.set_ylabel("Cumulative unique k-grams")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"gpt5_native_accumulation_{domain}.png", dpi=200)
        plt.close(fig)


def _plot_gpt_bootstrap(plt, path: Path, output_dir: Path) -> None:
    rows = _read_csv(path)
    by_domain: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    for domain, items in by_domain.items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for k in sorted({row["k"] for row in items}, key=int):
            series = [row for row in items if row["k"] == k]
            x = [int(r["trace_index"]) for r in series]
            mean = [float(r["mean_cumulative_unique_kgrams"]) for r in series]
            lo = [float(r["p2_5_cumulative_unique_kgrams"]) for r in series]
            hi = [float(r["p97_5_cumulative_unique_kgrams"]) for r in series]
            ax.plot(x, mean, label=f"k={k}")
            ax.fill_between(x, lo, hi, alpha=0.15)
        ax.set_title(f"GPT-5-Mini permutation bootstrap: {domain}")
        ax.set_xlabel("Valid trace index")
        ax.set_ylabel("Cumulative unique k-grams")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"gpt5_bootstrap_accumulation_{domain}.png", dpi=200)
        plt.close(fig)


def _plot_logicompgen(plt, curve_path: Path, summary_path: Path, output_dir: Path) -> None:
    rows = _read_csv(curve_path)
    summaries = {row["domain"]: row for row in _read_csv(summary_path)}
    by_domain: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    for domain, items in by_domain.items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for k in sorted({row["k"] for row in items}, key=int):
            series = [row for row in items if row["k"] == k]
            ax.plot([int(r["end_valid_trace_count"]) for r in series], [int(r["cumulative_unique_kgrams"]) for r in series], label=f"k={k}")
        stop = summaries.get(domain, {}).get("valid_trace_count_at_stop")
        if stop:
            ax.axvline(int(float(stop)), color="black", linestyle="--", linewidth=1, label="stop")
        ax.set_title(f"LOGICOMPGEN online accumulation: {domain}")
        ax.set_xlabel("Valid traces")
        ax.set_ylabel("Cumulative unique k-grams")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"logicompgen_accumulation_{domain}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
