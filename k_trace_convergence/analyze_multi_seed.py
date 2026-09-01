"""Aggregate multi-master-seed LOGICOMPGEN convergence outputs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from .config import resolve_output_dir
from .export import ensure_output_dir, write_csv, write_text


DOMAIN_ORDER = ["bank_manager", "teladoc", "smart_lock"]
K_VALUES = [2, 3, 4]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze multi-seed LOGICOMPGEN convergence outputs.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--input-root", type=Path, default=Path("results/k_trace_convergence/multi_seed"))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/k_trace_convergence/multi_seed_summary"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    input_root = resolve_output_dir(repo_root, args.input_root)
    output_dir = resolve_output_dir(repo_root, args.output_dir)
    ensure_output_dir(output_dir)

    domain_rows = load_multi_seed_domain_results(input_root, args.seeds)
    summary_rows = summarize_domain_results(domain_rows)
    kgram_rows = summarize_kgrams(domain_rows)
    runtime_rows = summarize_metric(domain_rows, "runtime_seconds", "runtime")
    duplicate_rows = summarize_metric(domain_rows, "duplicate_ratio", "duplicate_ratio")
    table_tex = latex_table(summary_rows)

    write_csv(output_dir / "multi_seed_domain_results.csv", domain_rows, fieldnames=DOMAIN_FIELDNAMES)
    write_csv(output_dir / "multi_seed_summary.csv", summary_rows, fieldnames=SUMMARY_FIELDNAMES)
    write_csv(output_dir / "multi_seed_kgram_summary.csv", kgram_rows)
    write_csv(output_dir / "multi_seed_runtime_summary.csv", runtime_rows)
    write_csv(output_dir / "multi_seed_duplicate_summary.csv", duplicate_rows)
    write_text(output_dir / "multi_seed_convergence_table.tex", table_tex)
    plot_accumulation_curves(input_root, args.seeds, output_dir / "figures")
    write_text(output_dir / "multi_seed_report.md", build_report(domain_rows, summary_rows))
    return 0


DOMAIN_FIELDNAMES = [
    "master_seed",
    "domain",
    "status",
    "valid_trace_count_at_stop",
    "trace_count_at_convergence",
    "unique_full_trace_count",
    "duplicate_ratio",
    "unique_2grams",
    "unique_3grams",
    "unique_4grams",
    "number_of_batches",
    "runtime_seconds",
    "stop_reason",
]

SUMMARY_FIELDNAMES = [
    "domain",
    "number_of_runs",
    "number_converged",
    "number_max_limit_reached",
    "mean_trace_count_at_convergence",
    "std_trace_count_at_convergence",
    "min_trace_count_at_convergence",
    "max_trace_count_at_convergence",
    "mean_unique_2grams",
    "std_unique_2grams",
    "mean_unique_3grams",
    "std_unique_3grams",
    "mean_unique_4grams",
    "std_unique_4grams",
    "mean_runtime_seconds",
    "std_runtime_seconds",
]


def load_multi_seed_domain_results(input_root: Path, seeds: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        seed_dir = input_root / f"seed_{seed}"
        path = seed_dir / "domain_convergence_summary.csv"
        if not seed_dir.exists():
            raise FileNotFoundError(f"Missing output directory for seed {seed}: {seed_dir}")
        if not path.exists():
            raise FileNotFoundError(f"Missing domain summary for seed {seed}: {path}")
        for row in _read_csv(path):
            status = row.get("status", "")
            valid_stop = _as_int(row.get("valid_trace_count_at_stop"))
            trace_count = _as_int(row.get("trace_count_at_convergence"))
            if status != "converged":
                trace_count = None
            elif trace_count is None:
                trace_count = valid_stop
            rows.append(
                {
                    "master_seed": seed,
                    "domain": row.get("domain", ""),
                    "status": status,
                    "valid_trace_count_at_stop": valid_stop,
                    "trace_count_at_convergence": trace_count,
                    "unique_full_trace_count": _as_int(row.get("unique_full_trace_count")),
                    "duplicate_ratio": _as_float(row.get("duplicate_ratio")),
                    "unique_2grams": _as_int(row.get("unique_2grams")),
                    "unique_3grams": _as_int(row.get("unique_3grams")),
                    "unique_4grams": _as_int(row.get("unique_4grams")),
                    "number_of_batches": _as_int(row.get("number_of_batches")),
                    "runtime_seconds": _as_float(row.get("runtime_seconds")),
                    "stop_reason": row.get("stop_reason", ""),
                }
            )
    return sorted(rows, key=lambda r: (_domain_sort(str(r["domain"])), int(r["master_seed"])))


def summarize_domain_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(row)
    summaries: list[dict[str, object]] = []
    for domain in sorted(by_domain, key=_domain_sort):
        items = by_domain[domain]
        convergence_counts = [int(r["trace_count_at_convergence"]) for r in items if r.get("trace_count_at_convergence") not in (None, "")]
        summaries.append(
            {
                "domain": domain,
                "number_of_runs": len(items),
                "number_converged": sum(1 for r in items if r.get("status") == "converged"),
                "number_max_limit_reached": sum(1 for r in items if r.get("status") == "max_limit_reached_without_convergence"),
                "mean_trace_count_at_convergence": _mean_or_blank(convergence_counts),
                "std_trace_count_at_convergence": _std_or_blank(convergence_counts),
                "min_trace_count_at_convergence": min(convergence_counts) if convergence_counts else None,
                "max_trace_count_at_convergence": max(convergence_counts) if convergence_counts else None,
                "mean_unique_2grams": _mean_or_blank(_metric_values(items, "unique_2grams")),
                "std_unique_2grams": _std_or_blank(_metric_values(items, "unique_2grams")),
                "mean_unique_3grams": _mean_or_blank(_metric_values(items, "unique_3grams")),
                "std_unique_3grams": _std_or_blank(_metric_values(items, "unique_3grams")),
                "mean_unique_4grams": _mean_or_blank(_metric_values(items, "unique_4grams")),
                "std_unique_4grams": _std_or_blank(_metric_values(items, "unique_4grams")),
                "mean_runtime_seconds": _mean_or_blank(_metric_values(items, "runtime_seconds")),
                "std_runtime_seconds": _std_or_blank(_metric_values(items, "runtime_seconds")),
            }
        )
    return summaries


def summarize_kgrams(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(row)
    for domain in sorted(by_domain, key=_domain_sort):
        for k in K_VALUES:
            values = _metric_values(by_domain[domain], f"unique_{k}grams")
            result.append(
                {
                    "domain": domain,
                    "k": k,
                    "number_of_runs": len(values),
                    "mean_unique_kgrams": _mean_or_blank(values),
                    "std_unique_kgrams": _std_or_blank(values),
                    "min_unique_kgrams": min(values) if values else None,
                    "max_unique_kgrams": max(values) if values else None,
                }
            )
    return result


def summarize_metric(rows: list[dict[str, object]], metric: str, label: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(row)
    for domain in sorted(by_domain, key=_domain_sort):
        values = _metric_values(by_domain[domain], metric)
        result.append(
            {
                "domain": domain,
                "metric": label,
                "number_of_runs": len(values),
                "mean": _mean_or_blank(values),
                "std": _std_or_blank(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        )
    return result


def latex_table(summary_rows: list[dict[str, object]]) -> str:
    newline = r"\\"
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Domain & Converged Runs & Trace Count at Convergence & Range & Unique 2-grams & Unique 3-grams & Unique 4-grams & Runtime " + newline,
        r"\midrule",
    ]
    for row in summary_rows:
        domain = str(row["domain"]).replace("_", r"\_")
        converged = f"{row['number_converged']}/{row['number_of_runs']}"
        count = _mean_std_tex(row["mean_trace_count_at_convergence"], row["std_trace_count_at_convergence"], decimals=0)
        if count == "N/A":
            range_text = "N/A"
        else:
            range_text = f"{_fmt(row['min_trace_count_at_convergence'], 0)}-{_fmt(row['max_trace_count_at_convergence'], 0)}"
        lines.append(
            f"{domain} & {converged} & {count} & {range_text} & "
            f"{_mean_std_tex(row['mean_unique_2grams'], row['std_unique_2grams'], 1)} & "
            f"{_mean_std_tex(row['mean_unique_3grams'], row['std_unique_3grams'], 1)} & "
            f"{_mean_std_tex(row['mean_unique_4grams'], row['std_unique_4grams'], 1)} & "
            f"{_mean_std_tex(row['mean_runtime_seconds'], row['std_runtime_seconds'], 2)} " + newline
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def build_report(domain_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> str:
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in domain_rows:
        by_domain[str(row["domain"])].append(row)
    summary_by_domain = {str(row["domain"]): row for row in summary_rows}
    lines = [
        "# Multi-Seed Empirical k-Trace Saturation Report",
        "",
        "This report aggregates LOGICOMPGEN adaptive convergence runs across master seeds 42, 123, and 2026. All convergence thresholds, batch settings, k values, maximum valid traces, LTL rules, benchmark/domain inputs, and trace validity criteria are held fixed; only the master seed changes.",
        "",
        "Seed derivation uses a deterministic 32-bit rolling hash over the master seed and typed components for domain, batch, benchmark-instance/allocation, and mutation/fuzzing. LOGICOMPGEN does not expose a fixed benchmark-instance scheduler, so the benchmark-instance seed refers to the candidate/allocation unit available to the adaptive wrapper. The mutation/fuzzing seed is applied immediately before each candidate generation attempt.",
        "",
        "## Paper Table",
        "",
        "| Domain | Converged Runs | Trace Count at Convergence, Mean +/- SD | Range | Unique 2-grams, Mean +/- SD | Unique 3-grams, Mean +/- SD | Unique 4-grams, Mean +/- SD | Runtime, Mean +/- SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        count = _mean_std_text(row["mean_trace_count_at_convergence"], row["std_trace_count_at_convergence"], 0)
        range_text = "N/A" if count == "N/A" else f"{_fmt(row['min_trace_count_at_convergence'], 0)}-{_fmt(row['max_trace_count_at_convergence'], 0)}"
        lines.append(
            f"| {row['domain']} | {row['number_converged']}/{row['number_of_runs']} | {count} | {range_text} | "
            f"{_mean_std_text(row['mean_unique_2grams'], row['std_unique_2grams'], 1)} | "
            f"{_mean_std_text(row['mean_unique_3grams'], row['std_unique_3grams'], 1)} | "
            f"{_mean_std_text(row['mean_unique_4grams'], row['std_unique_4grams'], 1)} | "
            f"{_mean_std_text(row['mean_runtime_seconds'], row['std_runtime_seconds'], 2)} |"
        )
    lines.extend(["", "## Stability Questions", ""])
    for domain in DOMAIN_ORDER:
        if domain not in by_domain:
            continue
        summary = summary_by_domain[domain]
        items = by_domain[domain]
        convergence_counts = [int(r["trace_count_at_convergence"]) for r in items if r.get("trace_count_at_convergence") not in (None, "")]
        lines.append(f"### {domain}")
        lines.append("")
        lines.append(f"- Saturation stopping point: {_stability_sentence(convergence_counts, int(summary['number_of_runs']))}")
        if domain == "smart_lock":
            any_converged = int(summary["number_converged"]) > 0
            lines.append(f"- Smart Lock convergence: {'At least one seed reached empirical k-trace saturation.' if any_converged else 'The domain did not reach empirical k-trace saturation in any run.'}")
        lines.append(
            f"- Unique k-gram counts: k=2 {_range_sentence(_metric_values(items, 'unique_2grams'))}; "
            f"k=3 {_range_sentence(_metric_values(items, 'unique_3grams'))}; "
            f"k=4 {_range_sentence(_metric_values(items, 'unique_4grams'))}."
        )
        lines.append(
            f"- Runtime and duplicate ratio: runtime {_range_sentence(_metric_values(items, 'runtime_seconds'), decimals=2)}; "
            f"duplicate ratio {_range_sentence(_metric_values(items, 'duplicate_ratio'), decimals=3)}."
        )
        lines.append("")
    lines.extend(
        [
            "## One-Seed Conclusion Check",
            "",
            _overall_support_sentence(summary_rows),
            "",
            "These results describe empirical k-trace saturation in the observed generated traces. They do not imply that the complete trace space has been covered.",
            "",
            "## Generated Figures",
            "",
            "Accumulation curves are written to `figures/multi_seed_accumulation_<domain>.png`.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_accumulation_curves(input_root: Path, seeds: list[int], output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "PLOTTING_SKIPPED.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return
    curve_rows: list[dict[str, object]] = []
    summaries: dict[tuple[str, int], dict[str, object]] = {}
    for seed in seeds:
        seed_dir = input_root / f"seed_{seed}"
        for row in _read_csv(seed_dir / "ktrace_accumulation_curves.csv"):
            row["master_seed"] = seed
            curve_rows.append(row)
        for row in _read_csv(seed_dir / "domain_convergence_summary.csv"):
            summaries[(row.get("domain", ""), seed)] = row
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in curve_rows:
        by_domain[str(row["domain"])].append(row)
    colors = {2: "#1f77b4", 3: "#2ca02c", 4: "#d62728"}
    for domain in sorted(by_domain, key=_domain_sort):
        fig, ax = plt.subplots(figsize=(8, 5))
        domain_rows = by_domain[domain]
        for k in K_VALUES:
            by_seed = defaultdict(list)
            for row in domain_rows:
                if int(row["k"]) == k:
                    by_seed[int(row["master_seed"])].append(row)
            all_points = defaultdict(list)
            for seed, items in sorted(by_seed.items()):
                items = sorted(items, key=lambda r: int(float(r["end_valid_trace_count"])))
                x = [int(float(r["end_valid_trace_count"])) for r in items]
                y = [int(float(r["cumulative_unique_kgrams"])) for r in items]
                ax.plot(x, y, color=colors[k], alpha=0.28, linewidth=1, label=f"k={k}, seed={seed}")
                if x and y:
                    ax.scatter([x[-1]], [y[-1]], color=colors[k], s=18, marker="o")
                for xx, yy in zip(x, y):
                    all_points[xx].append(yy)
            xs = sorted(all_points)
            if xs:
                means = [mean(all_points[x]) for x in xs]
                lows = [min(all_points[x]) for x in xs]
                highs = [max(all_points[x]) for x in xs]
                ax.plot(xs, means, color=colors[k], linewidth=2.3, label=f"k={k}, mean")
                ax.fill_between(xs, lows, highs, color=colors[k], alpha=0.10)
        for seed in seeds:
            stop = summaries.get((domain, seed), {}).get("valid_trace_count_at_stop")
            if stop not in (None, ""):
                ax.axvline(int(float(stop)), color="black", linestyle=":", linewidth=0.8, alpha=0.35)
        ax.set_title(f"Multi-seed LOGICOMPGEN accumulation: {domain}")
        ax.set_xlabel("Valid trace count")
        ax.set_ylabel("Cumulative unique k-grams")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / f"multi_seed_accumulation_{domain}.png", dpi=200)
        plt.close(fig)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            values.append(float(value))
    return values


def _mean_or_blank(values: list[float] | list[int]) -> float | None:
    return mean(values) if values else None


def _std_or_blank(values: list[float] | list[int]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return stdev(values)


def _as_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _domain_sort(domain: str) -> tuple[int, str]:
    return (DOMAIN_ORDER.index(domain) if domain in DOMAIN_ORDER else len(DOMAIN_ORDER), domain)


def _fmt(value: object, decimals: int = 1) -> str:
    if value in (None, ""):
        return "N/A"
    value = float(value)
    if math.isnan(value):
        return "N/A"
    return f"{value:.{decimals}f}" if decimals > 0 else f"{value:.0f}"


def _mean_std_text(mean_value: object, std_value: object, decimals: int = 1) -> str:
    if mean_value in (None, ""):
        return "N/A"
    return f"{_fmt(mean_value, decimals)} +/- {_fmt(std_value, decimals)}"


def _mean_std_tex(mean_value: object, std_value: object, decimals: int = 1) -> str:
    if mean_value in (None, ""):
        return "N/A"
    return f"{_fmt(mean_value, decimals)} $\\pm$ {_fmt(std_value, decimals)}"


def _range_sentence(values: list[float], decimals: int = 1) -> str:
    if not values:
        return "N/A"
    return f"mean {_fmt(mean(values), decimals)}, range {_fmt(min(values), decimals)}-{_fmt(max(values), decimals)}"


def _stability_sentence(convergence_counts: list[int], total_runs: int) -> str:
    if not convergence_counts:
        return "The domain did not reach empirical k-trace saturation in any run."
    if len(convergence_counts) < total_runs:
        return "Some seeds reached empirical k-trace saturation and some did not, so the stopping point is seed-sensitive."
    span = max(convergence_counts) - min(convergence_counts)
    avg = mean(convergence_counts)
    rel_span = span / avg if avg else 0.0
    if rel_span <= 0.10:
        return "The stopping point is stable across seeds."
    if rel_span <= 0.35:
        return "The stopping point shows moderate variability across seeds."
    return "The stopping point shows substantial variability across seeds."


def _overall_support_sentence(summary_rows: list[dict[str, object]]) -> str:
    pieces = []
    for row in summary_rows:
        domain = str(row["domain"])
        converged = int(row["number_converged"])
        runs = int(row["number_of_runs"])
        maxed = int(row["number_max_limit_reached"])
        if converged == runs:
            pieces.append(f"{domain}: all seeds reached empirical k-trace saturation")
        elif converged == 0 and maxed == runs:
            pieces.append(f"{domain}: all seeds reached the maximum-valid-trace limit without empirical k-trace saturation")
        else:
            pieces.append(f"{domain}: convergence behavior varied across seeds")
    return "Current multi-seed evidence: " + "; ".join(pieces) + "."


if __name__ == "__main__":
    raise SystemExit(main())
