"""Analyze empirical k-trace saturation of stored GPT-5-Mini outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .accumulation import cumulative_trace_points
from .bootstrap import final_window_new_kgrams, permutation_bootstrap
from .config import BootstrapConfig, RetrospectiveConfig, resolve_output_dir
from .convergence import assess_sequences, conservative_decision
from .export import ensure_output_dir, write_csv, write_text
from .repository_adapter import load_gpt5_records, valid_sequences
from .unseen_estimators import singleton_doubleton_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze GPT-5-Mini empirical k-trace saturation.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results/k_trace_convergence/gpt5_mini"))
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--retrospective-minimum-traces", type=int, default=40)
    parser.add_argument("--retrospective-batch-size", type=int, default=10)
    parser.add_argument("--retrospective-patience-batches", type=int, default=3)
    parser.add_argument("--relative-gain-threshold", type=float, default=0.01)
    parser.add_argument("--absolute-gain-threshold", type=int, default=2)
    parser.add_argument("--num-permutations", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_dir = resolve_output_dir(repo_root, args.output_dir)
    ensure_output_dir(output_dir)
    ks = tuple(args.ks)
    retrospective = RetrospectiveConfig(
        batch_size=args.retrospective_batch_size,
        minimum_valid_traces=args.retrospective_minimum_traces,
        patience_batches=args.retrospective_patience_batches,
        relative_gain_threshold=args.relative_gain_threshold,
        absolute_gain_threshold=args.absolute_gain_threshold,
        k_values=ks,
        maximum_valid_traces=80,
    )
    bootstrap_config = BootstrapConfig(args.num_permutations, args.random_seed)
    records_by_domain = load_gpt5_records(repo_root)

    native_rows: list[dict[str, object]] = []
    permutation_curve_rows: list[dict[str, object]] = []
    permutation_convergence_rows: list[dict[str, object]] = []
    permutation_detail_rows: list[dict[str, object]] = []
    final_batch_rows: list[dict[str, object]] = []
    singleton_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    report_sections: list[str] = ["# GPT-5-Mini Empirical k-Trace Saturation Report", ""]

    for domain, records in records_by_domain.items():
        sequences = valid_sequences(records)
        native_order_available = True
        assessment, batch_points = assess_sequences(sequences, retrospective)
        native_converged = assessment.converged
        for point in cumulative_trace_points(sequences, ks):
            native_rows.append({"domain": domain, **point.__dict__})

        curve_rows, convergence_rows, detail_rows = permutation_bootstrap(
            sequences, domain, ks, retrospective, bootstrap_config
        )
        permutation_curve_rows.extend(curve_rows)
        permutation_convergence_rows.extend(convergence_rows)
        permutation_detail_rows.extend(detail_rows)

        permutation_props = {int(row["k"]): float(row["proportion_converged_by_trace_80"]) for row in convergence_rows}
        decision = conservative_decision(native_converged, permutation_props)
        slowest_k = _slowest_k(batch_points, ks)

        for k in ks:
            new10, rel10 = final_window_new_kgrams(sequences, int(k), 10)
            new20, rel20 = final_window_new_kgrams(sequences, int(k), 20)
            final_batch_rows.append(
                {
                    "domain": domain,
                    "method": "gpt5_mini",
                    "k": int(k),
                    "new_kgrams_final_10_traces_native_order": new10,
                    "relative_gain_final_10_traces_native_order": rel10,
                    "new_kgrams_final_20_traces_native_order": new20,
                    "relative_gain_final_20_traces_native_order": rel20,
                }
            )
        singleton_rows.extend(singleton_doubleton_summary(sequences, domain, "gpt5_mini", ks))

        decision_row = {
            "domain": domain,
            "method": "gpt5_mini",
            "native_order_available": native_order_available,
            "valid_trace_count": len(sequences),
            "native_retrospective_status": assessment.status,
            "native_trace_count_at_convergence": assessment.trace_count_at_convergence,
            "slowest_k": slowest_k,
            "decision": decision,
        }
        for row in convergence_rows:
            decision_row[f"k{row['k']}_permutation_converged_proportion"] = row[
                "proportion_converged_by_trace_80"
            ]
            decision_row[f"k{row['k']}_median_convergence_trace_index"] = row[
                "median_convergence_trace_index"
            ]
        decision_rows.append(decision_row)

        report_sections.extend(_domain_report(domain, decision_row, convergence_rows))

    write_csv(output_dir / "native_order_accumulation.csv", native_rows)
    write_csv(output_dir / "permutation_accumulation_summary.csv", permutation_curve_rows)
    write_csv(output_dir / "permutation_convergence_results.csv", permutation_convergence_rows)
    write_csv(output_dir / "permutation_convergence_details.csv", permutation_detail_rows)
    write_csv(output_dir / "final_batch_marginal_gain.csv", final_batch_rows)
    write_csv(output_dir / "singleton_doubleton_summary.csv", singleton_rows)
    write_csv(output_dir / "gpt5_mini_convergence_decision.csv", decision_rows)
    write_text(output_dir / "gpt5_mini_convergence_report.md", "\n".join(report_sections) + "\n")
    return 0


def _slowest_k(batch_points, ks) -> int | None:
    if not batch_points:
        return None
    last_by_k = {int(k): None for k in ks}
    for point in batch_points:
        last_by_k[point.k] = point
    candidates = [point for point in last_by_k.values() if point is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.relative_gain, p.new_unique_kgrams)).k


def _domain_report(domain: str, decision_row: dict[str, object], convergence_rows: list[dict[str, object]]) -> list[str]:
    lines = [f"## {domain}", ""]
    lines.append(f"Native order available: {decision_row['native_order_available']}.")
    lines.append(f"Valid traces analyzed: {decision_row['valid_trace_count']}.")
    lines.append(f"Native retrospective status: {decision_row['native_retrospective_status']}.")
    lines.append(f"Slowest k by final native marginal gain: {decision_row['slowest_k']}.")
    lines.append(f"Decision: {decision_row['decision']}.")
    lines.append("")
    lines.append("Permutation robustness by k:")
    for row in sorted(convergence_rows, key=lambda r: int(r["k"])):
        lines.append(
            f"- k={row['k']}: proportion converged by trace 80 = "
            f"{float(row['proportion_converged_by_trace_80']):.3f}; "
            f"median convergence index = {row['median_convergence_trace_index']}"
        )
    lines.append("")
    lines.append(
        "Interpretation note: empirical k-trace saturation is an observed-corpus diagnostic, "
        "not proof of complete coverage of the executable trace space."
    )
    lines.append("")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
