"""Build cross-method convergence comparison tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import resolve_output_dir
from .export import write_csv, write_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare GPT-5-Mini and LOGICOMPGEN convergence outputs.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--gpt-dir", type=Path, default=Path("results/k_trace_convergence/gpt5_mini"))
    parser.add_argument("--logicompgen-dir", type=Path, default=Path("results/k_trace_convergence/logicompgen"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/k_trace_convergence/method_convergence_comparison.csv"))
    parser.add_argument("--output-tex", type=Path, default=Path("results/k_trace_convergence/method_convergence_comparison.tex"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    rows = build_comparison(
        resolve_output_dir(repo_root, args.gpt_dir),
        resolve_output_dir(repo_root, args.logicompgen_dir),
    )
    csv_path = resolve_output_dir(repo_root, args.output_csv)
    tex_path = resolve_output_dir(repo_root, args.output_tex)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, rows)
    write_text(tex_path, latex_table(rows))
    return 0


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_comparison(gpt_dir: Path, logicompgen_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    gpt_decisions = {row["domain"]: row for row in read_csv(gpt_dir / "gpt5_mini_convergence_decision.csv")}
    gpt_singletons = read_csv(gpt_dir / "singleton_doubleton_summary.csv")
    gpt_unique = {(row["domain"], row["k"]): row["unique_kgrams"] for row in gpt_singletons}
    for domain, row in sorted(gpt_decisions.items()):
        decision = row.get("decision", "")
        rows.append(
            {
                "domain": domain,
                "method": "gpt5_mini",
                "available_trace_count": row.get("valid_trace_count", ""),
                "convergence_assessment": decision,
                "trace_count_at_convergence": row.get("native_trace_count_at_convergence", "") if decision == "converged" else "",
                "unique_2grams_at_final_point": gpt_unique.get((domain, "2"), ""),
                "unique_3grams_at_final_point": gpt_unique.get((domain, "3"), ""),
                "unique_4grams_at_final_point": gpt_unique.get((domain, "4"), ""),
                "native_generation_mechanism": "direct GPT-5-Mini code generation from benchmark prompts",
            }
        )
    for row in read_csv(logicompgen_dir / "domain_convergence_summary.csv"):
        rows.append(
            {
                "domain": row.get("domain", ""),
                "method": "logicompgen",
                "available_trace_count": row.get("valid_trace_count_at_stop", ""),
                "convergence_assessment": row.get("status", ""),
                "trace_count_at_convergence": row.get("valid_trace_count_at_stop", "") if row.get("status") == "converged" else "",
                "unique_2grams_at_final_point": row.get("unique_2grams", ""),
                "unique_3grams_at_final_point": row.get("unique_3grams", ""),
                "unique_4grams_at_final_point": row.get("unique_4grams", ""),
                "native_generation_mechanism": "logic-guided fuzzing with adaptive empirical k-trace saturation stopping",
            }
        )
    return sorted(rows, key=lambda r: (str(r["domain"]), str(r["method"])))


def latex_table(rows: list[dict[str, object]]) -> str:
    newline = r"\\"
    lines = [
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        "Domain & Method & Traces & Status & Stop & U2 & U3/U4 " + newline,
        r"\midrule",
    ]
    for row in rows:
        domain = str(row["domain"]).replace("_", r"\_")
        method = str(row["method"]).replace("_", r"\_")
        lines.append(
            f"{domain} & {method} & {row['available_trace_count']} & "
            f"{row['convergence_assessment']} & {row['trace_count_at_convergence']} & "
            f"{row['unique_2grams_at_final_point']} & "
            f"{row['unique_3grams_at_final_point']}/{row['unique_4grams_at_final_point']} "
            + newline
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
