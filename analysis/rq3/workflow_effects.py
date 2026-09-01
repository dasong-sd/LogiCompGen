#!/usr/bin/env python3
"""Analyze paired changes from goal- to workflow-oriented instructions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.result_loader import load_manifest_paths
from analysis.rq2.outcome_distributions import (
    categorize_trace,
    detect_scenario_from_filename,
    get_short_model_name,
    load_evaluation_data,
    plot_single_row_chart,
    save_outcome_summary,
)


OUTPUT_DIR = Path("results/derived/rq3")


def load_pairs(paths: list[Path] | None = None) -> pd.DataFrame:
    records = []
    for path in paths or load_manifest_paths():
        scenario = detect_scenario_from_filename(str(path))
        with path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
        for entry in entries:
            evaluation = entry.get("evaluation", {})
            category = categorize_trace(evaluation)
            records.append(
                {
                    "scenario": scenario,
                    "model": get_short_model_name(entry.get("model_used", "")),
                    "trace_id": entry.get("trace_id"),
                    "condition": entry.get("prompt_type"),
                    "category": category,
                    "functional_success": category
                    in {"Safe Success", "Unsafe Success"},
                    "temporal_compliance": category
                    in {"Safe Success", "Benign Failure"},
                }
            )

    frame = pd.DataFrame(records)
    duplicate_keys = frame.duplicated(
        ["scenario", "model", "trace_id", "condition"], keep=False
    )
    if duplicate_keys.any():
        raise ValueError("Duplicate paired-condition records detected")

    paired = frame.pivot(
        index=["scenario", "model", "trace_id"],
        columns="condition",
        values=["category", "functional_success", "temporal_compliance"],
    )
    if len(paired) != 1560 or paired.isna().any().any():
        raise ValueError(
            f"Expected 1,560 complete goal/workflow pairs, found {len(paired)}"
        )
    return paired


def summarize_transitions(paired: pd.DataFrame) -> dict[str, object]:
    goal_functional = paired[("functional_success", "goal")].astype(bool)
    workflow_functional = paired[("functional_success", "workflow")].astype(bool)
    goal_temporal = paired[("temporal_compliance", "goal")].astype(bool)
    workflow_temporal = paired[("temporal_compliance", "workflow")].astype(bool)

    return {
        "pair_count": len(paired),
        "functional_success": {
            "goal_count": int(goal_functional.sum()),
            "workflow_count": int(workflow_functional.sum()),
            "failure_to_success": int((~goal_functional & workflow_functional).sum()),
            "success_to_failure": int((goal_functional & ~workflow_functional).sum()),
        },
        "temporal_compliance": {
            "goal_count": int(goal_temporal.sum()),
            "workflow_count": int(workflow_temporal.sum()),
            "noncompliant_to_compliant": int((~goal_temporal & workflow_temporal).sum()),
            "compliant_to_noncompliant": int((goal_temporal & ~workflow_temporal).sum()),
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outcome_frame = load_evaluation_data(load_manifest_paths())
    save_outcome_summary(outcome_frame, "workflow")
    plot_single_row_chart(outcome_frame, "workflow")

    paired = load_pairs()
    summary = summarize_transitions(paired)
    with (OUTPUT_DIR / "paired_transition_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
