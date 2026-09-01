#!/usr/bin/env python3
"""Generate the multi-seed LOGICOMPGEN robustness results for journal RQ1.

The direct GPT-5-Mini baseline contains one fixed batch of 40 traces per
domain. LOGICOMPGEN has three independent generation runs (seeds 42, 123, and
2026). For each run, this script repeatedly samples 40 valid traces without
replacement, computes structural coverage metrics, and then aggregates the
three run-level means.

The analysis estimates variation across LOGICOMPGEN generation seeds. It does
not estimate between-run variation for GPT-5-Mini.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "results/derived/rq1"
MASTER_SAMPLING_SEED = 20260729
GENERATION_SEEDS = (42, 123, 2026)
SUBSAMPLES_PER_SEED = 10_000
TRACE_BUDGET = 40
K_VALUES = (2, 3, 4)

DOMAIN_CONFIGS = {
    "financial": {
        "display_name": "Financial Services",
        "manifest_domain": "bank_manager",
        "gpt_baseline": "results/rq1_sources/direct_generation/BankManager_llm_baseline_raw.json",
        "api_doc": "data/api_docs/ToolEmu/BankManager/doc.json",
    },
    "tele_healthcare": {
        "display_name": "Tele-Healthcare",
        "manifest_domain": "teladoc",
        "gpt_baseline": "results/rq1_sources/direct_generation/Teladoc_llm_baseline_raw.json",
        "api_doc": "data/api_docs/ToolEmu/Teladoc/doc.json",
    },
    "smart_home": {
        "display_name": "Smart Home IoT",
        "manifest_domain": "smart_lock",
        "gpt_baseline": "results/rq1_sources/direct_generation/AugustSmartLock_llm_baseline_raw.json",
        "api_doc": "data/api_docs/ToolEmu/AugustSmartLock/doc.json",
    },
}


def read_json(relative_path: str) -> Any:
    with (REPO_ROOT / relative_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(relative_path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with (REPO_ROOT / relative_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{relative_path}:{line_number}: invalid JSONL record"
                ) from error
    return records


def sha256(relative_path: str) -> str:
    digest = hashlib.sha256()
    with (REPO_ROOT / relative_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kgrams(traces: Iterable[Sequence[str]], k: int) -> set[tuple[str, ...]]:
    return {
        tuple(trace[index : index + k])
        for trace in traces
        for index in range(len(trace) - k + 1)
    }


def kgram_opportunities(traces: Iterable[Sequence[str]], k: int) -> int:
    return sum(max(0, len(trace) - k + 1) for trace in traces)


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def deterministic_sampling_seed(
    generation_seed: int,
    domain_key: str,
) -> int:
    material = f"{MASTER_SAMPLING_SEED}:{generation_seed}:{domain_key}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def compute_metrics(
    traces: list[list[str]],
    api_count: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in K_VALUES:
        unique_count = len(kgrams(traces, k))
        opportunities = kgram_opportunities(traces, k)
        metrics[f"unique_{k}gram_count"] = float(unique_count)
        metrics[f"{k}gram_novelty_ratio"] = (
            unique_count / opportunities if opportunities else 0.0
        )
    metrics["atc"] = metrics["unique_2gram_count"] / (api_count**2)
    return metrics


def summarize_values(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sd": statistics.stdev(values),
        "ci95_lower": quantile(values, 0.025),
        "ci95_upper": quantile(values, 0.975),
        "minimum": min(values),
        "maximum": max(values),
    }


def read_run_status(
    generation_seed: int,
    manifest_domain: str,
) -> dict[str, Any]:
    relative_path = (
        f"results/rq1_sources/multi_seed/seed_{generation_seed}/"
        "domain_convergence_summary.csv"
    )
    with (REPO_ROOT / relative_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matching = [row for row in rows if row["domain"] == manifest_domain]
    if len(matching) != 1:
        raise ValueError(
            f"{relative_path}: expected one row for {manifest_domain}, got "
            f"{len(matching)}"
        )
    row = matching[0]
    return {
        "status": row["status"],
        "stop_reason": row["stop_reason"],
        "valid_trace_count_at_stop": int(row["valid_trace_count_at_stop"]),
        "number_of_batches": int(row["number_of_batches"]),
        "attempt_count": int(row["attempt_count"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "rq": "RQ1",
        "analysis": "LOGICOMPGEN multi-generation-seed robustness",
        "design": {
            "logicompgen_generation_seeds": list(GENERATION_SEEDS),
            "subsamples_per_seed": SUBSAMPLES_PER_SEED,
            "trace_budget_per_subsample": TRACE_BUDGET,
            "sampling": "without replacement within each generation run",
            "aggregation": (
                "For each metric, first average repeated 40-trace subsamples "
                "within each generation seed, then report the mean and sample "
                "standard deviation of the three seed-level means."
            ),
            "gpt5_mini": (
                "One fixed direct-generation batch of 40 traces per domain; "
                "reported as a point estimate without between-run variation."
            ),
            "comparison_boundary": (
                "Equal metric-computation sample size (40 traces), not equal "
                "token, monetary, wall-clock, or total generation cost."
            ),
        },
        "master_sampling_seed": MASTER_SAMPLING_SEED,
        "domains": {},
    }

    per_seed_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for domain_key, config in DOMAIN_CONFIGS.items():
        api_doc = read_json(config["api_doc"])
        api_count = len({tool["name"] for tool in api_doc["tools"]})

        gpt_items = read_json(config["gpt_baseline"])
        gpt_traces = [list(item.get("extracted_trace", [])) for item in gpt_items]
        if len(gpt_traces) != TRACE_BUDGET:
            raise ValueError(
                f"{domain_key}: expected {TRACE_BUDGET} GPT traces, got "
                f"{len(gpt_traces)}"
            )
        gpt_metrics = compute_metrics(gpt_traces, api_count)

        domain_seed_results: dict[str, Any] = {}
        seed_metric_means: dict[str, list[float]] = {}

        for generation_seed in GENERATION_SEEDS:
            manifest_path = (
                "results/rq1_sources/multi_seed/"
                f"seed_{generation_seed}/{config['manifest_domain']}/"
                "generated_trace_manifest.jsonl"
            )
            records = read_jsonl(manifest_path)
            traces = [record["api_sequence"] for record in records]
            if len(traces) < TRACE_BUDGET:
                raise ValueError(
                    f"{manifest_path}: fewer than {TRACE_BUDGET} valid traces"
                )
            if any(len(trace) != 8 for trace in traces):
                raise ValueError(f"{manifest_path}: expected all trace lengths to be 8")

            run_status = read_run_status(
                generation_seed,
                config["manifest_domain"],
            )
            if run_status["valid_trace_count_at_stop"] != len(traces):
                raise ValueError(
                    f"{manifest_path}: manifest has {len(traces)} traces but "
                    "run summary reports "
                    f"{run_status['valid_trace_count_at_stop']}"
                )

            sampling_seed = deterministic_sampling_seed(generation_seed, domain_key)
            rng = random.Random(sampling_seed)
            sampled_values: dict[str, list[float]] = {}
            for _ in range(SUBSAMPLES_PER_SEED):
                sample = rng.sample(traces, TRACE_BUDGET)
                metrics = compute_metrics(sample, api_count)
                for metric, value in metrics.items():
                    sampled_values.setdefault(metric, []).append(value)

            metric_summaries: dict[str, Any] = {}
            for metric, values in sampled_values.items():
                summary = summarize_values(values)
                baseline = gpt_metrics[metric]
                summary["gpt5_mini_value"] = baseline
                summary["proportion_logicompgen_greater"] = (
                    sum(value > baseline for value in values) / len(values)
                )
                summary["proportion_logicompgen_greater_or_equal"] = (
                    sum(value >= baseline for value in values) / len(values)
                )
                metric_summaries[metric] = summary
                seed_metric_means.setdefault(metric, []).append(summary["mean"])
                per_seed_rows.append(
                    {
                        "domain": domain_key,
                        "display_name": config["display_name"],
                        "generation_seed": generation_seed,
                        "trace_pool_size": len(traces),
                        "run_status": run_status["status"],
                        "metric": metric,
                        "logicompgen_mean": summary["mean"],
                        "logicompgen_median": summary["median"],
                        "logicompgen_within_seed_sd": summary["sd"],
                        "logicompgen_ci95_lower": summary["ci95_lower"],
                        "logicompgen_ci95_upper": summary["ci95_upper"],
                        "gpt5_mini_value": baseline,
                        "proportion_logicompgen_greater": summary[
                            "proportion_logicompgen_greater"
                        ],
                        "proportion_logicompgen_greater_or_equal": summary[
                            "proportion_logicompgen_greater_or_equal"
                        ],
                    }
                )

            domain_seed_results[str(generation_seed)] = {
                "trace_pool_size": len(traces),
                "sampling_seed": sampling_seed,
                "run_status": run_status,
                "metrics": metric_summaries,
            }
            run_rows.append(
                {
                    "domain": domain_key,
                    "display_name": config["display_name"],
                    "generation_seed": generation_seed,
                    "trace_pool_size": len(traces),
                    **run_status,
                }
            )
            source_rows.append(
                {
                    "domain": domain_key,
                    "generation_seed": generation_seed,
                    "source_type": "logicompgen_trace_manifest",
                    "path": manifest_path,
                    "sha256": sha256(manifest_path),
                }
            )

        aggregate_metrics: dict[str, Any] = {}
        for metric, seed_means in seed_metric_means.items():
            if len(seed_means) != len(GENERATION_SEEDS):
                raise ValueError(
                    f"{domain_key}/{metric}: expected {len(GENERATION_SEEDS)} "
                    f"seed means, got {len(seed_means)}"
                )
            baseline = gpt_metrics[metric]
            aggregate = {
                "logicompgen_mean_across_generation_seeds": statistics.mean(seed_means),
                "logicompgen_sd_across_generation_seeds": statistics.stdev(seed_means),
                "logicompgen_minimum_seed_mean": min(seed_means),
                "logicompgen_maximum_seed_mean": max(seed_means),
                "gpt5_mini_value": baseline,
                "generation_seed_means": {
                    str(seed): value
                    for seed, value in zip(GENERATION_SEEDS, seed_means)
                },
                "all_seed_means_greater_than_gpt5_mini": all(
                    value > baseline for value in seed_means
                ),
            }
            aggregate_metrics[metric] = aggregate
            aggregate_rows.append(
                {
                    "domain": domain_key,
                    "display_name": config["display_name"],
                    "metric": metric,
                    "logicompgen_mean_across_generation_seeds": aggregate[
                        "logicompgen_mean_across_generation_seeds"
                    ],
                    "logicompgen_sd_across_generation_seeds": aggregate[
                        "logicompgen_sd_across_generation_seeds"
                    ],
                    "logicompgen_minimum_seed_mean": aggregate[
                        "logicompgen_minimum_seed_mean"
                    ],
                    "logicompgen_maximum_seed_mean": aggregate[
                        "logicompgen_maximum_seed_mean"
                    ],
                    "gpt5_mini_value": baseline,
                    "all_seed_means_greater_than_gpt5_mini": aggregate[
                        "all_seed_means_greater_than_gpt5_mini"
                    ],
                }
            )

        result["domains"][domain_key] = {
            "display_name": config["display_name"],
            "api_count": api_count,
            "gpt5_mini_fixed_40": gpt_metrics,
            "logicompgen_generation_seeds": domain_seed_results,
            "aggregate_across_generation_seeds": aggregate_metrics,
        }

        for source_type in ("gpt_baseline", "api_doc"):
            relative_path = config[source_type]
            source_rows.append(
                {
                    "domain": domain_key,
                    "generation_seed": "",
                    "source_type": source_type,
                    "path": relative_path,
                    "sha256": sha256(relative_path),
                }
            )

    with (OUTPUT_DIR / "rq1_multiseed_results.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_csv(OUTPUT_DIR / "rq1_multiseed_per_seed.csv", per_seed_rows)
    write_csv(OUTPUT_DIR / "rq1_multiseed_summary.csv", aggregate_rows)
    write_csv(OUTPUT_DIR / "rq1_multiseed_run_status.csv", run_rows)
    write_csv(OUTPUT_DIR / "rq1_multiseed_source_manifest.csv", source_rows)


if __name__ == "__main__":
    main()
