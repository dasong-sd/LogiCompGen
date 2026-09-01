import csv
import json
import tempfile
import unittest
from pathlib import Path

from k_trace_convergence.analyze_multi_seed import load_multi_seed_domain_results, summarize_domain_results
from k_trace_convergence.config import ConvergenceConfig
from k_trace_convergence.run_multi_seed import _can_reuse_seed_42
from k_trace_convergence.seeding import derive_candidate_seeds, derive_seed


class MultiSeedTests(unittest.TestCase):
    def test_seed_derivation_is_deterministic(self):
        first = derive_candidate_seeds(42, "bank_manager", 3, "17")
        second = derive_candidate_seeds(42, "bank_manager", 3, "17")
        self.assertEqual(first, second)
        self.assertEqual(derive_seed(42, "domain", "bank_manager"), derive_seed(42, "domain", "bank_manager"))

    def test_different_master_seed_changes_child_seed(self):
        first = derive_candidate_seeds(42, "bank_manager", 3, "17")
        second = derive_candidate_seeds(123, "bank_manager", 3, "17")
        self.assertNotEqual(first.mutation_seed, second.mutation_seed)
        self.assertNotEqual(first.domain_seed, second.domain_seed)

    def test_same_master_seed_rerun_seed_plan_is_identical(self):
        run_a = [derive_candidate_seeds(2026, "teladoc", batch, attempt).to_dict() for batch, attempt in [(1, 1), (1, 2), (2, 21)]]
        run_b = [derive_candidate_seeds(2026, "teladoc", batch, attempt).to_dict() for batch, attempt in [(1, 1), (1, 2), (2, 21)]]
        self.assertEqual(run_a, run_b)

    def test_unconverged_run_is_not_counted_as_converged(self):
        rows = [
            _domain_row(42, "smart_lock", "max_limit_reached_without_convergence", 5000, None, 139, 1134, 4791),
            _domain_row(123, "smart_lock", "max_limit_reached_without_convergence", 5000, None, 140, 1100, 4700),
        ]
        summary = summarize_domain_results(rows)[0]
        self.assertEqual(summary["number_converged"], 0)
        self.assertEqual(summary["number_max_limit_reached"], 2)
        self.assertIsNone(summary["mean_trace_count_at_convergence"])
        self.assertIsNone(summary["min_trace_count_at_convergence"])

    def test_multi_seed_mean_std_and_range(self):
        rows = [
            _domain_row(42, "bank_manager", "converged", 100, 100, 10, 20, 30, runtime=1.0),
            _domain_row(123, "bank_manager", "converged", 140, 140, 14, 26, 38, runtime=3.0),
        ]
        summary = summarize_domain_results(rows)[0]
        self.assertEqual(summary["number_of_runs"], 2)
        self.assertEqual(summary["mean_trace_count_at_convergence"], 120)
        self.assertAlmostEqual(summary["std_trace_count_at_convergence"], 28.284271247, places=6)
        self.assertEqual(summary["min_trace_count_at_convergence"], 100)
        self.assertEqual(summary["max_trace_count_at_convergence"], 140)
        self.assertEqual(summary["mean_unique_2grams"], 12)
        self.assertAlmostEqual(summary["std_runtime_seconds"], 1.414213562, places=6)

    def test_missing_seed_output_reports_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "Missing output directory for seed 999"):
                load_multi_seed_domain_results(Path(tmp), [999])

    def test_existing_seed_42_is_reused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "logicompgen"
            target = root / "multi_seed" / "seed_42"
            _write_complete_seed_dir(source, 42)
            _write_complete_seed_dir(target, 42)
            marker = "existing target marker\n"
            (target / "generation_report.md").write_text(marker, encoding="utf-8")
            reused = _can_reuse_seed_42(target, source, ["bank_manager", "teladoc", "smart_lock"], ConvergenceConfig())
            self.assertTrue(reused)
            self.assertEqual((target / "generation_report.md").read_text(encoding="utf-8"), marker)


def _domain_row(seed, domain, status, valid_stop, convergence_count, u2, u3, u4, runtime=1.0):
    return {
        "master_seed": seed,
        "domain": domain,
        "status": status,
        "valid_trace_count_at_stop": valid_stop,
        "trace_count_at_convergence": convergence_count,
        "unique_full_trace_count": valid_stop,
        "duplicate_ratio": 0.1,
        "unique_2grams": u2,
        "unique_3grams": u3,
        "unique_4grams": u4,
        "number_of_batches": valid_stop // 20,
        "runtime_seconds": runtime,
        "stop_reason": "convergence" if status == "converged" else status,
    }


def _write_complete_seed_dir(path: Path, seed: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "generation_config.json").write_text(json.dumps({"seed": seed, **ConvergenceConfig().to_dict()}), encoding="utf-8")
    (path / "generated_trace_manifest.jsonl").write_text("", encoding="utf-8")
    _write_csv(path / "generation_failures.csv", [])
    _write_csv(path / "batch_convergence_log.csv", [])
    _write_csv(path / "ktrace_accumulation_curves.csv", [])
    _write_csv(path / "duplicate_trace_summary.csv", [])
    _write_csv(path / "instance_generation_distribution.csv", [])
    (path / "final_unique_kgrams.json").write_text("{}", encoding="utf-8")
    (path / "generation_report.md").write_text("report\n", encoding="utf-8")
    _write_csv(
        path / "domain_convergence_summary.csv",
        [
            {"domain": "bank_manager", "status": "converged", "valid_trace_count_at_stop": 100, "stop_reason": "convergence"},
            {"domain": "teladoc", "status": "converged", "valid_trace_count_at_stop": 100, "stop_reason": "convergence"},
            {"domain": "smart_lock", "status": "max_limit_reached_without_convergence", "valid_trace_count_at_stop": 5000, "stop_reason": "max_limit_reached_without_convergence"},
        ],
    )
    for domain in ["bank_manager", "teladoc", "smart_lock"]:
        domain_dir = path / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        (domain_dir / "generated_trace_manifest.jsonl").write_text("", encoding="utf-8")
        _write_csv(domain_dir / "generation_failures.csv", [])
        _write_csv(domain_dir / "batch_convergence_log.csv", [])
        _write_csv(domain_dir / "ktrace_accumulation_curves.csv", [])
        (domain_dir / "duplicate_trace_summary.json").write_text("{}", encoding="utf-8")
        (domain_dir / "final_unique_kgrams.json").write_text("{}", encoding="utf-8")
        _write_csv(domain_dir / "instance_generation_distribution.csv", [])
        (domain_dir / "generation_report.md").write_text("domain report\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
