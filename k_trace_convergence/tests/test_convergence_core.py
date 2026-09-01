import tempfile
import unittest
from pathlib import Path

from k_trace_convergence.accumulation import (
    batch_accumulation_points,
    cumulative_trace_points,
    full_sequence_duplicate_summary,
)
from k_trace_convergence.bootstrap import permutation_bootstrap
from k_trace_convergence.checkpoints import read_json, write_atomic_json
from k_trace_convergence.config import BootstrapConfig, ConvergenceConfig, RetrospectiveConfig
from k_trace_convergence.convergence import assess_sequences, conservative_decision
from k_trace_convergence.repository_adapter import derive_seed
from k_trace_convergence.sensitivity import run_sensitivity
from k_trace_convergence.unseen_estimators import chao1_bias_corrected
from k_trace_convergence.export import write_csv


class ConvergenceCoreTests(unittest.TestCase):
    def test_cumulative_unique_discovery_and_marginal_gain(self):
        seqs = [("A", "B", "C"), ("A", "B", "D")]
        rows = [r for r in cumulative_trace_points(seqs, (2,))]
        self.assertEqual(rows[0].cumulative_unique_kgrams, 2)
        self.assertEqual(rows[1].new_unique_kgrams, 1)
        self.assertAlmostEqual(rows[1].relative_gain, 1 / 2)

    def test_patience_and_minimum_trace_requirement_across_all_k(self):
        seqs = [("A", "B", "C", "D")] * 6
        cfg = ConvergenceConfig(batch_size=1, minimum_valid_traces=3, patience_batches=2, k_values=(2, 3), maximum_valid_traces=10)
        assessment, _ = assess_sequences(seqs, cfg)
        self.assertTrue(assessment.converged)
        self.assertEqual(assessment.trace_count_at_convergence, 4)

    def test_maximum_limit_stopping(self):
        seqs = [(str(i), str(i + 1)) for i in range(5)]
        cfg = ConvergenceConfig(batch_size=1, minimum_valid_traces=1, patience_batches=2, absolute_gain_threshold=0, relative_gain_threshold=0, k_values=(2,), maximum_valid_traces=5)
        assessment, _ = assess_sequences(seqs, cfg)
        self.assertFalse(assessment.converged)
        self.assertEqual(assessment.status, "max_limit_reached_without_convergence")

    def test_duplicate_traces_and_short_empty_traces(self):
        seqs = [("A", "B"), ("A", "B"), tuple(), ("A",)]
        summary = full_sequence_duplicate_summary(seqs)
        self.assertEqual(summary["duplicate_full_traces"], 1)
        rows = batch_accumulation_points(seqs, (2, 3), 2, 2, 0.01)
        self.assertEqual([r.cumulative_unique_kgrams for r in rows if r.k == 3][-1], 0)

    def test_permutation_bootstrap_reproducibility(self):
        seqs = [("A", "B", str(i)) for i in range(8)]
        cfg = RetrospectiveConfig(batch_size=2, minimum_valid_traces=4, patience_batches=2, k_values=(2,), maximum_valid_traces=8)
        boot = BootstrapConfig(num_permutations=10, random_seed=7)
        first = permutation_bootstrap(seqs, "d", (2,), cfg, boot)
        second = permutation_bootstrap(seqs, "d", (2,), cfg, boot)
        self.assertEqual(first, second)

    def test_unavailable_native_order_decision_is_inconclusive(self):
        self.assertEqual(conservative_decision(None, {2: 1.0}), "inconclusive")

    def test_zero_doubletons_chao1(self):
        self.assertEqual(chao1_bias_corrected(3, 2, 0), 4.0)

    def test_checkpoint_resume_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            write_atomic_json(path, {"a": 1})
            self.assertEqual(read_json(path), {"a": 1})

    def test_deterministic_seed_derivation(self):
        self.assertEqual(derive_seed(42, "d", 1, 2), derive_seed(42, "d", 1, 2))
        self.assertNotEqual(derive_seed(42, "d", 1, 2), derive_seed(42, "d", 1, 3))

    def test_sensitivity_from_stored_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "batch_convergence_log.csv"
            rows = []
            for batch in range(1, 5):
                rows.append({
                    "domain": "d",
                    "batch_index": batch,
                    "k": 2,
                    "start_valid_trace_count": batch,
                    "end_valid_trace_count": batch,
                    "previous_total_unique_kgrams": 10,
                    "cumulative_unique_kgrams": 10,
                    "new_unique_kgrams": 0,
                    "relative_gain": 0.0,
                })
            write_csv(path, rows)
            out = run_sensitivity(path, [0.01], [2], 2, 1, 1, 10, (2,))
            self.assertEqual(out[0]["trace_count_at_convergence"], 2)


if __name__ == "__main__":
    unittest.main()
