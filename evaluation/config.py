"""Shared paths and LTL loading for generation and evaluator-only reruns."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger


SCENARIO_CONFIG = {
    "bank_manager": {
        "ground_truth": "data/ground_truth/bank_manager_ground_truth_cases.json",
        "prompts": "data/benchmark/labeled_bank_manager_dataset.json",
        "policy": "data/policies/psd2.json",
        "ltl": "data/ltl/psd2/7_label_ltl_rules.json",
        "api_doc": "data/api_docs/ToolEmu/BankManager/doc.json",
    },
    "teladoc": {
        "ground_truth": "data/ground_truth/teladoc_ground_truth_cases.json",
        "prompts": "data/benchmark/labeled_teladoc_dataset.json",
        "policy": "data/policies/hipaa.json",
        "ltl": "data/ltl/hipaa/7_label_ltl_rules.json",
        "api_doc": "data/api_docs/ToolEmu/Teladoc/doc.json",
    },
    "smart_lock": {
        "ground_truth": "data/ground_truth/smart_lock_ground_truth_cases.json",
        "prompts": "data/benchmark/labeled_smart_lock_dataset.json",
        "policy": "data/policies/esti.json",
        "ltl": "data/ltl/esti/7_label_ltl_rules.json",
        "api_doc": "data/api_docs/ToolEmu/AugustSmartLock/doc.json",
    },
}


def load_filtered_ltl_rules(rules_filepath: str) -> list[str]:
    path = Path(rules_filepath)
    if not path.exists():
        raise FileNotFoundError(f"Verified LTL file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    rules = [rule["final_ltl_rule"] for rule in data.get("valid_ltl_rules", [])]
    if not rules:
        raise ValueError(f"No verified LTL rules found in {path}")
    logger.success(f"Loaded {len(rules)} verified LTL rules from {path}")
    return rules
