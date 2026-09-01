"""Adapters that connect convergence analysis to repository data and generator code."""

from __future__ import annotations

import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from k_trace_coverage.config import DomainConfig, default_domain_configs
from k_trace_coverage.data_loader import load_vocabularies
from k_trace_coverage.trace_extractors import TraceRecord, extract_baseline_records, extract_logicompgen_records

from .seeding import derive_candidate_seeds, derive_seed

VALID_STATUSES = {"success", "empty_trace"}

SCENARIO_RULE_FILES = {
    "bank_manager": "data/ltl/psd2/7_label_ltl_rules.json",
    "smart_lock": "data/ltl/esti/7_label_ltl_rules.json",
    "teladoc": "data/ltl/hipaa/7_label_ltl_rules.json",
}

def repository_domain_configs(repo_root: Path) -> tuple[DomainConfig, ...]:
    return default_domain_configs(repo_root)


def load_gpt5_records(repo_root: Path) -> dict[str, list[TraceRecord]]:
    domains = repository_domain_configs(repo_root)
    vocabularies = load_vocabularies(domains)
    return {domain.domain: extract_baseline_records(domain, vocabularies[domain.domain]) for domain in domains}


def load_logicompgen_records(repo_root: Path) -> dict[str, list[TraceRecord]]:
    domains = repository_domain_configs(repo_root)
    vocabularies = load_vocabularies(domains)
    return {domain.domain: extract_logicompgen_records(domain, vocabularies[domain.domain]) for domain in domains}


def valid_sequences(records: list[TraceRecord]) -> list[tuple[str, ...]]:
    return [record.api_sequence for record in records if record.extraction_status in VALID_STATUSES]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        from faker import Faker

        Faker.seed(seed)
    except Exception:
        pass


def _ensure_loguru_fallback() -> None:
    if "loguru" in sys.modules:
        return

    class _FallbackLogger:
        def __getattr__(self, _name):
            def _log(*_args, **_kwargs):
                return None
            return _log

    module = types.ModuleType("loguru")
    module.logger = _FallbackLogger()
    sys.modules["loguru"] = module

def _load_generation_stack():
    _ensure_loguru_fallback()
    from trace_generation import load_filtered_ltl_rules
    from trace_generator.bank_manager_state import BankManagerRandomInitializer, BankManagerVariableSchema
    from trace_generator.smart_lock_state import AugustLockRandomInitializer, AugustLockVariableSchema
    from trace_generator.state import TraceGenerator
    from trace_generator.teladoc_state import TeladocRandomInitializer, TeladocVariableSchema

    scenario_classes = {
        "bank_manager": (BankManagerVariableSchema, BankManagerRandomInitializer),
        "smart_lock": (AugustLockVariableSchema, AugustLockRandomInitializer),
        "teladoc": (TeladocVariableSchema, TeladocRandomInitializer),
    }
    return scenario_classes, load_filtered_ltl_rules, TraceGenerator


def _load_schema_base():
    from trace_generator.state import Schema

    return Schema

@dataclass
class CandidateTrace:
    trace_id: str
    domain: str
    success: bool
    failure_reason: str | None
    api_sequence: tuple[str, ...]
    generated_program: str | None
    initial_state: dict[str, Any] | None
    final_state: dict[str, Any] | None
    guiding_ltls: list[str]
    dynamic_inputs: dict[str, Any]
    seed: int
    master_seed: int
    domain_seed: int
    batch_seed: int
    benchmark_instance_seed: int
    mutation_seed: int
    attempt_index: int

    def manifest_record(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "domain": self.domain,
            "api_sequence": list(self.api_sequence),
            "trace_length": len(self.api_sequence),
            "generated_program": self.generated_program,
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "guiding_ltls": self.guiding_ltls,
            "dynamic_inputs": self.dynamic_inputs,
            "seed": self.seed,
            "master_seed": self.master_seed,
            "domain_seed": self.domain_seed,
            "batch_seed": self.batch_seed,
            "benchmark_instance_seed": self.benchmark_instance_seed,
            "mutation_seed": self.mutation_seed,
            "attempt_index": self.attempt_index,
        }


class ExistingFuzzerAdapter:
    """Thin wrapper around the repository's existing TraceGenerator path."""

    def __init__(self, repo_root: Path, domain: str, master_seed: int = 42) -> None:
        scenario_classes, load_filtered_ltl_rules, trace_generator_class = _load_generation_stack()
        if domain not in scenario_classes:
            raise ValueError(f"Unknown domain: {domain}")
        self.repo_root = repo_root
        self.domain = domain
        self.master_seed = master_seed
        self.schema_class, self.random_init_class = scenario_classes[domain]
        config_path = repo_root / "config" / f"{domain}_generation.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle)
        self.generation_config = config_dict.get("generation_config", {})
        rules_path = repo_root / SCENARIO_RULE_FILES[domain]
        self.ltl_rules = load_filtered_ltl_rules(str(rules_path))
        self.trace_generator = trace_generator_class(
            state_schema=self.schema_class(),
            random_generator=self.random_init_class(),
            config=self.generation_config.get("trace_config", {}),
            occurence_book={},
            log_dir=None,
            ltl_rule_strings=self.ltl_rules,
        )
        self.num_of_apis = int(self.generation_config.get("num_of_apis", 8))
        self.enable_coverage = bool(self.generation_config.get("enable_coverage", True))

    def generate_candidate(self, batch_index: int, attempt_index: int) -> CandidateTrace:
        seeds = derive_candidate_seeds(self.master_seed, self.domain, batch_index, attempt_index)
        seed_everything(seeds.mutation_seed)
        self._seed_faker_instances(seeds.mutation_seed)
        self.trace_generator.prepare_initial_state()
        trace_id = f"trace_{self.domain}_adaptive_seed{self.master_seed}_batch{batch_index}_attempt{attempt_index}"
        self.trace_generator.trace_id = trace_id
        initial_state = self.trace_generator.state_schema.get_serializable_state()["implicit_states"]
        trace, guiding_ltls, _duplicate_map, is_success = self.trace_generator.generate_trace(
            call_num=self.num_of_apis,
            enable_coverage=self.enable_coverage,
        )
        api_sequence = tuple(call_info[0] for call_info in trace[0])
        if not guiding_ltls:
            is_success = False
            failure_reason = "no_guiding_ltls"
        elif not is_success:
            failure_reason = "generate_trace_returned_failure"
        else:
            failure_reason = None

        generated_program = None
        final_state = None
        dynamic_inputs = {}
        if is_success:
            schema_base = _load_schema_base()
            init_program, _ = schema_base.return_init_local_info(
                self.trace_generator.state_schema.init_local_info,
                self.trace_generator.state_schema.dynamic_inputs,
            )
            main_program = "".join(line for block_info in trace[1] for line in block_info[0]) if trace[1] else ""
            generated_program = init_program + "\n\n" + main_program
            final_state = self.trace_generator.state_schema.get_serializable_state()["implicit_states"]
            dynamic_inputs = dict(self.trace_generator.state_schema.dynamic_inputs)

        return CandidateTrace(
            trace_id=trace_id,
            domain=self.domain,
            success=is_success,
            failure_reason=failure_reason,
            api_sequence=api_sequence,
            generated_program=generated_program,
            initial_state=initial_state if is_success else None,
            final_state=final_state,
            guiding_ltls=list(guiding_ltls),
            dynamic_inputs=dynamic_inputs,
            seed=seeds.mutation_seed,
            master_seed=seeds.master_seed,
            domain_seed=seeds.domain_seed,
            batch_seed=seeds.batch_seed,
            benchmark_instance_seed=seeds.benchmark_instance_seed,
            mutation_seed=seeds.mutation_seed,
            attempt_index=attempt_index,
        )

    def _seed_faker_instances(self, seed: int) -> None:
        random_generator = getattr(self.trace_generator, "random_generator", None)
        fake = getattr(random_generator, "fake", None)
        if fake is not None and hasattr(fake, "seed_instance"):
            fake.seed_instance(seed)
