"""Deterministic seed derivation for adaptive convergence experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SEED_DERIVATION_VERSION = "rolling-hash-legacy-compatible-v2"
_MAX_NUMPY_SEED = 2**32 - 1


@dataclass(frozen=True)
class CandidateSeeds:
    """All derived seeds used for one generated candidate trace."""

    master_seed: int
    domain_seed: int
    batch_seed: int
    benchmark_instance_seed: int
    mutation_seed: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def derive_seed(master_seed: int, *components: Any) -> int:
    """Derive a stable 32-bit child seed from a master seed and components.

    The rolling hash is intentionally backward-compatible with the candidate
    seed formula used by the original one-seed adaptive run when called as
    ``derive_seed(master_seed, domain, batch_index, attempt_index)``.
    """

    value = int(master_seed) & 0xFFFFFFFF
    for component in components:
        for char in str(component):
            value = ((value * 131) + ord(char)) & 0xFFFFFFFF
    return value % _MAX_NUMPY_SEED or 1


def derive_candidate_seeds(master_seed: int, domain: str, batch_index: int, instance_id: int | str) -> CandidateSeeds:
    """Derive domain, batch, instance, and mutation/fuzzing seeds."""

    domain_seed = derive_seed(master_seed, "domain", domain)
    batch_seed = derive_seed(master_seed, "batch", domain, int(batch_index))
    benchmark_instance_seed = derive_seed(
        master_seed,
        "benchmark_instance",
        domain,
        int(batch_index),
        str(instance_id),
    )
    mutation_seed = derive_seed(master_seed, domain, int(batch_index), str(instance_id))
    return CandidateSeeds(
        master_seed=int(master_seed),
        domain_seed=domain_seed,
        batch_seed=batch_seed,
        benchmark_instance_seed=benchmark_instance_seed,
        mutation_seed=mutation_seed,
    )


def seed_derivation_metadata() -> dict[str, object]:
    return {
        "version": SEED_DERIVATION_VERSION,
        "description": (
            "Child seeds are derived with a deterministic 32-bit rolling hash over "
            "the master seed and typed components. The mutation/fuzzing seed is "
            "backward-compatible with the candidate seed formula used by the original "
            "seed-42 run: derive_seed(master_seed, domain, batch_index, instance_id). "
            "The adaptive wrapper also records separate domain, batch, and "
            "benchmark-instance/allocation seeds for auditability. The mutation seed "
            "is used to seed Python random, NumPy, and Faker before each candidate "
            "generation attempt."
        ),
    }
