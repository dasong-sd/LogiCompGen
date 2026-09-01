"""Configuration objects for empirical k-trace saturation analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_KS = (2, 3, 4)


@dataclass(frozen=True)
class ConvergenceConfig:
    batch_size: int = 20
    minimum_valid_traces: int = 100
    patience_batches: int = 5
    relative_gain_threshold: float = 0.01
    absolute_gain_threshold: int = 2
    k_values: tuple[int, ...] = DEFAULT_KS
    maximum_valid_traces: int = 5000

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.minimum_valid_traces < 0:
            raise ValueError("minimum_valid_traces must be non-negative")
        if self.patience_batches <= 0:
            raise ValueError("patience_batches must be positive")
        if self.relative_gain_threshold < 0:
            raise ValueError("relative_gain_threshold must be non-negative")
        if self.absolute_gain_threshold < 0:
            raise ValueError("absolute_gain_threshold must be non-negative")
        if self.maximum_valid_traces <= 0:
            raise ValueError("maximum_valid_traces must be positive")
        if any(k <= 0 for k in self.k_values):
            raise ValueError("k_values must be positive")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["k_values"] = list(self.k_values)
        return data


@dataclass(frozen=True)
class RetrospectiveConfig(ConvergenceConfig):
    batch_size: int = 10
    minimum_valid_traces: int = 40
    patience_batches: int = 3
    relative_gain_threshold: float = 0.01
    absolute_gain_threshold: int = 2
    k_values: tuple[int, ...] = DEFAULT_KS
    maximum_valid_traces: int = 80


@dataclass(frozen=True)
class BootstrapConfig:
    num_permutations: int = 1000
    random_seed: int = 42

    def validate(self) -> None:
        if self.num_permutations <= 0:
            raise ValueError("num_permutations must be positive")


def resolve_output_dir(repo_root: Path, output_dir: Path) -> Path:
    return output_dir if output_dir.is_absolute() else repo_root / output_dir
