"""Rare-k-gram diagnostics and Chao1 estimators."""

from __future__ import annotations

from .accumulation import kgram_frequency


def chao1_bias_corrected(observed_unique: int, singletons: int, doubletons: int) -> float:
    """Return bias-corrected Chao1, including the zero-doubleton case."""

    if observed_unique < 0 or singletons < 0 or doubletons < 0:
        raise ValueError("counts must be non-negative")
    return observed_unique + (singletons * (singletons - 1)) / (2 * (doubletons + 1))


def singleton_doubleton_summary(sequences, domain: str, method: str, ks) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k in ks:
        freq = kgram_frequency(sequences, int(k))
        unique = len(freq)
        singletons = sum(1 for count in freq.values() if count == 1)
        doubletons = sum(1 for count in freq.values() if count == 2)
        rows.append(
            {
                "domain": domain,
                "method": method,
                "k": int(k),
                "unique_kgrams": unique,
                "singleton_kgrams": singletons,
                "doubleton_kgrams": doubletons,
                "singleton_proportion": None if unique == 0 else singletons / unique,
                "chao1_bias_corrected": chao1_bias_corrected(unique, singletons, doubletons),
            }
        )
    return rows
