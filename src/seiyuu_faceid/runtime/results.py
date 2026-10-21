"""Identity classification results shared by the runtime and interfaces."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    identity: str
    score: float


@dataclass(frozen=True)
class Decision:
    status: str
    identity: str | None
    candidates: tuple[Candidate, ...]
    reason: str | None = None


def decide_candidates(candidates: Sequence[Candidate], *, probability_threshold: float) -> Decision:
    """Accept the highest-ranked identity only when its probability reaches the threshold."""
    if not 0 <= probability_threshold <= 1:
        raise ValueError("probability_threshold must be in [0, 1].")
    ranked = tuple(candidates)
    if not ranked or ranked[0].score < probability_threshold:
        return Decision("review", None, ranked, "low_probability")
    return Decision("recognized", ranked[0].identity, ranked)
