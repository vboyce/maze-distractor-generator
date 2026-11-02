"""Base protocol for surprisal backends"""
from typing import Protocol


class SurprisalBackend(Protocol):
    """Interface for computing word surprisal given a prefix"""

    # The exact model version loaded (e.g. a Hugging Face commit), or None if
    # there is none to record; written to the run record.
    model_commit: "str | None"

    def get_surprisal(self, prefix: str, word: str, base: float = 2.0) -> float:
        """Compute surprisal of word given prefix. Surprisal = -log_base(P(word|prefix))."""
        ...
