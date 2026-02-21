"""Surprisal computation - requires a backend from load_surprisal_model"""
from backends import load_surprisal_model
from backends.base import SurprisalBackend

__all__ = ["get_surprisal", "load_surprisal_model", "SurprisalBackend"]


def get_surprisal(
    backend: SurprisalBackend,
    prefix: str,
    word: str,
    base: float = 2.0
) -> float:
    """Calculate the surprisal of a word given a prefix using the provided backend.

    Surprisal is defined as: -log_base(P(word|prefix))

    Args:
        backend: SurprisalBackend instance from load_surprisal_model()
        prefix: Context before the word (leading/trailing whitespace is stripped)
        word: The word to score (leading/trailing whitespace is stripped)
        base: Log base (default 2.0 for bits, use e for nats)

    Returns:
        Surprisal in the requested units

    Example:
        >>> backend = load_surprisal_model("gpt2")
        >>> get_surprisal(backend, "The cat sat on the", "mat")
        4.23
    """
    return backend.get_surprisal(prefix=prefix, word=word, base=base)


if __name__ == "__main__":
    print("Testing surprisal calculation...")
    backend = load_surprisal_model("gpt2")
    print("\nGPT-2 example:")
    s = get_surprisal(backend, "The dog chased the", "cat.")
    print(f"Surprisal of 'cat.' given 'The dog chased the': {s:.4f} bits")
    s = get_surprisal(backend, "The dog chased the", "of.")
    print(f"Surprisal of 'of.' given 'The dog chased the': {s:.4f} bits")
    s = get_surprisal(backend, "The dog", "barked.")
    print(f"Surprisal of 'barked.' given 'The dog': {s:.4f} bits")
