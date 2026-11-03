"""Holm correction for a complete research-session test family."""

import math


def holm_adjust(
    p_values: list[float],
    family_size: int | None = None,
) -> list[float]:
    """Return adjusted p-values in the original order.

    Unused family slots are treated as p=1.
    """
    size = len(p_values) if family_size is None else family_size

    if size < len(p_values):
        raise ValueError("family_size cannot be smaller than tests run")

    for value in p_values:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("p-values must be finite and between 0 and 1")

    order = sorted(
        range(len(p_values)),
        key=lambda index: p_values[index],
    )
    adjusted = [1.0] * len(p_values)
    previous = 0.0

    for rank, index in enumerate(order):
        value = min(1.0, (size - rank) * p_values[index])
        previous = max(previous, value)
        adjusted[index] = previous

    return adjusted
