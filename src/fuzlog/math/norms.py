"""Norms: how two degrees combine into one.

A norm is any function of two broadcastable arrays of degrees. NumPy has three
already: `np.minimum` and `np.multiply` are t-norms (and), `np.maximum` is an
s-norm (or).
"""

from __future__ import annotations

import numpy as np

from fuzlog.types import Array


def probor(a: Array, b: Array) -> Array:
    """Probabilistic sum `a + b - a * b`: the s-norm that pairs with the product."""
    return a + b - a * b


def bounded_sum(a: Array, b: Array) -> Array:
    """Bounded sum `min(a + b, 1)`: the Lukasiewicz s-norm."""
    return np.minimum(a + b, 1.0)


def bounded_diff(a: Array, b: Array) -> Array:
    """Bounded difference `max(a + b - 1, 0)`: the Lukasiewicz t-norm."""
    return np.maximum(a + b - 1.0, 0.0)
