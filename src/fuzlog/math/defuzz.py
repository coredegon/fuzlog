"""Defuzzifiers: how a sampled fuzzy set becomes one number.

Each takes a grid `x` and degrees `mu` whose last axis runs over that grid, and
returns one value per set: NaN for an empty set.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from fuzlog.types import Array


@lru_cache(maxsize=32)
def _weights(grid: bytes) -> tuple[Array, Array]:
    """Weights that integrate a set drawn as straight lines between its samples.

    `mu @ area` is the area under the outline and `mu @ moment` its first moment.
    """
    x = np.frombuffer(grid)
    half = np.diff(x) / 2.0
    area, moment = np.zeros_like(x), np.zeros_like(x)
    area[:-1] += half
    area[1:] += half
    moment += area * x
    tilt = half * half * (2.0 / 3.0)  # a sloping segment leans towards its higher end
    moment[:-1] += tilt
    moment[1:] -= tilt
    return area, moment


def centroid(x: Array, mu: Array) -> Array:
    """Centre of gravity of the area under `mu`.

    Exact when the grid holds every corner of the set.
    """
    area, moment = _weights(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    with np.errstate(invalid="ignore"):
        return (mu @ moment) / (mu @ area)


def bisector(x: Array, mu: Array) -> Array:
    """The value that splits the area under `mu` in two halves."""
    area = np.cumsum(mu, axis=-1)
    total = area[..., -1:]
    first = (area < total / 2.0).sum(axis=-1)  # where the running area reaches half
    return np.where(total[..., 0] > 0.0, x[np.minimum(first, x.size - 1)], np.nan)


def _top(mu: Array) -> np.ndarray:
    """Mark where `mu` is highest; nowhere for an empty set."""
    peak = mu.max(axis=-1, keepdims=True)
    return (mu == peak) & (peak > 0.0)


def mom(x: Array, mu: Array) -> Array:
    """Mean of the values at which `mu` is highest."""
    top = _top(mu)
    with np.errstate(invalid="ignore"):
        return (top @ x) / top.sum(axis=-1)


def som(x: Array, mu: Array) -> Array:
    """Smallest value at which `mu` is highest."""
    top = _top(mu)
    return np.where(top.any(axis=-1), x[top.argmax(axis=-1)], np.nan)


def lom(x: Array, mu: Array) -> Array:
    """Largest value at which `mu` is highest."""
    top = _top(mu)
    last = x.size - 1 - top[..., ::-1].argmax(axis=-1)
    return np.where(top.any(axis=-1), x[last], np.nan)
