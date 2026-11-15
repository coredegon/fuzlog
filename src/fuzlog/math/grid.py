"""Lookup in values laid on an even grid: one dimension per input.

`locate` turns inputs into positions between the nodes of an axis; `nearest` and
`linear` read the values at such positions.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from fuzlog.types import Array


def locate(x: Array, low: float, high: float, points: int, periodic: bool) -> Array:
    """Return the position of `x` on an axis of `points` nodes, counted in nodes.

    Values outside the axis move to its nearest end; a periodic axis, whose last
    node lies one step before `high`, wraps them around. NaN stays NaN.
    """
    if periodic:
        with np.errstate(invalid="ignore"):  # an infinite value becomes NaN
            at = np.mod(x - low, high - low)
        at *= points / (high - low)
        return at
    at = (x - low) * ((points - 1) / (high - low))
    return np.clip(at, 0.0, points - 1.0, out=at)


def _strides(shape: tuple[int, ...]) -> list[int]:
    """Distance between neighbours along every axis of flattened values."""
    return [math.prod(shape[i + 1 :]) for i in range(len(shape))]


def nearest(
    values: np.ndarray, places: Sequence[Array], periodic: Sequence[bool]
) -> np.ndarray:
    """Return the value of the node closest to every position."""
    cell: NDArray[np.intp] | int = 0
    for at, size, stride, wraps in zip(
        places, values.shape, _strides(values.shape), periodic, strict=True
    ):
        node = np.rint(at).astype(np.intp)
        if wraps:
            node %= size
        node *= stride
        cell = cell + node
    return values.reshape(-1)[cell]


def linear(
    values: np.ndarray,
    places: Sequence[Array],
    periodic: Sequence[bool],
    holes: bool = False,
) -> Array:
    """Return the values of the nodes around every position, weighted by closeness.

    With `holes`, a NaN node spoils only the positions that lean on it.
    """
    sides = []  # per axis: the node below and the node above, with their weights
    for at, size, stride, wraps in zip(
        places, values.shape, _strides(values.shape), periodic, strict=True
    ):
        below = at.astype(np.intp)
        if wraps:
            share = at - below
            below %= size
            above = (below + 1) % size
        else:
            np.minimum(below, size - 2, out=below)
            share = at - below
            above = below + 1
        sides.append(((below * stride, 1.0 - share), (above * stride, share)))
    cells = values.reshape(-1)
    out: Array | None = None
    for corner in itertools.product(*sides):
        cell, weight = corner[0]
        for other, share in corner[1:]:
            cell = cell + other
            weight = weight * share
        term = cells[cell] * weight
        if holes:  # a hole must not leak into where it has no weight
            term[np.broadcast_to(weight == 0.0, term.shape)] = 0.0
        out = term if out is None else out + term
    assert out is not None
    return out
