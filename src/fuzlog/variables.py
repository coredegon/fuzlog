"""Linguistic variables, and the partitions that build them."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike

from fuzlog.math import curves
from fuzlog.rules import Prop
from fuzlog.sets import Set, falling, rising, trapezoid, triangle
from fuzlog.types import Array

type Core = float | tuple[float, float]
type Span = tuple[float, float]


class Variable:
    """A linguistic variable: a quantity and the terms that describe it.

    ```python
    temp = fz.Variable("temp", (0, 40), cold=fz.falling(10, 20), hot=fz.rising(20, 30))
    temp.cold          # the proposition "temp is cold", for rules
    temp(12.5)         # array([0.75, 0.  ]): the degree of every term
    temp.terms         # {"cold": falling(10, 20), "hot": rising(20, 30)}
    ```

    Terms also come as a mapping, which takes any name: `temp["very hot"]`.
    A `periodic` universe is a circle, like an angle: `high` is `low` again.
    """

    __slots__ = ("name", "periodic", "terms", "universe")

    def __init__(
        self,
        name: str,
        universe: tuple[float, float],
        terms: Mapping[str, Set] | None = None,
        /,
        *,
        periodic: bool = False,
        **more: Set,
    ) -> None:
        low, high = (float(bound) for bound in universe)
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError(f"a universe needs finite low < high, got {universe}")
        sets = {**(terms or {}), **more}
        strangers = [term for term, value in sets.items() if not isinstance(value, Set)]
        if not sets or strangers:
            raise TypeError(f"{name!r} needs terms that are sets, got {strangers}")
        self.name = name
        self.universe = (low, high)
        self.terms = MappingProxyType(sets)
        self.periodic = periodic

    def __getitem__(self, term: str) -> Prop:
        if term not in self.terms:
            raise KeyError(
                f"{self.name!r} has no term {term!r}, only {list(self.terms)}"
            )
        return Prop("is", self, term)

    def __getattr__(self, term: str) -> Prop:
        if term.startswith("_") or term in Variable.__slots__:
            raise AttributeError(term)
        try:
            return self[term]
        except KeyError as error:
            raise AttributeError(error.args[0]) from None

    def __call__(self, x: ArrayLike) -> Array:
        """Return the degree of `x` in every term: shape `(len(terms), *x.shape)`."""
        values = np.asarray(x, dtype=np.float64)
        wrapped = self.wrap(np.atleast_1d(values))
        degrees = np.empty((len(self.terms), *wrapped.shape))
        for i, term in enumerate(self.terms.values()):
            degrees[i] = term.curve(wrapped)
        return degrees.reshape(len(self.terms), *values.shape)

    def grid(self, points: int = 101) -> Array:
        """Return evenly spaced values that cover the universe."""
        return np.linspace(*self.universe, points, endpoint=not self.periodic)

    def wrap(self, x: Array) -> Array:
        """Bring the values of a periodic variable into `[low, high)`."""
        low, high = self.universe
        if not self.periodic or x.size == 0 or (x.min() >= low and x.max() < high):
            return x
        with np.errstate(invalid="ignore"):  # an infinite value has no place
            return low + np.mod(x - low, high - low)

    def __repr__(self) -> str:
        low, high = self.universe
        terms = ", ".join(f"{name!r}: {term}" for name, term in self.terms.items())
        periodic = ", periodic=True" if self.periodic else ""
        return f"Variable({self.name!r}, ({low:g}, {high:g}), {{{terms}}}{periodic})"


def partition(
    name: str,
    universe: tuple[float, float],
    cores: Mapping[str, Core] | Iterable[str] | None = None,
    /,
    *,
    periodic: bool = False,
    **more: Core,
) -> Variable:
    """Build a variable whose terms always sum to 1.

    ```python
    fz.partition("level", (0, 1), ["low", "medium", "high"])   # evenly spread
    fz.partition("temp", (0, 40), cold=10, warm=(18, 22), hot=30)
    fz.partition("hue", (0, 360), red=(350, 10), green=120, blue=240, periodic=True)
    ```

    A core is the value or `(low, high)` range that belongs to a term completely;
    between neighbouring cores membership passes linearly from one term to the
    next. Cores go in increasing order; on a `periodic` universe the last term
    blends back into the first and one core may wrap around.
    """
    low, high = (float(bound) for bound in universe)
    if cores is None or isinstance(cores, Mapping):
        named: dict[str, Core] = {**(cores or {}), **more}
    elif more:
        raise TypeError("give either term names or cores, not both")
    else:
        names = list(cores)
        if len(set(names)) < len(names):
            raise ValueError(f"the terms of a partition must differ, got {names}")
        even = np.linspace(low, high, len(names), endpoint=not periodic).tolist()
        named = dict(zip(names, even, strict=True))
    if len(named) < 2:
        raise ValueError(f"a partition needs at least two terms, got {list(named)}")
    ranges = [c if isinstance(c, tuple | list) else (c, c) for c in named.values()]
    spans = [(float(lower), float(upper)) for lower, upper in ranges]
    sets = _ring(spans, low, high) if periodic else _line(spans, low, high)
    return Variable(
        name, (low, high), dict(zip(named, sets, strict=True)), periodic=periodic
    )


def _bump(a: float, b: float, c: float, d: float) -> Set:
    """A term that rises from `a` to its core `[b, c]` and falls to `d`."""
    return triangle(a, b, d) if b == c else trapezoid(a, b, c, d)


def _line(spans: list[Span], low: float, high: float) -> list[Set]:
    """Blend consecutive cores along a line; the outer terms become shoulders."""
    flat = [low, *itertools.chain.from_iterable(spans), high]
    apart = all(left[1] < right[0] for left, right in itertools.pairwise(spans))
    if flat != sorted(flat) or not apart:
        raise ValueError(
            f"cores must increase inside the universe ({low:g}, {high:g}) "
            f"and leave gaps between neighbours, got {spans}"
        )
    inner = [
        _bump(before[1], *core, after[0])
        for before, core, after in zip(spans, spans[1:], spans[2:], strict=False)
    ]
    return [
        falling(spans[0][1], spans[1][0]),
        *inner,
        rising(spans[-2][1], spans[-1][0]),
    ]


def _ring(spans: list[Span], low: float, high: float) -> list[Set]:
    """Blend consecutive cores around a circle; the last blends into the first."""
    period = high - low
    bumps = []
    for i, (lower, upper) in enumerate(spans):
        before, after = spans[i - 1][1], spans[(i + 1) % len(spans)][0]
        a = low + (before - low) % period  # going around: each corner follows the last
        b = a + (lower - before) % period
        c = b + (upper - lower) % period
        bumps.append((a, b, c, c + (after - upper) % period))
    covered = sum(d - b for _, b, _, d in bumps)
    if not math.isclose(covered, period) or any(c == d for *_, c, d in bumps):
        raise ValueError(
            f"cores must go once around the circle of {period:g} in increasing "
            f"order and leave gaps between neighbours, got {spans}"
        )
    sets = []
    for bump in bumps:
        term = _bump(*bump)
        if bump[3] > high:  # it sticks out past `high`: a copy comes in at `low`
            corners = [corner - period for corner in bump] + list(bump)
            outline = zip(corners, [0, 1, 1, 0] * 2, strict=True)
            term = Set(curves.piecewise(outline), term.name)
        sets.append(term)
    return sets
