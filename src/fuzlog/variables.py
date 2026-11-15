"""Linguistic variables, and the partitions that build them."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, overload

import numpy as np
from numpy.typing import ArrayLike

from fuzlog.rules import Prop
from fuzlog.sets import MF, falling, piecewise, rising, trapezoid, triangle
from fuzlog.types import Array

if TYPE_CHECKING:
    from matplotlib.axes import Axes

type Core = float | tuple[float, float]


class Variable:
    """A linguistic variable: a quantity whose values are described by fuzzy terms.

    ``universe`` is the range ``(low, high)`` of the quantity. Terms are given as a
    mapping, as keyword arguments, or both::

        temp = Variable("temp", (0, 40), cold=falling(10, 20), hot=rising(20, 30))

    ``temp.cold`` and ``temp["cold"]`` are the proposition "temp is cold", the
    building block of rules; ``temp.terms["cold"]`` is the fuzzy set itself.

    A ``periodic`` universe is a circle on which ``high`` is the same point as
    ``low``, like an angle: values are wrapped into ``[low, high)`` before the terms
    see them.
    """

    __slots__ = ("name", "periodic", "terms", "universe")

    def __init__(
        self,
        name: str,
        universe: tuple[float, float],
        terms: Mapping[str, MF] | None = None,
        /,
        *,
        periodic: bool = False,
        **more: MF,
    ) -> None:
        low, high = (float(bound) for bound in universe)
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError(f"a universe needs finite low < high, got {universe}")
        sets = {**(terms or {}), **more}
        strangers = [term for term, mf in sets.items() if not isinstance(mf, MF)]
        if not sets or strangers:
            raise TypeError(
                f"{name!r} needs terms that are MF, got {strangers or sets}"
            )
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
        """Return the degree of ``x`` in every term: shape ``(len(terms), *x.shape)``.

        Unpacks into one array per term: ``cold, warm, hot = temp(x)``.
        """
        values = np.asarray(x, dtype=np.float64)
        wrapped = self._wrap(np.atleast_1d(values))
        degrees = np.empty((len(self.terms), *wrapped.shape))
        for i, mf in enumerate(self.terms.values()):
            degrees[i] = mf.fn(wrapped)
        return degrees.reshape(len(self.terms), *values.shape)

    @overload
    def fuzzify(self, x: float) -> dict[str, float]: ...
    @overload
    def fuzzify(self, x: ArrayLike) -> dict[str, Array]: ...
    def fuzzify(self, x: ArrayLike) -> dict[str, float] | dict[str, Array]:
        """Return the degree of ``x`` in every term, by term name."""
        degrees = self(x)
        values = degrees if degrees.ndim > 1 else degrees.tolist()
        return dict(zip(self.terms, values, strict=True))

    def grid(self, points: int = 101) -> Array:
        """Return evenly spaced values that cover the universe."""
        return np.linspace(*self.universe, points, endpoint=not self.periodic)

    def peaks(self, points: int = 2001) -> Array:
        """Return the most typical value of every term: the middle of its core.

        The core, where membership is highest, is located on a grid of ``points``
        values; on a periodic universe its middle is found going around the circle.
        """
        x = self.grid(points)
        degrees = self(x)
        core = (degrees >= degrees.max(axis=1, keepdims=True) - 1e-9).astype(np.float64)
        if not self.periodic:
            return (core @ x) / core.sum(axis=1)
        low, high = self.universe
        angle = (x - low) * (math.tau / (high - low))
        middle = np.arctan2(core @ np.sin(angle), core @ np.cos(angle)) % math.tau
        return low + middle * ((high - low) / math.tau)

    def plot(self, ax: Axes | None = None) -> Axes:
        """Draw the membership function of every term (needs matplotlib)."""
        from fuzlog.plotting import plot_variable

        return plot_variable(self, ax)

    def _wrap(self, x: Array) -> Array:
        """Bring the values of a periodic variable into ``[low, high)``."""
        low, high = self.universe
        if not self.periodic or x.size == 0 or (x.min() >= low and x.max() < high):
            return x
        return low + np.mod(x - low, high - low)

    def __repr__(self) -> str:
        low, high = self.universe
        terms = ", ".join(f"{term!r}: {mf}" for term, mf in self.terms.items())
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
    """Build a variable whose terms always sum to 1: a strong fuzzy partition.

    A term is given by its core, the value or ``(low, high)`` range that belongs to
    it completely; between neighbouring cores membership passes linearly from one
    term to the next. A list of names instead spreads the cores evenly::

        partition("temp", (0, 40), cold=10, warm=20, hot=30)
        partition("level", (0, 1), ["low", "medium", "high"])

    Cores are listed in increasing order, and the outer terms keep full membership
    up to the ends of the universe. On a ``periodic`` universe the last term blends
    back into the first and one core may wrap around: ``(350, 10)`` on ``(0, 360)``.
    """
    low, high = (float(bound) for bound in universe)
    if cores is None or isinstance(cores, Mapping):
        named = {**(cores or {}), **more}
        given = named.values()
        spans = [tuple(c) if isinstance(c, tuple | list) else (c, c) for c in given]
    elif more:
        raise TypeError("give either term names or cores, not both")
    else:
        named = dict.fromkeys(cores)
        even = np.linspace(low, high, len(named), endpoint=not periodic).tolist()
        spans = [(c, c) for c in even]
    if len(named) < 2:
        raise ValueError(f"a partition needs at least two terms, got {list(named)}")
    spans = [(float(lower), float(upper)) for lower, upper in spans]
    sets = _ring(spans, low, high) if periodic else _line(spans, low, high)
    terms = dict(zip(named, sets, strict=True))
    return Variable(name, (low, high), terms, periodic=periodic)


def _line(spans: list[tuple[float, float]], low: float, high: float) -> list[MF]:
    """Blend consecutive cores along a line; the outer terms become shoulders."""
    flat = [low, *itertools.chain.from_iterable(spans), high]
    apart = all(left[1] < right[0] for left, right in itertools.pairwise(spans))
    if flat != sorted(flat) or not apart:
        raise ValueError(
            f"cores must increase inside the universe ({low:g}, {high:g}) "
            f"and leave gaps between neighbours, got {spans}"
        )
    sets = [falling(spans[0][1], spans[1][0])]
    for i in range(1, len(spans) - 1):
        before, (lower, upper), after = spans[i - 1][1], spans[i], spans[i + 1][0]
        if lower == upper:
            sets.append(triangle(before, lower, after))
        else:
            sets.append(trapezoid(before, lower, upper, after))
    return [*sets, rising(spans[-2][1], spans[-1][0])]


def _ring(spans: list[tuple[float, float]], low: float, high: float) -> list[MF]:
    """Blend consecutive cores around a circle; the last blends into the first."""
    period = high - low
    corners = []
    for i, (lower, upper) in enumerate(spans):
        before, after = spans[i - 1][1], spans[(i + 1) % len(spans)][0]
        a = low + (before - low) % period
        b = a + (lower - before) % period
        c = b + (upper - lower) % period
        corners.append((a, b, c, c + (after - upper) % period))
    covered = sum(d - b for _, b, _, d in corners)
    if not math.isclose(covered, period) or any(c == d for *_, c, d in corners):
        raise ValueError(
            f"cores must go once around the circle of {period:g} in increasing "
            f"order and leave gaps between neighbours, got {spans}"
        )
    sets = []
    for a, b, c, d in corners:
        # The bump may stick out past ``high``: fold that part back onto the start.
        xs = np.array([a, b, c, d])
        x = np.concatenate([xs - period, xs])
        y = np.tile([0.0, 1.0, 1.0, 0.0], 2)
        inside = (x > low) & (x < high)
        seam = float(np.interp(low, x, y))
        bump = zip(x[inside].tolist(), y[inside].tolist(), strict=True)
        outline = piecewise([(low, seam), *bump, (high, seam)])
        shape, bends = (
            ("triangle", (a, b, d)) if b == c else ("trapezoid", (a, b, c, d))
        )
        label = f"{shape}({', '.join(f'{bend:g}' for bend in bends)})"
        sets.append(MF(outline.fn, label))
    return sets
