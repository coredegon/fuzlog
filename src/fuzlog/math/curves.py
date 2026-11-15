"""Membership curves: the shapes of fuzzy sets.

Each function takes the parameters of a shape and returns its curve: a function
of a float64 array, of one dimension or more, that returns a new array of degrees
in [0, 1].
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable

import numpy as np

from fuzlog.types import Array, Curve

_SMALL = 2048  # up to this size np.interp is the fastest piecewise-linear curve
_BLOCK = 1 << 14  # larger arrays are worked on in blocks that fit the CPU cache
_TINY = float(np.finfo(np.float64).tiny)

type Ramp = tuple[float, float, float, float]  # start, slope, lowest, highest


# --- piecewise-linear -----------------------------------------------------------------


def piecewise(points: Iterable[tuple[float, float]]) -> Curve:
    """Straight lines through `(x, degree)` points, flat before and after them.

    The points go by ascending x. A repeated x is a jump, at which the curve takes
    the larger degree.
    """
    xp, fp = _knots(points)
    base, ramps = float(fp[0]), _ramps(xp, fp)

    def curve(x: Array) -> Array:
        if x.size <= _SMALL or not ramps:
            return np.interp(x, xp, fp)
        out = np.empty(x.shape)
        values, degrees, spare = x.reshape(-1), out.reshape(-1), np.empty(_BLOCK)
        with np.errstate(over="ignore"):  # a jump is a ramp steep enough to overflow
            for i in range(0, values.size, _BLOCK):
                block = values[i : i + _BLOCK]
                _climb(block, base, ramps, degrees[i : i + _BLOCK], spare[: block.size])
        return out

    return curve


def _knots(points: Iterable[tuple[float, float]]) -> tuple[Array, Array]:
    """Check the points and open every jump, so that x strictly ascends."""
    pairs = [(float(x), float(y)) for x, y in points]
    xs = [x for x, _ in pairs]
    if not all(map(math.isfinite, xs)) or xs != sorted(xs):
        raise ValueError(f"parameters must be finite and ascending, got {xs}")
    if not pairs or not all(0.0 <= y <= 1.0 for _, y in pairs):
        raise ValueError(f"a curve needs points with degrees in [0, 1], got {pairs}")
    unique = [pair for pair, _ in itertools.groupby(pairs)]
    xp, fp = (np.array(column) for column in zip(*unique, strict=True))
    for i in np.flatnonzero(np.diff(xp) == 0.0):
        # The lower end of a jump steps one float aside: the larger degree stays.
        if fp[i + 1] > fp[i]:
            xp[i] = min(np.nextafter(xp[i], -np.inf), xp[i] - _TINY)
        else:
            xp[i + 1] = max(np.nextafter(xp[i + 1], np.inf), xp[i + 1] + _TINY)
    return xp, fp


def _ramps(xp: Array, fp: Array) -> list[Ramp]:
    """Split a curve into ramps that sum to it: `clip((x - start) * slope, ...)`.

    Summing ramps is several times faster than `np.interp` on large arrays. No
    ramps are returned when rounding keeps them from hitting every point exactly.
    """
    ramps = []
    steps = zip(
        xp[:-1].tolist(), np.diff(xp).tolist(), np.diff(fp).tolist(), strict=True
    )
    for start, width, rise in steps:
        if rise:
            slope = rise / width
            if abs(width * slope) < abs(rise):  # the ramp must reach its end
                slope = math.nextafter(slope, math.copysign(math.inf, slope))
            ramps.append((start, slope, min(0.0, rise), max(0.0, rise)))
    if ramps:
        probe = np.empty_like(xp)
        with np.errstate(over="ignore", invalid="ignore"):
            _climb(xp, float(fp[0]), ramps, probe, np.empty_like(xp))
        if not np.array_equal(probe, fp):
            return []
    return ramps


def _climb(x: Array, base: float, ramps: list[Ramp], out: Array, spare: Array) -> None:
    """Write `base` plus the sum of the ramps at `x` into `out`."""
    start, slope, low, high = ramps[0]
    np.subtract(x, start, out=out)
    out *= slope
    np.clip(out, low, high, out=out)
    for start, slope, low, high in ramps[1:]:
        np.subtract(x, start, out=spare)
        spare *= slope
        np.clip(spare, low, high, out=spare)
        out += spare
    if base:
        out += base


def triangle(a: float, b: float, c: float) -> Curve:
    """Rises from `a` to its peak at `b` and falls to `c`."""
    return piecewise([(a, 0), (b, 1), (c, 0)])


def trapezoid(a: float, b: float, c: float, d: float) -> Curve:
    """Rises from `a` to `b`, is 1 until `c` and falls to `d`."""
    return piecewise([(a, 0), (b, 1), (c, 1), (d, 0)])


def rising(a: float, b: float) -> Curve:
    """0 up to `a`, then linear, 1 from `b` on."""
    return piecewise([(a, 0), (b, 1)])


def falling(a: float, b: float) -> Curve:
    """1 up to `a`, then linear, 0 from `b` on."""
    return piecewise([(a, 1), (b, 0)])


def rectangle(a: float, b: float) -> Curve:
    """1 on `[a, b]` and 0 elsewhere."""
    return piecewise([(a, 0), (a, 1), (b, 1), (b, 0)])


def singleton(a: float) -> Curve:
    """1 at `a` and 0 elsewhere."""
    return piecewise([(a, 0), (a, 1), (a, 0)])


# --- smooth ---------------------------------------------------------------------------


def _positive(**parameters: float) -> None:
    for name, value in parameters.items():
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be positive and finite, got {value}")


def gauss(c: float, sigma: float) -> Curve:
    """`exp(-(x - c)**2 / (2 * sigma**2))`."""
    _positive(sigma=sigma)
    scale = -0.5 / sigma**2

    def curve(x: Array) -> Array:
        t = x - c
        t *= t
        t *= scale
        return np.exp(t, out=t)

    return curve


def gauss2(c1: float, sigma1: float, c2: float, sigma2: float) -> Curve:
    """`gauss(c1, sigma1)` left of `c1`, 1 until `c2`, `gauss(c2, sigma2)` after it.

    If `c1 > c2` the two sides overlap and the peak stays below 1.
    """
    _positive(sigma1=sigma1, sigma2=sigma2)
    scale1, scale2 = -0.5 / sigma1**2, -0.5 / sigma2**2

    def curve(x: Array) -> Array:
        left = np.minimum(x, c1)
        left -= c1
        left *= left
        left *= scale1
        right = np.maximum(x, c2)
        right -= c2
        right *= right
        right *= scale2
        left += right
        return np.exp(left, out=left)

    return curve


def bell(c: float, width: float, slope: float) -> Curve:
    """`1 / (1 + |(x - c) / width|**(2 * slope))`."""
    _positive(width=width, slope=slope)

    def curve(x: Array) -> Array:
        t = x - c
        t /= width
        t *= t
        if slope != 1.0:
            np.power(t, slope, out=t)
        t += 1.0
        return np.reciprocal(t, out=t)

    return curve


def sigmoid(c: float, slope: float) -> Curve:
    """`1 / (1 + exp(-slope * (x - c)))`; a negative slope makes it fall."""

    def curve(x: Array) -> Array:
        t = x - c
        t *= -slope
        with np.errstate(over="ignore"):  # exp overflows to inf, which gives 0
            np.exp(t, out=t)
        t += 1.0
        return np.reciprocal(t, out=t)

    return curve


def dsigmoid(c1: float, slope1: float, c2: float, slope2: float) -> Curve:
    """`|sigmoid(c1, slope1) - sigmoid(c2, slope2)|`: a plateau between the centres."""
    first, second = sigmoid(c1, slope1), sigmoid(c2, slope2)

    def curve(x: Array) -> Array:
        t = first(x)
        t -= second(x)
        return np.abs(t, out=t)

    return curve


def psigmoid(c1: float, slope1: float, c2: float, slope2: float) -> Curve:
    """`sigmoid(c1, slope1) * sigmoid(c2, slope2)`; opposite slopes make a plateau."""
    first, second = sigmoid(c1, slope1), sigmoid(c2, slope2)

    def curve(x: Array) -> Array:
        t = first(x)
        t *= second(x)
        return t

    return curve


def s_shape(a: float, b: float) -> Curve:
    """Spline from 0 at `a` to 1 at `b`: `2t**2`, then `1 - 2(1 - t)**2`."""
    if not (math.isfinite(a) and math.isfinite(b) and a < b):
        raise ValueError(f"a spline needs finite a < b, got {a} and {b}")
    scale = 1.0 / (b - a)

    def curve(x: Array) -> Array:
        t = x - a
        t *= scale
        np.clip(t, 0.0, 1.0, out=t)
        u = 1.0 - t
        np.minimum(t, u, out=u)  # distance to the nearer end
        u *= u
        u *= 2.0  # the lower half of the curve; the upper half mirrors it
        return np.subtract(1.0, u, out=u, where=t > 0.5)

    return curve


def z_shape(a: float, b: float) -> Curve:
    """Spline from 1 at `a` to 0 at `b`: `1 - s_shape(a, b)`."""
    rise = s_shape(a, b)

    def curve(x: Array) -> Array:
        t = rise(x)
        return np.subtract(1.0, t, out=t)

    return curve


def pi_shape(a: float, b: float, c: float, d: float) -> Curve:
    """`s_shape(a, b) * z_shape(c, d)`: rises, is 1 from `b` to `c`, falls."""
    if not b <= c:
        raise ValueError(f"parameters must be ascending, got {(a, b, c, d)}")
    rise, fall = s_shape(a, b), z_shape(c, d)

    def curve(x: Array) -> Array:
        t = rise(x)
        t *= fall(x)
        return t

    return curve
