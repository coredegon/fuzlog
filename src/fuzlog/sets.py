"""Fuzzy sets: the membership function and its sixteen shapes.

A shape is a function that returns an :class:`MF`. Any other set is made by giving
``MF`` a function of your own, or by combining sets with ``& | ~ **``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from typing import overload

import numpy as np
from numpy.typing import ArrayLike

from fuzlog.types import Array, Kernel

_SMALL = 2048  # up to this many values np.interp wins: one call instead of several
_BLOCK = 1 << 14  # larger inputs are processed in blocks that stay in the CPU cache
_TINY = float(np.finfo(np.float64).tiny)


class MF:
    """A fuzzy set, defined by its membership function.

    ``fn`` maps a float64 array of at least one dimension to degrees in [0, 1] of the
    same shape and leaves its argument untouched. Calling the set takes anything
    array-like and gives a float for a scalar::

        MF(lambda x: np.exp(-np.abs(x)), "spike")(0.0)  # 1.0

    Sets combine pointwise: ``a & b`` is the minimum, ``a | b`` the maximum, ``~a``
    the complement and ``a ** p`` a hedge that concentrates (``p > 1``) or dilates.
    """

    __slots__ = ("fn", "label")

    def __init__(self, fn: Kernel, label: str = "mf") -> None:
        self.fn = fn
        self.label = label

    @overload
    def __call__(self, x: float) -> float: ...
    @overload
    def __call__(self, x: ArrayLike) -> Array: ...
    def __call__(self, x: ArrayLike) -> Array | float:
        """Return the degree of membership of every value in ``x``."""
        values = np.asarray(x, dtype=np.float64)
        return self.fn(values) if values.ndim else self.fn(values[None]).item()

    def __and__(self, other: MF) -> MF:
        if not isinstance(other, MF):
            return NotImplemented
        return MF(lambda x: np.minimum(self.fn(x), other.fn(x)), f"({self} & {other})")

    def __or__(self, other: MF) -> MF:
        if not isinstance(other, MF):
            return NotImplemented
        return MF(lambda x: np.maximum(self.fn(x), other.fn(x)), f"({self} | {other})")

    def __invert__(self) -> MF:
        return MF(lambda x: 1.0 - self.fn(x), f"~{self}")

    def __pow__(self, power: float) -> MF:
        if not power > 0.0:
            raise ValueError(f"a set is raised to a positive power, got {power}")
        return MF(lambda x: self.fn(x) ** power, f"{self}**{power:g}")

    def __bool__(self) -> bool:
        raise TypeError("a MF has no truth value: combine with & | ~, not and/or/not")

    def __repr__(self) -> str:
        return self.label


# --- piecewise-linear shapes ----------------------------------------------------------


def _pwl(points: Iterable[tuple[float, float]]) -> Kernel:
    """Build the kernel of a piecewise-linear function that is flat outside its points.

    A repeated x is a jump, at which the function takes the larger value, so cores
    stay closed. Small inputs go through ``np.interp``; large ones are summed ramp by
    ramp in cache-sized blocks, which is several times faster and agrees with
    ``np.interp`` to the last bit or two.
    """
    pairs = ((float(x), float(y)) for x, y in points)
    unique = [point for point, _ in itertools.groupby(pairs)]
    xp = np.array([x for x, _ in unique])
    fp = np.array([y for _, y in unique])
    for i in np.flatnonzero(np.diff(xp) == 0.0):  # a jump lies between two floats
        if fp[i + 1] > fp[i]:
            xp[i] = min(np.nextafter(xp[i], -np.inf), xp[i] - _TINY)
        else:
            xp[i + 1] = max(np.nextafter(xp[i + 1], np.inf), xp[i + 1] + _TINY)
    ramps = []
    starts, widths, rises = xp[:-1].tolist(), np.diff(xp).tolist(), np.diff(fp).tolist()
    for start, width, rise in zip(starts, widths, rises, strict=True):
        if rise:
            slope = rise / width
            if abs(width * slope) < abs(rise):  # the ramp must reach its end exactly
                slope = math.nextafter(slope, math.copysign(math.inf, slope))
            ramps.append((start, slope, min(0.0, rise), max(0.0, rise)))
    base = float(fp[0])

    def sum_ramps(x: Array, out: Array, tmp: Array) -> None:
        start, slope, low, high = ramps[0]
        np.subtract(x, start, out=out)
        out *= slope
        np.clip(out, low, high, out=out)
        for start, slope, low, high in ramps[1:]:
            np.subtract(x, start, out=tmp)
            tmp *= slope
            np.clip(tmp, low, high, out=tmp)
            out += tmp
        if base:
            out += base

    exact = False  # ramps are used only if they reproduce every point exactly
    if ramps:
        probe = np.empty_like(xp)
        with np.errstate(over="ignore", invalid="ignore"):
            sum_ramps(xp, probe, np.empty_like(xp))
        exact = np.array_equal(probe, fp)

    def kernel(x: Array) -> Array:
        if x.size <= _SMALL or not exact:
            return np.interp(x, xp, fp)
        out = np.empty(x.shape)
        values, degrees = x.reshape(-1), out.reshape(-1)
        tmp = np.empty(_BLOCK)
        with np.errstate(over="ignore"):  # a jump is a ramp steep enough to overflow
            for i in range(0, values.size, _BLOCK):
                block = values[i : i + _BLOCK]
                sum_ramps(block, degrees[i : i + _BLOCK], tmp[: block.size])
        return out

    return kernel


def _label(name: str, *parameters: float) -> str:
    return f"{name}({', '.join(f'{p:g}' for p in parameters)})"


def _ascending(*values: float) -> None:
    ordered = all(p <= q for p, q in itertools.pairwise(values))
    if not (ordered and all(map(math.isfinite, values))):
        raise ValueError(f"parameters must be finite and ascending, got {values}")


def _positive(**values: float) -> None:
    for name, value in values.items():
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be positive and finite, got {value}")


def triangle(a: float, b: float, c: float) -> MF:
    """Triangle that rises from ``a`` to its peak at ``b`` and falls to ``c``."""
    _ascending(a, b, c)
    return MF(_pwl([(a, 0), (b, 1), (c, 0)]), _label("triangle", a, b, c))


def trapezoid(a: float, b: float, c: float, d: float) -> MF:
    """Trapezoid with support ``(a, d)`` and core ``[b, c]``."""
    _ascending(a, b, c, d)
    points = [(a, 0), (b, 1), (c, 1), (d, 0)]
    return MF(_pwl(points), _label("trapezoid", a, b, c, d))


def rising(a: float, b: float) -> MF:
    """Linear rise from 0 at ``a`` to 1 from ``b`` on: a right shoulder."""
    _ascending(a, b)
    return MF(_pwl([(a, 0), (b, 1)]), _label("rising", a, b))


def falling(a: float, b: float) -> MF:
    """Linear fall from 1 up to ``a`` to 0 at ``b``: a left shoulder."""
    _ascending(a, b)
    return MF(_pwl([(a, 1), (b, 0)]), _label("falling", a, b))


def rectangle(a: float, b: float) -> MF:
    """Crisp interval: 1 on ``[a, b]`` and 0 elsewhere."""
    _ascending(a, b)
    points = [(a, 0), (a, 1), (b, 1), (b, 0)]
    return MF(_pwl(points), _label("rectangle", a, b))


def singleton(a: float) -> MF:
    """Crisp point: 1 at ``a`` and 0 elsewhere."""
    _ascending(a)
    return MF(_pwl([(a, 0), (a, 1), (a, 0)]), _label("singleton", a))


def piecewise(points: Iterable[tuple[float, float]]) -> MF:
    """Straight lines through ``(x, degree)`` points, flat before and after them.

    The points are listed by ascending x; repeating an x makes a jump.
    """
    pairs = [(float(x), float(y)) for x, y in points]
    _ascending(*(x for x, _ in pairs))
    if not pairs or not all(0.0 <= y <= 1.0 for _, y in pairs):
        raise ValueError(f"piecewise needs points with degrees in [0, 1], got {pairs}")
    label = ", ".join(f"({x:g}, {y:g})" for x, y in pairs)
    return MF(_pwl(pairs), f"piecewise([{label}])")


# --- smooth shapes --------------------------------------------------------------------


def gauss(c: float, sigma: float) -> MF:
    """Gaussian bell ``exp(-(x - c)**2 / (2 * sigma**2))``."""
    _positive(sigma=sigma)
    scale = -0.5 / sigma**2

    def kernel(x: Array) -> Array:
        t = x - c
        t *= t
        t *= scale
        return np.exp(t, out=t)

    return MF(kernel, _label("gauss", c, sigma))


def gauss2(c1: float, sigma1: float, c2: float, sigma2: float) -> MF:
    """Two-sided Gaussian: one bell left of ``c1``, another right of ``c2``, 1 between.

    If ``c1 > c2`` the two sides overlap and the peak stays below 1.
    """
    _positive(sigma1=sigma1, sigma2=sigma2)
    scale1, scale2 = -0.5 / sigma1**2, -0.5 / sigma2**2

    def kernel(x: Array) -> Array:
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

    return MF(kernel, _label("gauss2", c1, sigma1, c2, sigma2))


def bell(c: float, width: float, slope: float) -> MF:
    """Generalized bell ``1 / (1 + |(x - c) / width|**(2 * slope))``."""
    _positive(width=width, slope=slope)

    def kernel(x: Array) -> Array:
        t = x - c
        t /= width
        t *= t
        if slope != 1.0:
            np.power(t, slope, out=t)
        t += 1.0
        return np.reciprocal(t, out=t)

    return MF(kernel, _label("bell", c, width, slope))


def _sigmoid(c: float, slope: float) -> Kernel:
    def kernel(x: Array) -> Array:
        t = x - c
        t *= -slope
        with np.errstate(over="ignore"):  # exp overflows to inf, which gives 0
            np.exp(t, out=t)
        t += 1.0
        return np.reciprocal(t, out=t)

    return kernel


def sigmoid(c: float, slope: float) -> MF:
    """Sigmoid ``1 / (1 + exp(-slope * (x - c)))``; a negative slope makes it fall."""
    return MF(_sigmoid(c, slope), _label("sigmoid", c, slope))


def dsigmoid(c1: float, slope1: float, c2: float, slope2: float) -> MF:
    """Absolute difference of two sigmoids: a plateau between ``c1`` and ``c2``."""
    first, second = _sigmoid(c1, slope1), _sigmoid(c2, slope2)

    def kernel(x: Array) -> Array:
        t = first(x)
        t -= second(x)
        return np.abs(t, out=t)

    return MF(kernel, _label("dsigmoid", c1, slope1, c2, slope2))


def psigmoid(c1: float, slope1: float, c2: float, slope2: float) -> MF:
    """Product of two sigmoids; opposite slopes make a plateau between the centres."""
    first, second = _sigmoid(c1, slope1), _sigmoid(c2, slope2)

    def kernel(x: Array) -> Array:
        t = first(x)
        t *= second(x)
        return t

    return MF(kernel, _label("psigmoid", c1, slope1, c2, slope2))


def _s_curve(a: float, b: float) -> Kernel:
    """Kernel of the spline that rises from 0 at ``a`` to 1 at ``b``."""
    _ascending(a, b)
    if a == b:
        raise ValueError(f"a spline needs a < b, got {a} and {b}")
    scale = 1.0 / (b - a)

    def kernel(x: Array) -> Array:
        t = x - a
        t *= scale
        np.clip(t, 0.0, 1.0, out=t)
        u = 1.0 - t
        np.minimum(t, u, out=u)  # distance to the nearer end
        u *= u
        u *= 2.0  # the lower half of the curve; the upper half mirrors it
        return np.subtract(1.0, u, out=u, where=t > 0.5)

    return kernel


def s_shape(a: float, b: float) -> MF:
    """S-shaped spline that rises smoothly from 0 at ``a`` to 1 at ``b``."""
    return MF(_s_curve(a, b), _label("s_shape", a, b))


def z_shape(a: float, b: float) -> MF:
    """Z-shaped spline that falls smoothly from 1 at ``a`` to 0 at ``b``."""
    rise = _s_curve(a, b)

    def kernel(x: Array) -> Array:
        t = rise(x)
        return np.subtract(1.0, t, out=t)

    return MF(kernel, _label("z_shape", a, b))


def pi_shape(a: float, b: float, c: float, d: float) -> MF:
    """Pi-shaped spline: rises from ``a`` to ``b``, is 1 until ``c``, falls to ``d``."""
    _ascending(a, b, c, d)
    rise, fall = _s_curve(a, b), _s_curve(c, d)

    def kernel(x: Array) -> Array:
        t = fall(x)
        np.subtract(1.0, t, out=t)
        t *= rise(x)
        return t

    return MF(kernel, _label("pi_shape", a, b, c, d))
