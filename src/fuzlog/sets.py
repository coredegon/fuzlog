"""Fuzzy sets: the membership function and its sixteen shapes.

A shape is a function that returns an :class:`MF`. Any other set is made by giving
``MF`` a function of your own, or by combining sets with ``& | ~ **``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import overload

import numpy as np
from numpy.typing import ArrayLike

from fuzlog.math import curves
from fuzlog.types import Array, Curve


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

    def __init__(self, fn: Curve, label: str = "mf") -> None:
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


# --- shapes: the curves of `fuzlog.math.curves` as sets -------------------------------


def _label(name: str, *parameters: float) -> str:
    return f"{name}({', '.join(f'{p:g}' for p in parameters)})"


def triangle(a: float, b: float, c: float) -> MF:
    """Triangle that rises from ``a`` to its peak at ``b`` and falls to ``c``."""
    return MF(curves.triangle(a, b, c), _label("triangle", a, b, c))


def trapezoid(a: float, b: float, c: float, d: float) -> MF:
    """Trapezoid with support ``(a, d)`` and core ``[b, c]``."""
    return MF(curves.trapezoid(a, b, c, d), _label("trapezoid", a, b, c, d))


def rising(a: float, b: float) -> MF:
    """Linear rise from 0 at ``a`` to 1 from ``b`` on: a right shoulder."""
    return MF(curves.rising(a, b), _label("rising", a, b))


def falling(a: float, b: float) -> MF:
    """Linear fall from 1 up to ``a`` to 0 at ``b``: a left shoulder."""
    return MF(curves.falling(a, b), _label("falling", a, b))


def rectangle(a: float, b: float) -> MF:
    """Crisp interval: 1 on ``[a, b]`` and 0 elsewhere."""
    return MF(curves.rectangle(a, b), _label("rectangle", a, b))


def singleton(a: float) -> MF:
    """Crisp point: 1 at ``a`` and 0 elsewhere."""
    return MF(curves.singleton(a), _label("singleton", a))


def gauss(c: float, sigma: float) -> MF:
    """Gaussian bell ``exp(-(x - c)**2 / (2 * sigma**2))``."""
    return MF(curves.gauss(c, sigma), _label("gauss", c, sigma))


def gauss2(c1: float, sigma1: float, c2: float, sigma2: float) -> MF:
    """Two-sided Gaussian: a bell left of ``c1``, another right of ``c2``."""
    return MF(
        curves.gauss2(c1, sigma1, c2, sigma2), _label("gauss2", c1, sigma1, c2, sigma2)
    )


def bell(c: float, width: float, slope: float) -> MF:
    """Generalized bell ``1 / (1 + |(x - c) / width|**(2 * slope))``."""
    return MF(curves.bell(c, width, slope), _label("bell", c, width, slope))


def sigmoid(c: float, slope: float) -> MF:
    """Sigmoid ``1 / (1 + exp(-slope * (x - c)))``."""
    return MF(curves.sigmoid(c, slope), _label("sigmoid", c, slope))


def dsigmoid(c1: float, slope1: float, c2: float, slope2: float) -> MF:
    """Absolute difference of two sigmoids."""
    return MF(
        curves.dsigmoid(c1, slope1, c2, slope2),
        _label("dsigmoid", c1, slope1, c2, slope2),
    )


def psigmoid(c1: float, slope1: float, c2: float, slope2: float) -> MF:
    """Product of two sigmoids."""
    return MF(
        curves.psigmoid(c1, slope1, c2, slope2),
        _label("psigmoid", c1, slope1, c2, slope2),
    )


def s_shape(a: float, b: float) -> MF:
    """S-shaped spline that rises smoothly from 0 at ``a`` to 1 at ``b``."""
    return MF(curves.s_shape(a, b), _label("s_shape", a, b))


def z_shape(a: float, b: float) -> MF:
    """Z-shaped spline that falls smoothly from 1 at ``a`` to 0 at ``b``."""
    return MF(curves.z_shape(a, b), _label("z_shape", a, b))


def pi_shape(a: float, b: float, c: float, d: float) -> MF:
    """Pi-shaped spline: rises from ``a`` to ``b``, falls from ``c`` to ``d``."""
    return MF(curves.pi_shape(a, b, c, d), _label("pi_shape", a, b, c, d))


def piecewise(points: Iterable[tuple[float, float]]) -> MF:
    """Straight lines through ``(x, degree)`` points, flat before and after them."""
    pairs = [(float(x), float(y)) for x, y in points]
    label = ", ".join(f"({x:g}, {y:g})" for x, y in pairs)
    return MF(curves.piecewise(pairs), f"piecewise([{label}])")
