r"""Fuzzy sets: `Set` and the sixteen shapes that make one."""

from __future__ import annotations

from collections.abc import Iterable
from typing import overload

import numpy as np
from numpy.typing import ArrayLike

from fuzlog.math import curves
from fuzlog.types import Array, Curve


class Set:
    """A fuzzy set: call it with values to get their degrees of membership.

    ```python
    warm = fz.triangle(10, 20, 30)
    warm(15)              # 0.5
    warm([5, 15, 20])     # array([0. , 0.5, 1. ])
    warm & ~fz.gauss(20, 3)   # & min, | max, ~ complement, ** power
    ```

    Any function of a float64 array is a `curve`: `Set(np.cos, "wave")`.
    """

    __slots__ = ("curve", "name")

    def __init__(self, curve: Curve, name: str = "set") -> None:
        self.curve = curve
        self.name = name

    @overload
    def __call__(self, x: float) -> float: ...
    @overload
    def __call__(self, x: ArrayLike) -> Array: ...
    def __call__(self, x: ArrayLike) -> Array | float:
        """Return the degree of every value: a float for a scalar."""
        values = np.asarray(x, dtype=np.float64)
        return self.curve(values) if values.ndim else self.curve(values[None]).item()

    def __and__(self, other: Set) -> Set:
        if not isinstance(other, Set):
            return NotImplemented
        a, b = self.curve, other.curve
        return Set(lambda x: np.minimum(a(x), b(x)), f"({self} & {other})")

    def __or__(self, other: Set) -> Set:
        if not isinstance(other, Set):
            return NotImplemented
        a, b = self.curve, other.curve
        return Set(lambda x: np.maximum(a(x), b(x)), f"({self} | {other})")

    def __invert__(self) -> Set:
        curve = self.curve
        return Set(lambda x: 1.0 - curve(x), f"~{self}")

    def __pow__(self, power: float) -> Set:
        if not power > 0.0:
            raise ValueError(f"a set is raised to a positive power, got {power}")
        curve = self.curve
        return Set(lambda x: curve(x) ** power, f"{self}**{power:g}")

    def __bool__(self) -> bool:
        raise TypeError("a Set has no truth value: combine with & | ~, not and/or/not")

    def __repr__(self) -> str:
        return self.name


def _name(shape: str, *parameters: float) -> str:
    return f"{shape}({', '.join(f'{p:g}' for p in parameters)})"


# --- straight lines -------------------------------------------------------------------


def triangle(a: float, b: float, c: float) -> Set:
    r"""Rises from `a` to its peak at `b` and falls to `c`.

    ```
         /\
    ____/  \____
        a b c
    ```
    """
    return Set(curves.triangle(a, b, c), _name("triangle", a, b, c))


def trapezoid(a: float, b: float, c: float, d: float) -> Set:
    r"""Rises from `a` to `b`, is 1 until `c` and falls to `d`.

    ```
         ______
    ____/      \____
        ab    cd
    ```
    """
    return Set(curves.trapezoid(a, b, c, d), _name("trapezoid", a, b, c, d))


def rising(a: float, b: float) -> Set:
    """Is 0 up to `a`, rises, is 1 from `b` on.

    ```
         ______
    ____/
        ab
    ```
    """
    return Set(curves.rising(a, b), _name("rising", a, b))


def falling(a: float, b: float) -> Set:
    r"""Is 1 up to `a`, falls, is 0 from `b` on.

    ```
    ____
        \______
        ab
    ```
    """
    return Set(curves.falling(a, b), _name("falling", a, b))


def rectangle(a: float, b: float) -> Set:
    """Is 1 on `[a, b]` and 0 elsewhere.

    ```
         ____
    ____|    |____
        a    b
    ```
    """
    return Set(curves.rectangle(a, b), _name("rectangle", a, b))


def singleton(a: float) -> Set:
    """Is 1 at `a` and 0 elsewhere.

    ```
    ____|____
        a
    ```
    """
    return Set(curves.singleton(a), _name("singleton", a))


def piecewise(points: Iterable[tuple[float, float]]) -> Set:
    """Straight lines through `(x, degree)` points, flat before and after them.

    ```python
    fz.piecewise([(0, 0), (2, 1), (4, 0.5)])
    ```

    The points go by ascending x; repeating an x makes a jump.
    """
    pairs = [(float(x), float(y)) for x, y in points]
    listed = ", ".join(f"({x:g}, {y:g})" for x, y in pairs)
    return Set(curves.piecewise(pairs), f"piecewise([{listed}])")


# --- smooth curves --------------------------------------------------------------------


def gauss(c: float, sigma: float) -> Set:
    """Gaussian bell around `c`: `exp(-(x - c)**2 / (2 * sigma**2))`."""
    return Set(curves.gauss(c, sigma), _name("gauss", c, sigma))


def gauss2(c1: float, sigma1: float, c2: float, sigma2: float) -> Set:
    """Bell with a flat top: `gauss(c1, sigma1)`, 1 until `c2`, `gauss(c2, sigma2)`."""
    curve = curves.gauss2(c1, sigma1, c2, sigma2)
    return Set(curve, _name("gauss2", c1, sigma1, c2, sigma2))


def bell(c: float, width: float, slope: float) -> Set:
    """Bell with a flat top around `c`: `1 / (1 + |(x - c) / width|**(2 * slope))`."""
    return Set(curves.bell(c, width, slope), _name("bell", c, width, slope))


def sigmoid(c: float, slope: float) -> Set:
    """Smooth step at `c`: `1 / (1 + exp(-slope * (x - c)))`; falls if `slope < 0`."""
    return Set(curves.sigmoid(c, slope), _name("sigmoid", c, slope))


def dsigmoid(c1: float, slope1: float, c2: float, slope2: float) -> Set:
    """Plateau between `c1` and `c2`: `|sigmoid(c1, slope1) - sigmoid(c2, slope2)|`."""
    curve = curves.dsigmoid(c1, slope1, c2, slope2)
    return Set(curve, _name("dsigmoid", c1, slope1, c2, slope2))


def psigmoid(c1: float, slope1: float, c2: float, slope2: float) -> Set:
    """Plateau if the slopes oppose: `sigmoid(c1, slope1) * sigmoid(c2, slope2)`."""
    curve = curves.psigmoid(c1, slope1, c2, slope2)
    return Set(curve, _name("psigmoid", c1, slope1, c2, slope2))


def s_shape(a: float, b: float) -> Set:
    """Smooth rise from 0 at `a` to 1 at `b`."""
    return Set(curves.s_shape(a, b), _name("s_shape", a, b))


def z_shape(a: float, b: float) -> Set:
    """Smooth fall from 1 at `a` to 0 at `b`."""
    return Set(curves.z_shape(a, b), _name("z_shape", a, b))


def pi_shape(a: float, b: float, c: float, d: float) -> Set:
    """Smooth rise from `a` to `b`, 1 until `c`, smooth fall to `d`."""
    return Set(curves.pi_shape(a, b, c, d), _name("pi_shape", a, b, c, d))
