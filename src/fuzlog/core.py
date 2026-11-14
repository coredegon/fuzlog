"""Everything that computes: membership functions, partitions, norms and inference.

The module reads top to bottom: shapes of fuzzy sets, partitions of a universe,
hedges, norms, defuzzifiers and, built on all of them, the two inference systems.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping
from functools import lru_cache, reduce
from numbers import Real
from typing import TYPE_CHECKING, overload

import numpy as np
from numpy.typing import ArrayLike

from fuzlog.types import (
    MF,
    Array,
    Defuzzifier,
    Kernel,
    Norm,
    Prop,
    Rule,
    Variable,
    fuzzifier,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

type Core = float | tuple[float, float]

_SMALL = 2048  # up to this many values np.interp wins: one call instead of several
_BLOCK = 1 << 14  # larger inputs are processed in blocks that stay in the CPU cache
_CELLS = 1 << 15  # size of the blocks in which Mamdani output sets are built
_TINY = float(np.finfo(np.float64).tiny)


# --- membership functions: piecewise-linear shapes ------------------------------------


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


# --- membership functions: smooth shapes ----------------------------------------------


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


# --- partitions -----------------------------------------------------------------------


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
        kernel = _pwl([(low, seam), *bump, (high, seam)])
        triangular = _label("triangle", a, b, d)
        sets.append(
            MF(kernel, triangular if b == c else _label("trapezoid", a, b, c, d))
        )
    return sets


# --- hedges ---------------------------------------------------------------------------


def very[S: (MF, Prop)](s: S) -> S:
    """Concentrate a set or a proposition: degrees are squared."""
    return s**2


def extremely[S: (MF, Prop)](s: S) -> S:
    """Concentrate a set or a proposition strongly: degrees are cubed."""
    return s**3


def somewhat[S: (MF, Prop)](s: S) -> S:
    """Dilate a set or a proposition: the square root of the degrees."""
    return s**0.5


# --- norms: numpy's minimum, multiply and maximum are norms too -----------------------


def probor(a: Array, b: Array) -> Array:
    """Probabilistic sum ``a + b - a * b``: the s-norm that goes with the product."""
    return a + b - a * b


def bounded_sum(a: Array, b: Array) -> Array:
    """Bounded sum ``min(a + b, 1)``: the Lukasiewicz s-norm."""
    return np.minimum(a + b, 1.0)


def bounded_diff(a: Array, b: Array) -> Array:
    """Bounded difference ``max(a + b - 1, 0)``: the Lukasiewicz t-norm."""
    return np.maximum(a + b - 1.0, 0.0)


# --- defuzzification: (x, mu) -> crisp, along the last axis of mu ---------------------


@lru_cache(maxsize=32)
def _outline(grid: bytes) -> tuple[Array, Array]:
    """Weights that integrate a set, drawn as straight lines between its samples.

    ``mu @ area`` is the area under the outline and ``mu @ moment`` its first moment.
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
    """Centre of gravity of the area under ``mu``; NaN for an empty set.

    The set is taken as straight lines between its samples, so the result is exact
    when the grid holds every corner of the set.
    """
    area, moment = _outline(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    with np.errstate(invalid="ignore"):
        return (mu @ moment) / (mu @ area)


def bisector(x: Array, mu: Array) -> Array:
    """The value that splits the area under ``mu`` in two halves; NaN if empty."""
    area = np.cumsum(mu, axis=-1)
    total = area[..., -1:]
    first = (area < total / 2.0).sum(axis=-1)  # where the running area reaches half
    return np.where(total[..., 0] > 0.0, x[np.minimum(first, x.size - 1)], np.nan)


def _top(mu: Array) -> np.ndarray:
    """Mark where ``mu`` is highest; nowhere for an empty set."""
    peak = mu.max(axis=-1, keepdims=True)
    return (mu == peak) & (peak > 0.0)


def mom(x: Array, mu: Array) -> Array:
    """Mean of the values at which ``mu`` is highest; NaN for an empty set."""
    top = _top(mu)
    with np.errstate(invalid="ignore"):
        return (top @ x) / top.sum(axis=-1)


def som(x: Array, mu: Array) -> Array:
    """Smallest value at which ``mu`` is highest; NaN for an empty set."""
    top = _top(mu)
    return np.where(top.any(axis=-1), x[top.argmax(axis=-1)], np.nan)


def lom(x: Array, mu: Array) -> Array:
    """Largest value at which ``mu`` is highest; NaN for an empty set."""
    top = _top(mu)
    last = x.size - 1 - top[..., ::-1].argmax(axis=-1)
    return np.where(top.any(axis=-1), x[last], np.nan)


# --- inference ------------------------------------------------------------------------


def _strengths(
    rules: tuple[Rule, ...], tnorm: Norm, snorm: Norm, inputs: Mapping[str, ArrayLike]
) -> tuple[list[Array], tuple[int, ...]]:
    """Firing strength of every rule, and the shape the inputs broadcast to."""
    degree, shape = fuzzifier(inputs)
    fired = [rule.when.truth(degree, tnorm, snorm) for rule in rules]
    for i, rule in enumerate(rules):
        if rule.weight != 1.0:
            fired[i] = fired[i] * rule.weight
    return fired, shape


def _inputs(rules: tuple[Rule, ...]) -> dict[str, Variable]:
    return {v.name: v for rule in rules for v in rule.when.variables}


class Mamdani:
    """Mamdani inference: fuzzy rules in, a crisp value out.

    Every rule concludes ``output is term`` about one and the same variable::

        fan = Mamdani(temp.cold >> speed.slow, temp.hot & humid.wet >> speed.fast)
        fan(temp=31, humid=0.8)

    Inputs are passed by variable name, as scalars or as arrays that broadcast
    against each other. The result is NaN where no rule fires.

    Args:
        *rules: The rule base.
        tnorm: How ``&`` combines degrees.
        snorm: How ``|`` combines degrees.
        implication: How a firing strength shapes the set a rule concludes; a t-norm.
        aggregation: How the shaped sets of all rules merge into the output set.
        defuzz: How the output set becomes a number.
        points: Size of the grid the output universe is sampled on.
    """

    def __init__(
        self,
        *rules: Rule,
        tnorm: Norm = np.minimum,
        snorm: Norm = np.maximum,
        implication: Norm = np.minimum,
        aggregation: Norm = np.maximum,
        defuzz: Defuzzifier = centroid,
        points: int = 101,
    ) -> None:
        conclusions = [rule.then for rule in rules if isinstance(rule.then, Prop)]
        outputs = {v for then in conclusions for v in then.variables}
        if not rules or len(conclusions) < len(rules) or len(outputs) != 1:
            raise ValueError("Mamdani rules must all conclude about one variable")
        (output,) = outputs
        if output.periodic:
            raise ValueError(f"cannot defuzzify the periodic output {output.name!r}")
        if points < 2:
            raise ValueError(f"the output grid needs at least 2 points, got {points}")
        self.rules = rules
        self.inputs = _inputs(rules)
        self.output = output
        self.tnorm, self.snorm = tnorm, snorm
        self.implication, self.aggregation = implication, aggregation
        self.defuzz = defuzz
        self.x = output.grid(points)

        def on_grid(variable: Variable, term: str) -> Array:
            return variable.terms[term].fn(self.x)

        # Under max aggregation, rules that conclude the same share one shaped set.
        shared = aggregation is np.maximum
        keys = [f"{then}" if shared else i for i, then in enumerate(conclusions)]
        groups = {key: [i for i, k in enumerate(keys) if k == key] for key in keys}
        self._groups = list(groups.values())
        firsts = (conclusions[group[0]] for group in self._groups)
        self._sets = np.stack([then.truth(on_grid, tnorm, snorm) for then in firsts])
        self._chunk = max(1, _CELLS // self._sets.size)

    @overload
    def __call__(self, **inputs: float) -> float: ...
    @overload
    def __call__(self, **inputs: ArrayLike) -> Array: ...
    def __call__(self, **inputs: ArrayLike) -> Array | float:
        """Infer the crisp output for the given inputs."""
        strengths, shape = self._fire(inputs)
        out = np.empty(strengths.shape[1])
        for i in range(0, out.size, self._chunk):
            block = slice(i, i + self._chunk)
            out[block] = self.defuzz(self.x, self._shape(strengths[:, block]))
        return out.reshape(shape) if shape else out.item()

    def aggregate(self, **inputs: ArrayLike) -> Array:
        """Return the output fuzzy set before defuzzification, sampled on ``self.x``.

        The result has the broadcast shape of the inputs plus a last axis over the
        grid.
        """
        strengths, shape = self._fire(inputs)
        return self._shape(strengths).reshape(*shape, self.x.size)

    def plot(self, ax: Axes | None = None, /, **inputs: float) -> Axes:
        """Draw the output set inferred for scalar inputs and its crisp value."""
        from fuzlog.plotting import plot_inference

        return plot_inference(self, ax, **inputs)

    def surface(
        self,
        x: str,
        y: str,
        /,
        ax: Axes | None = None,
        *,
        points: int = 64,
        **fixed: float,
    ) -> Axes:
        """Draw the output over the inputs ``x`` and ``y``; the others are ``fixed``."""
        from fuzlog.plotting import plot_surface

        return plot_surface(self, x, y, ax, points=points, **fixed)

    def _fire(self, inputs: Mapping[str, ArrayLike]) -> tuple[Array, tuple[int, ...]]:
        """Firing strength of every group of rules: shape ``(groups, samples)``."""
        fired, shape = _strengths(self.rules, self.tnorm, self.snorm, inputs)
        strengths = np.empty((len(self._groups), *(shape or (1,))))
        for i, group in enumerate(self._groups):
            strengths[i] = reduce(np.maximum, (fired[j] for j in group))
        return strengths.reshape(len(self._groups), -1), shape

    def _shape(self, strengths: Array) -> Array:
        """Output set of every sample on the grid: shape ``(samples, points)``."""
        shaped = self.implication(strengths[:, :, None], self._sets[:, None, :])
        return reduce(self.aggregation, shaped)

    def __repr__(self) -> str:
        inputs = ", ".join(self.inputs)
        return f"Mamdani({len(self.rules)} rules: {inputs} -> {self.output.name})"


class Sugeno:
    """Takagi-Sugeno inference: rules conclude a number or a function of the inputs.

    The output is the average of the conclusions, weighted by firing strength::

        fan = Sugeno(temp.cold >> 10.0, temp.hot >> (lambda temp: 2.5 * temp))
        fan(temp=31)

    A function is called with all inputs as keyword arguments. Inputs are passed by
    variable name and broadcast against each other; the result is NaN where no rule
    fires.

    Args:
        *rules: The rule base.
        tnorm: How ``&`` combines degrees.
        snorm: How ``|`` combines degrees.
    """

    def __init__(
        self, *rules: Rule, tnorm: Norm = np.minimum, snorm: Norm = np.maximum
    ) -> None:
        for rule in rules:
            then = rule.then
            if isinstance(then, Prop) or not (callable(then) or isinstance(then, Real)):
                raise TypeError(f"Sugeno rules conclude a number or a function: {rule}")
        if not rules:
            raise ValueError("a Sugeno system needs at least one rule")
        self.rules = rules
        self.inputs = _inputs(rules)
        self.tnorm, self.snorm = tnorm, snorm

    @overload
    def __call__(self, **inputs: float) -> float: ...
    @overload
    def __call__(self, **inputs: ArrayLike) -> Array: ...
    def __call__(self, **inputs: ArrayLike) -> Array | float:
        """Infer the crisp output for the given inputs."""
        fired, shape = _strengths(self.rules, self.tnorm, self.snorm, inputs)
        values = {name: np.asarray(x, dtype=np.float64) for name, x in inputs.items()}
        weighted, total = np.zeros(1), np.zeros(1)
        for strength, rule in zip(fired, self.rules, strict=True):
            then = rule.then
            value = np.asarray(then(**values)) if callable(then) else then
            weighted = weighted + strength * value
            total = total + strength
        with np.errstate(invalid="ignore"):
            out = weighted / total
        return np.broadcast_to(out, shape).copy() if shape else out.item()

    def surface(
        self,
        x: str,
        y: str,
        /,
        ax: Axes | None = None,
        *,
        points: int = 64,
        **fixed: float,
    ) -> Axes:
        """Draw the output over the inputs ``x`` and ``y``; the others are ``fixed``."""
        from fuzlog.plotting import plot_surface

        return plot_surface(self, x, y, ax, points=points, **fixed)

    def __repr__(self) -> str:
        return f"Sugeno({len(self.rules)} rules: {', '.join(self.inputs)})"
