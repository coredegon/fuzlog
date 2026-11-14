"""The data model: fuzzy sets, linguistic variables, propositions and rules.

Everything that computes with them lives in :mod:`fuzl.core`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes

type Array = NDArray[np.float64]
type Kernel = Callable[[Array], Array]  # values -> degrees in [0, 1], same shape
type Norm = Callable[[Array, Array], Array]  # t-norm or s-norm on broadcastable degrees
type Defuzzifier = Callable[[Array, Array], Array]  # (x, mu[..., len(x)]) -> crisp[...]
type Degree = Callable[[Variable, str], Array]  # degree of the inputs in a term

_HEDGES = {2.0: "very", 3.0: "extremely", 0.5: "somewhat"}


def _ambiguous(self: object) -> bool:
    kind = type(self).__name__
    raise TypeError(f"a {kind} has no truth value: combine with & | ~, not and/or/not")


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
    __bool__ = _ambiguous

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
        return MF(lambda x: self.fn(x) ** power, f"{self}**{power:g}")

    def __repr__(self) -> str:
        return self.label


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
        from fuzl.plotting import plot_variable

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


class Prop:
    """A fuzzy proposition: ``variable is term`` or a combination of propositions.

    Propositions combine with ``&`` (and), ``|`` (or), ``~`` (not) and ``**`` (a
    hedge), and ``>>`` turns one into a rule: ``temp.hot & humidity.high >> fan.fast``.
    Calling a proposition gives the degree to which it holds::

        (temp.hot & ~humidity.high)(temp=31, humidity=0.4)

    ``op`` is one of ``is``, ``and``, ``or``, ``not`` and ``pow``; ``args`` holds the
    variable and term name, the operands, or the operand and its exponent.
    """

    __slots__ = ("args", "op")
    __bool__ = _ambiguous

    def __init__(self, op: str, *args: Any) -> None:
        self.op = op
        self.args = args

    def __and__(self, other: Prop) -> Prop:
        return Prop("and", self, other) if isinstance(other, Prop) else NotImplemented

    def __or__(self, other: Prop) -> Prop:
        return Prop("or", self, other) if isinstance(other, Prop) else NotImplemented

    def __invert__(self) -> Prop:
        return Prop("not", self)

    def __pow__(self, power: float) -> Prop:
        return Prop("pow", self, float(power))

    def __rshift__(self, then: Prop | float | Callable[..., ArrayLike]) -> Rule:
        return Rule(self, then)

    @property
    def variables(self) -> tuple[Variable, ...]:
        """The variables the proposition talks about, in order of first mention."""
        if self.op == "is":
            return (self.args[0],)
        nested = (
            v for part in self.args if isinstance(part, Prop) for v in part.variables
        )
        return tuple(dict.fromkeys(nested))

    def truth(
        self, degree: Degree, tnorm: Norm = np.minimum, snorm: Norm = np.maximum
    ) -> Array:
        """Evaluate the proposition, given the degree of every ``variable is term``.

        ``and`` is computed by ``tnorm`` and ``or`` by ``snorm``.
        """
        if self.op == "is":
            return degree(*self.args)
        first, *rest = self.args
        value = first.truth(degree, tnorm, snorm)
        match self.op:
            case "and":
                return tnorm(value, rest[0].truth(degree, tnorm, snorm))
            case "or":
                return snorm(value, rest[0].truth(degree, tnorm, snorm))
            case "not":
                return 1.0 - value
            case _:
                return value ** rest[0]

    @overload
    def __call__(self, **inputs: float) -> float: ...
    @overload
    def __call__(self, **inputs: ArrayLike) -> Array: ...
    def __call__(self, **inputs: ArrayLike) -> Array | float:
        """Return the degree to which the proposition holds for the given inputs.

        Inputs are passed by variable name and broadcast against each other; ``and``
        is the minimum and ``or`` the maximum.
        """
        degree, shape = fuzzifier(inputs)
        truth = self.truth(degree)
        return np.broadcast_to(truth, shape).copy() if shape else truth.item()

    def __repr__(self) -> str:
        first, *rest = self.args
        match self.op:
            case "is":
                return f"{first.name} is {rest[0]}"
            case "not":
                return f"not ({first})"
            case "pow":
                hedge = _HEDGES.get(rest[0])
                return f"{hedge} ({first})" if hedge else f"({first})**{rest[0]:g}"
            case op:  # "a or b" is bracketed inside an "and", as it binds looser
                bracket = op == "and"
                parts = (
                    f"({p})" if bracket and p.op == "or" else f"{p}" for p in self.args
                )
                return f" {op} ".join(parts)


class Rule:
    """A fuzzy rule: if ``when`` holds, then ``then``.

    ``then`` is a proposition about an output variable (Mamdani), or a number or a
    function of the inputs (Sugeno). ``weight`` in [0, 1] scales the firing strength.
    Rules are usually written ``when >> then``; because Python binds ``>>`` tighter
    than ``&`` and ``|``, a rule also absorbs conditions joined to its left, so the
    antecedent needs no parentheses: ``a & b >> c`` is ``(a & b) >> c``.
    """

    __slots__ = ("then", "weight", "when")

    def __init__(
        self,
        when: Prop,
        then: Prop | float | Callable[..., ArrayLike],
        weight: float = 1.0,
    ) -> None:
        if not isinstance(when, Prop):
            raise TypeError(f"the condition of a rule must be a Prop, got {when!r}")
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"a rule weight must lie in [0, 1], got {weight}")
        self.when = when
        self.then = then
        self.weight = weight

    def __rand__(self, other: Prop) -> Rule:
        return Rule(other & self.when, self.then, self.weight)

    def __ror__(self, other: Prop) -> Rule:
        return Rule(other | self.when, self.then, self.weight)

    def __repr__(self) -> str:
        then = getattr(self.then, "__name__", self.then)
        weight = "" if self.weight == 1.0 else f" [{self.weight:g}]"
        return f"if {self.when} then {then}{weight}"


def fuzzifier(inputs: Mapping[str, ArrayLike]) -> tuple[Degree, tuple[int, ...]]:
    """Prepare crisp inputs, given by variable name, for evaluating propositions.

    Returns ``degree(variable, term)``, which fuzzifies on demand and remembers the
    result, and the shape the inputs broadcast to (``()`` if all are scalars).
    """
    values = {name: np.asarray(x, dtype=np.float64) for name, x in inputs.items()}
    shapes = {x.shape for x in values.values()}
    shape = shapes.pop() if len(shapes) == 1 else np.broadcast_shapes(*shapes)
    wrapped: dict[Variable, Array] = {}
    degrees: dict[tuple[Variable, str], Array] = {}

    def degree(variable: Variable, term: str) -> Array:
        key = (variable, term)
        if key not in degrees:
            if variable not in wrapped:
                if variable.name not in values:
                    raise TypeError(f"missing input {variable.name!r}")
                wrapped[variable] = variable._wrap(np.atleast_1d(values[variable.name]))
            degrees[key] = variable.terms[term].fn(wrapped[variable])
        return degrees[key]

    return degree, shape
