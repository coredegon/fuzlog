"""What is said about variables: propositions, hedges and rules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from numpy.typing import ArrayLike

from fuzlog.sets import MF
from fuzlog.types import Array, Norm

if TYPE_CHECKING:
    from fuzlog.variables import Variable

type Degree = Callable[[Variable, str], Array]  # degree of the inputs in a term

_HEDGES = {2.0: "very", 3.0: "extremely", 0.5: "somewhat"}


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
        if not power > 0.0:
            raise ValueError(
                f"a proposition is raised to a positive power, got {power}"
            )
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

    def __bool__(self) -> bool:
        raise TypeError("a Prop has no truth value: combine with & | ~, not and/or/not")

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


def very[S: (MF, Prop)](s: S) -> S:
    """Concentrate a set or a proposition: degrees are squared."""
    return s**2


def extremely[S: (MF, Prop)](s: S) -> S:
    """Concentrate a set or a proposition strongly: degrees are cubed."""
    return s**3


def somewhat[S: (MF, Prop)](s: S) -> S:
    """Dilate a set or a proposition: the square root of the degrees."""
    return s**0.5
