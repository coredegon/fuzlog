"""What is said about variables: propositions, rules and the hedges."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from numpy.typing import ArrayLike

from fuzlog.types import Array, Norm

if TYPE_CHECKING:
    from fuzlog.sets import Set
    from fuzlog.variables import Variable

_HEDGES = {2.0: "very", 3.0: "extremely", 0.5: "somewhat"}


class Inputs:
    """Crisp inputs by variable name, fuzzified term by term and only once.

    `shape` is the shape the inputs broadcast to: `()` if all are scalars.
    """

    __slots__ = ("_seen", "shape", "values")

    def __init__(self, values: Mapping[str, ArrayLike]) -> None:
        given = {name: np.asarray(x, dtype=np.float64) for name, x in values.items()}
        shapes = {x.shape for x in given.values()}
        self.shape = shapes.pop() if len(shapes) == 1 else np.broadcast_shapes(*shapes)
        self.values = {name: x if x.ndim else x[None] for name, x in given.items()}
        self._seen: dict[Any, Array] = {}

    def __getitem__(self, name: str) -> Array:
        if name not in self.values:
            raise TypeError(f"missing input {name!r}")
        return self.values[name]

    def degree(self, variable: Variable, term: str) -> Array:
        """Return the degree of the input of `variable` in its `term`."""
        seen = self._seen
        if (variable, term) not in seen:
            if variable not in seen:
                seen[variable] = variable.wrap(self[variable.name])
            seen[variable, term] = variable.terms[term].curve(seen[variable])
        return seen[variable, term]

    def result(self, out: Array) -> Any:
        """Give `out` the shape of the inputs: a float if they are scalars."""
        return np.broadcast_to(out, self.shape).copy() if self.shape else out.item()


class Prop:
    """A proposition: `variable is term`, or propositions joined together.

    ```python
    muggy = temp.hot & ~humid.dry     # & and, | or, ~ not, ** hedge
    muggy(temp=31, humid=80)          # 0.8: the degree to which it holds
    muggy >> fan.fast                 # a rule
    ```

    `op` is `is`, `and`, `or`, `not` or `pow`; `args` holds what it applies to.
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
    def inputs(self) -> dict[str, Variable]:
        """The variables the proposition talks about, by name."""
        if self.op == "is":
            return {self.args[0].name: self.args[0]}
        parts = (part for part in self.args if isinstance(part, Prop))
        return {name: v for part in parts for name, v in part.inputs.items()}

    def truth(
        self, inputs: Inputs, tnorm: Norm = np.minimum, snorm: Norm = np.maximum
    ) -> Array:
        """Return the degree to which it holds: `and` by `tnorm`, `or` by `snorm`."""
        if self.op == "is":
            return inputs.degree(*self.args)
        first, *rest = self.args
        value = first.truth(inputs, tnorm, snorm)
        match self.op:
            case "and":
                return tnorm(value, rest[0].truth(inputs, tnorm, snorm))
            case "or":
                return snorm(value, rest[0].truth(inputs, tnorm, snorm))
            case "not":
                return 1.0 - value
            case _:
                return value ** rest[0]

    @overload
    def __call__(self, **inputs: float) -> float: ...
    @overload
    def __call__(self, **inputs: ArrayLike) -> Array: ...
    def __call__(self, **inputs: ArrayLike) -> Array | float:
        """Return the degree to which the proposition holds for inputs given by name."""
        given = Inputs(inputs)
        return given.result(self.truth(given))

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
    """A rule: if `when` holds, then `then`.

    ```python
    temp.hot & humid.wet >> fan.fast          # for Mamdani
    temp.hot >> (lambda temp: 2 * temp)       # for Sugeno: a number or a function
    fz.Rule(temp.hot, fan.fast, weight=0.5)   # fires at half strength
    ```

    `>>` takes everything to its left as the condition, so it needs no brackets.
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

    # Python binds >> tighter than & and |: the rule takes in what was left outside.
    def __rand__(self, other: Prop) -> Rule:
        return Rule(other & self.when, self.then, self.weight)

    def __ror__(self, other: Prop) -> Rule:
        return Rule(other | self.when, self.then, self.weight)

    def fire(
        self, inputs: Inputs, tnorm: Norm = np.minimum, snorm: Norm = np.maximum
    ) -> Array:
        """Return how strongly the rule fires: the truth of `when` times `weight`."""
        truth = self.when.truth(inputs, tnorm, snorm)
        return truth if self.weight == 1.0 else truth * self.weight

    def __repr__(self) -> str:
        then = getattr(self.then, "__name__", self.then)
        weight = "" if self.weight == 1.0 else f" [{self.weight:g}]"
        return f"if {self.when} then {then}{weight}"


def very[S: (Set, Prop)](s: S) -> S:
    """Concentrate a set or a proposition: `s ** 2`."""
    return s**2


def extremely[S: (Set, Prop)](s: S) -> S:
    """Concentrate a set or a proposition strongly: `s ** 3`."""
    return s**3


def somewhat[S: (Set, Prop)](s: S) -> S:
    """Dilate a set or a proposition: `s ** 0.5`."""
    return s**0.5
