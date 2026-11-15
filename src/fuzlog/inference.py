"""Inference: norms, defuzzifiers and the systems that answer for given inputs.

``Mamdani`` and ``Sugeno`` infer a number from rules; ``Labels`` describes the
inputs in words.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from functools import reduce
from numbers import Real
from typing import TYPE_CHECKING, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fuzlog.math.defuzz import centroid
from fuzlog.rules import Prop, Rule, fuzzifier
from fuzlog.types import Array, Defuzzifier, Norm
from fuzlog.variables import Variable

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_CELLS = 1 << 15  # size of the blocks in which Mamdani output sets are built


# --- systems --------------------------------------------------------------------------


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
            if callable(then):  # counts only where its rule fires
                value = np.where(strength > 0.0, np.asarray(then(**values)), 0.0)
                weighted = weighted + strength * value
            else:
                weighted = weighted + strength * then
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


class Labels:
    """Describe inputs in words: the strongest term of every variable.

    The terms are joined by ``sep`` in the order of the variables::

        describe = Labels(temp, humid, sep=" ")
        describe(temp=31, humid=80)  # "hot wet"

    Inputs are passed by variable name, as scalars or as arrays that broadcast
    against each other; a scalar gives a string and arrays give an array of
    strings. Of terms that hold equally the first one wins. The label is empty
    where an input is NaN or belongs to no term of its variable.

    Args:
        *variables: The variables to describe.
        sep: What joins the terms.
    """

    def __init__(self, *variables: Variable, sep: str = "_") -> None:
        strangers = [v for v in variables if not isinstance(v, Variable)]
        if not variables or strangers:
            raise TypeError(f"Labels describe variables, got {strangers or variables}")
        if not isinstance(sep, str):
            raise TypeError(f"sep must be a string, got {sep!r}")
        self.inputs = {variable.name: variable for variable in variables}
        if len(self.inputs) < len(variables):
            raise ValueError(f"the variables must differ, got {list(variables)}")
        self.sep = sep
        terms = itertools.product(*(variable.terms for variable in variables))
        # The empty label closes the list: it is what the number -1 counts to.
        self._words = np.array([*(sep.join(names) for names in terms), ""])

    @overload
    def __call__(self, **inputs: float) -> str: ...
    @overload
    def __call__(self, **inputs: ArrayLike) -> NDArray[np.str_]: ...
    def __call__(self, **inputs: ArrayLike) -> NDArray[np.str_] | str:
        """Describe the given inputs."""
        missing = [name for name in self.inputs if name not in inputs]
        if missing:
            raise TypeError(f"missing input {missing[0]!r}")
        values = {name: np.asarray(inputs[name], np.float64) for name in self.inputs}
        shape = np.broadcast_shapes(*(x.shape for x in values.values()))
        number: NDArray[np.intp] = np.zeros(1, dtype=np.intp)
        blank: NDArray[np.bool_] = np.zeros(1, dtype=np.bool_)
        for name, variable in self.inputs.items():
            degrees = variable(np.atleast_1d(values[name]))
            number = number * len(variable.terms) + degrees.argmax(axis=0)
            blank = blank | ~(degrees.max(axis=0) > 0.0)
        words = self._words[np.where(blank, -1, number)]
        return np.broadcast_to(words, shape).copy() if shape else str(words.item())

    def __repr__(self) -> str:
        return f"Labels({', '.join(self.inputs)})"
