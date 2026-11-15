"""Inference systems: rules in, a crisp value out."""

from __future__ import annotations

import inspect
from functools import reduce
from numbers import Real
from typing import overload

import numpy as np
from numpy.typing import ArrayLike

from fuzlog.math.defuzz import centroid
from fuzlog.rules import Inputs, Prop, Rule
from fuzlog.types import Array, Defuzzifier, Norm
from fuzlog.variables import Variable

_CELLS = 1 << 15  # size of the blocks in which Mamdani output sets are built


def _inputs(rules: tuple[Rule, ...]) -> dict[str, Variable]:
    return {name: v for rule in rules for name, v in rule.when.inputs.items()}


class Mamdani:
    """Mamdani inference: every rule concludes `output is term`.

    ```python
    fan = fz.Mamdani(temp.cold >> speed.slow, temp.hot | humid.wet >> speed.fast)
    fan(temp=31, humid=80)              # 74.3
    fan(temp=[18, 25, 31], humid=80)    # an array: inputs broadcast
    ```

    The result is NaN where no rule fires.

    Args:
        *rules: The rule base.
        tnorm: How `&` combines degrees.
        snorm: How `|` combines degrees.
        implication: How a firing strength shapes the set a rule concludes.
        aggregation: How the shaped sets of all rules merge into the output set.
        defuzz: How the output set becomes a number.
        points: Size of the grid `x` the output universe is sampled on.
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
        thens = [rule.then for rule in rules if isinstance(rule.then, Prop)]
        outputs = {v for then in thens for v in then.inputs.values()}
        if not rules or len(thens) < len(rules) or len(outputs) != 1:
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
        # Under max aggregation, rules that conclude the same share one shaped set.
        shared = aggregation is np.maximum
        groups: dict[object, list[int]] = {}
        for i, then in enumerate(thens):
            groups.setdefault(f"{then}" if shared else i, []).append(i)
        self._groups = list(groups.values())
        on_grid = Inputs({output.name: self.x})
        firsts = (thens[group[0]] for group in self._groups)
        self._sets = np.stack([then.truth(on_grid, tnorm, snorm) for then in firsts])
        self._block = max(1, _CELLS // self._sets.size)

    @overload
    def __call__(self, **inputs: float) -> float: ...
    @overload
    def __call__(self, **inputs: ArrayLike) -> Array: ...
    def __call__(self, **inputs: ArrayLike) -> Array | float:
        """Infer the crisp output for inputs given by variable name."""
        given = Inputs(inputs)
        fired = self._fire(given)
        out = np.empty(fired.shape[1])
        for i in range(0, out.size, self._block):
            block = slice(i, i + self._block)
            out[block] = self.defuzz(self.x, self._merge(fired[:, block]))
        return out.reshape(given.shape) if given.shape else out.item()

    def aggregate(self, **inputs: ArrayLike) -> Array:
        """Return the output set before defuzzification, sampled on `x`.

        Its shape is the broadcast shape of the inputs plus a last axis over `x`.
        """
        given = Inputs(inputs)
        return self._merge(self._fire(given)).reshape(*given.shape, self.x.size)

    def _fire(self, inputs: Inputs) -> Array:
        """Firing strength of every group of rules: shape `(groups, samples)`."""
        fired = [rule.fire(inputs, self.tnorm, self.snorm) for rule in self.rules]
        strengths = np.empty((len(self._groups), *(inputs.shape or (1,))))
        for i, group in enumerate(self._groups):
            strengths[i] = reduce(np.maximum, (fired[j] for j in group))
        return strengths.reshape(len(self._groups), -1)

    def _merge(self, strengths: Array) -> Array:
        """Output set of every sample on the grid: shape `(samples, points)`."""
        shaped = self.implication(strengths[:, :, None], self._sets[:, None, :])
        return reduce(self.aggregation, shaped)

    def __repr__(self) -> str:
        inputs = ", ".join(self.inputs)
        return f"Mamdani({len(self.rules)} rules: {inputs} -> {self.output.name})"


class Sugeno:
    """Takagi-Sugeno inference: every rule concludes a number or a function.

    ```python
    fan = fz.Sugeno(temp.cold >> 10, temp.hot >> (lambda temp: 2.5 * temp))
    fan(temp=31)    # the conclusions, averaged by how strongly their rules fire
    ```

    A function receives the inputs it names. The result is NaN where no rule fires.

    Args:
        *rules: The rule base.
        tnorm: How `&` combines degrees.
        snorm: How `|` combines degrees.
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
        self._asks = [_parameters(rule.then) for rule in rules]

    @overload
    def __call__(self, **inputs: float) -> float: ...
    @overload
    def __call__(self, **inputs: ArrayLike) -> Array: ...
    def __call__(self, **inputs: ArrayLike) -> Array | float:
        """Infer the crisp output for inputs given by variable name."""
        given = Inputs(inputs)
        weighted, total = np.zeros(1), np.zeros(1)
        for rule, asks in zip(self.rules, self._asks, strict=True):
            strength, then = rule.fire(given, self.tnorm, self.snorm), rule.then
            if callable(then):  # counts only where its rule fires
                names = given.values if asks is None else asks
                value = np.asarray(then(**{name: given[name] for name in names}))
                weighted = weighted + strength * np.where(strength > 0.0, value, 0.0)
            else:
                weighted = weighted + strength * then
            total = total + strength
        with np.errstate(invalid="ignore"):
            return given.result(weighted / total)

    def __repr__(self) -> str:
        return f"Sugeno({len(self.rules)} rules: {', '.join(self.inputs)})"


def _parameters(then: object) -> tuple[str, ...] | None:
    """Names of the inputs a conclusion asks for; None if it takes them all."""
    if not callable(then):
        return ()
    parameters = inspect.signature(then).parameters.values()
    if any(p.kind is p.VAR_KEYWORD for p in parameters):
        return None
    return tuple(p.name for p in parameters)
