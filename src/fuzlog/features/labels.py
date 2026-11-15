"""Labels: inputs described in words."""

from __future__ import annotations

import itertools
from typing import overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fuzlog.variables import Variable


class Labels:
    """Describe inputs in words: the strongest term of every variable.

    ```python
    words = fz.Labels(temp, humid, sep=" ")
    words(temp=31, humid=80)            # "hot wet"
    words(temp=[5, 31], humid=80)       # array(["cold wet", "hot wet"])
    ```

    Of equal terms the first wins. The label is empty where an input is NaN or
    belongs to no term of its variable.
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
        """Describe inputs given by variable name."""
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
