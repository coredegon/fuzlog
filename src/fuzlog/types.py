"""The types the library is annotated with."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

type Array = NDArray[np.float64]
type Kernel = Callable[[Array], Array]  # values -> degrees in [0, 1], same shape
type Norm = Callable[[Array, Array], Array]  # t-norm or s-norm on broadcastable degrees
type Defuzzifier = Callable[[Array, Array], Array]  # (x, mu[..., len(x)]) -> crisp[...]
