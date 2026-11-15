"""Fuzzy logic for Python: small, fast, typed.

```python
import fuzlog as fz

temp = fz.partition("temp", (0, 40), cold=10, warm=20, hot=30)
fan = fz.partition("fan", (0, 100), ["slow", "medium", "fast"])
cooler = fz.Mamdani(temp.cold >> fan.slow, temp.hot >> fan.fast)
cooler(temp=27)
```

- sets: `triangle`, `gauss` and 14 more shapes, combined with `& | ~ **`;
- variables: `Variable` and `partition`;
- rules: `temp.hot & humid.wet >> fan.fast`;
- systems: `Mamdani` and `Sugeno`;
- features: `Labels`, `tabulate` and `Table`, `plot` and `surface`;
- `fuzlog.math`: all of the mathematics as plain NumPy functions.
"""

from fuzlog import features, math
from fuzlog.features.labels import Labels
from fuzlog.features.plots import plot, surface
from fuzlog.features.tables import Table, tabulate
from fuzlog.math.defuzz import bisector, centroid, lom, mom, som
from fuzlog.math.norms import bounded_diff, bounded_sum, probor
from fuzlog.rules import Prop, Rule, extremely, somewhat, very
from fuzlog.sets import (
    Set,
    bell,
    dsigmoid,
    falling,
    gauss,
    gauss2,
    pi_shape,
    piecewise,
    psigmoid,
    rectangle,
    rising,
    s_shape,
    sigmoid,
    singleton,
    trapezoid,
    triangle,
    z_shape,
)
from fuzlog.systems import Mamdani, Sugeno
from fuzlog.variables import Variable, partition

__all__ = [
    "Labels",
    "Mamdani",
    "Prop",
    "Rule",
    "Set",
    "Sugeno",
    "Table",
    "Variable",
    "bell",
    "bisector",
    "bounded_diff",
    "bounded_sum",
    "centroid",
    "dsigmoid",
    "extremely",
    "falling",
    "features",
    "gauss",
    "gauss2",
    "lom",
    "math",
    "mom",
    "partition",
    "pi_shape",
    "piecewise",
    "plot",
    "probor",
    "psigmoid",
    "rectangle",
    "rising",
    "s_shape",
    "sigmoid",
    "singleton",
    "som",
    "somewhat",
    "surface",
    "tabulate",
    "trapezoid",
    "triangle",
    "very",
    "z_shape",
]
