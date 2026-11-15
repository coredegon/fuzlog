"""Fuzzy logic for Python: small, fast, typed.

``import fuzlog as fz`` gives the whole library:

- sets: ``triangle``, ``trapezoid``, ``gauss`` and 13 more shapes, combined with
  ``& | ~ **``;
- variables: ``Variable`` and ``partition``;
- rules: ``temp.hot & humidity.high >> fan.fast``;
- inference: ``Mamdani`` and ``Sugeno``;
- words: ``Labels`` names the strongest terms of the inputs;
- plots: ``variable.plot()``, ``system.plot(...)``, ``system.surface(...)``;
- tables: ``compile(system)`` for lookup speed, ``table.save(path)`` and ``load``.
"""

from fuzlog.inference import (
    Labels,
    Mamdani,
    Sugeno,
    bisector,
    bounded_diff,
    bounded_sum,
    centroid,
    lom,
    mom,
    probor,
    som,
)
from fuzlog.rules import Prop, Rule, extremely, somewhat, very
from fuzlog.sets import (
    MF,
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
from fuzlog.table import Table, compile, load
from fuzlog.types import Array, Defuzzifier, Kernel, Norm
from fuzlog.variables import Variable, partition

__all__ = [
    "MF",
    "Array",
    "Defuzzifier",
    "Kernel",
    "Labels",
    "Mamdani",
    "Norm",
    "Prop",
    "Rule",
    "Sugeno",
    "Table",
    "Variable",
    "bell",
    "bisector",
    "bounded_diff",
    "bounded_sum",
    "centroid",
    "compile",
    "dsigmoid",
    "extremely",
    "falling",
    "gauss",
    "gauss2",
    "load",
    "lom",
    "mom",
    "partition",
    "pi_shape",
    "piecewise",
    "probor",
    "psigmoid",
    "rectangle",
    "rising",
    "s_shape",
    "sigmoid",
    "singleton",
    "som",
    "somewhat",
    "trapezoid",
    "triangle",
    "very",
    "z_shape",
]
