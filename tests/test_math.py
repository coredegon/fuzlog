import numpy as np
import pytest

import fuzlog as fz
from fuzlog.math import curves, defuzz, grid, norms


def test_norms():
    a, b = np.array([0.2, 0.9]), np.array([0.5, 0.6])
    assert norms.probor(a, b) == pytest.approx([0.6, 0.96])
    assert norms.bounded_sum(a, b) == pytest.approx([0.7, 1.0])
    assert norms.bounded_diff(a, b) == pytest.approx([0.0, 0.5])


def test_defuzzifiers():
    x = np.linspace(0, 10, 11)
    hill = fz.trapezoid(2, 4, 6, 9)(x)
    assert defuzz.centroid(x, fz.triangle(2, 4, 9)(x)) == pytest.approx(
        5.0
    )  # (2 + 4 + 9) / 3
    assert fz.bisector(x, hill) == 5.0
    assert (fz.som(x, hill), fz.mom(x, hill), fz.lom(x, hill)) == (4.0, 5.0, 6.0)
    both = np.stack(
        [hill, np.zeros(11)]
    )  # the last axis is the grid; an empty set has no value
    for method in (fz.centroid, fz.bisector, fz.mom, fz.som, fz.lom):
        value, empty = method(x, both)
        assert np.isfinite(value)
        assert np.isnan(empty)


def test_curves_are_plain_functions_of_arrays():
    x = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    assert curves.triangle(0, 5, 10)(x).tolist() == [0, 0.5, 1, 0.5, 0]
    assert curves.gauss(5, 2)(x)[2] == 1.0
    grid_2d = curves.s_shape(0, 10)(x.reshape(1, 5))
    assert grid_2d.shape == (1, 5)
    assert fz.Set(curves.rising(0, 10), "ramp")(2.5) == 0.25
    with pytest.raises(ValueError, match="ascending"):
        curves.trapezoid(0, 3, 2, 4)


def test_grid_lookup_reads_values_between_nodes():
    values = np.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
    at = [grid.locate(np.array([0.0, 0.5, 9.0]), 0.0, 1.0, 2, False), np.array([0.5])]
    assert at[0].tolist() == [0.0, 0.5, 1.0]  # outside the axis: its nearest end
    assert grid.linear(values, at, [False, False]).tolist() == [0.5, 5.5, 10.5]
    assert grid.nearest(values, at, [False, False]).tolist() == [0.0, 0.0, 10.0]
    around = grid.locate(np.array([-90.0, 360.0, 450.0]), 0.0, 360.0, 4, True)
    assert around.tolist() == [3.0, 0.0, 1.0]
