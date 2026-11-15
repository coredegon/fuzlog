import numpy as np
import pytest

import fuzlog as fz

# Every shape with values that follow from its definition: (set, x, degrees).
SHAPES = [
    (
        fz.triangle(0, 49, 98),
        [-1, 0, 24.5, 49, 73.5, 98, 99],
        [0, 0, 0.5, 1, 0.5, 0, 0],
    ),
    (fz.trapezoid(0, 2, 5, 9), [0, 1, 2, 5, 7, 9], [0, 0.5, 1, 1, 0.5, 0]),
    (fz.trapezoid(0, 0, 2, 4), [-1e-300, 0, 2, 3, 4], [0, 1, 1, 0.5, 0]),
    (fz.rising(2, 4), [1, 2, 3, 4, 5], [0, 0, 0.5, 1, 1]),
    (fz.falling(2, 4), [1, 2, 3, 4, 5], [1, 1, 0.5, 0, 0]),
    (fz.rectangle(1, 3), [0.999, 1, 2, 3, 3.001], [0, 1, 1, 1, 0]),
    (fz.singleton(2), [1.999, 2, 2.001], [0, 1, 0]),
    (
        fz.piecewise([(0, 0), (2, 1), (4, 0.5)]),
        [-1, 1, 2, 3, 9],
        [0, 0.5, 1, 0.75, 0.5],
    ),
    (fz.gauss(5, 2), [3, 5, 7], [np.exp(-0.5), 1, np.exp(-0.5)]),
    (fz.gauss2(3, 1, 6, 2), [2, 3, 4.5, 6, 8], [np.exp(-0.5), 1, 1, 1, np.exp(-0.5)]),
    (fz.bell(5, 2, 3), [3, 5, 7], [0.5, 1, 0.5]),
    (fz.sigmoid(5, 2), [-1e4, 5, 1e4], [0, 0.5, 1]),
    (fz.dsigmoid(2, 50, 7, 50), [0, 2, 4.5, 7, 9], [0, 0.5, 1, 0.5, 0]),
    (fz.psigmoid(2, 50, 7, -50), [0, 2, 4.5, 7, 9], [0, 0.5, 1, 0.5, 0]),
    (fz.s_shape(1, 8), [0, 1, 2.75, 4.5, 6.25, 8, 9], [0, 0, 0.125, 0.5, 0.875, 1, 1]),
    (fz.z_shape(1, 8), [0, 1, 2.75, 4.5, 6.25, 8, 9], [1, 1, 0.875, 0.5, 0.125, 0, 0]),
    (fz.pi_shape(1, 4, 5, 10), [1, 2.5, 4, 5, 7.5, 10], [0, 0.5, 1, 1, 0.5, 0]),
]
shapes = pytest.mark.parametrize(
    ("mf", "x", "degrees"), SHAPES, ids=[str(mf) for mf, *_ in SHAPES]
)


@shapes
def test_shape_follows_its_definition(mf, x, degrees):
    assert mf(x) == pytest.approx(degrees, abs=1e-12)
    assert [mf(value) for value in x] == pytest.approx(degrees, abs=1e-12)
    assert isinstance(mf(x[0]), float)


@shapes
def test_large_arrays_take_the_same_values(mf, x, degrees):
    values = np.random.default_rng(0).uniform(-2, 12, (40, 500))
    values.flat[: len(x)] = x
    whole = mf(values)
    rows = np.stack([mf(row) for row in values])  # small enough for the slow path
    assert whole.shape == values.shape
    assert whole == pytest.approx(rows, abs=1e-15)
    assert whole.flat[: len(x)] == pytest.approx(degrees, abs=1e-12)
    assert whole.min() >= 0.0
    assert whole.max() <= 1.0


BAD = {
    "descending": lambda: fz.triangle(3, 2, 1),
    "infinite": lambda: fz.trapezoid(0, 1, float("inf"), 3),
    "flat gauss": lambda: fz.gauss(0, 0),
    "negative slope": lambda: fz.bell(0, 1, -1),
    "vertical spline": lambda: fz.s_shape(2, 2),
    "degree above 1": lambda: fz.piecewise([(0, 0), (1, 1.5)]),
}


@pytest.mark.parametrize("build", BAD.values(), ids=BAD.keys())
def test_bad_parameters_are_rejected(build):
    with pytest.raises(ValueError, match=r"must|needs"):
        build()


def test_sets_combine_pointwise():
    low, high = fz.falling(0, 10), fz.rising(0, 10)
    x = np.array([2.0, 5.0, 8.0])
    assert (low & high)(x) == pytest.approx([0.2, 0.5, 0.2])
    assert (low | high)(x) == pytest.approx([0.8, 0.5, 0.8])
    assert (~low)(x) == pytest.approx(high(x))
    assert (high**2)(5.0) == 0.25
    assert (
        repr(~(low & high) | high**2)
        == "(~(falling(0, 10) & rising(0, 10)) | rising(0, 10)**2)"
    )
    custom = fz.MF(lambda v: np.exp(-np.abs(v)), "spike")
    assert (custom(0.0), repr(custom)) == (1.0, "spike")


def test_sets_take_positive_powers_only():
    with pytest.raises(ValueError, match="positive power"):
        fz.rising(0, 1) ** -1
    with pytest.raises(ValueError, match="positive power"):
        fz.rising(0, 1) ** 0
