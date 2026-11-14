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
    "single term": lambda: fz.partition("x", (0, 1), ["only"]),
    "touching cores": lambda: fz.partition("x", (0, 10), a=(1, 5), b=5),
    "core outside": lambda: fz.partition("x", (0, 10), a=1, b=12),
    "circle out of order": lambda: fz.partition(
        "x", (0, 360), a=10, b=200, c=100, periodic=True
    ),
}


@pytest.mark.parametrize("build", BAD.values(), ids=BAD.keys())
def test_bad_parameters_are_rejected(build):
    with pytest.raises(ValueError, match=r"must|needs"):
        build()


def test_partition_sums_to_one():
    temp = fz.partition("temp", (0, 40), cold=10, warm=(18, 22), hot=30)
    x = np.linspace(-5, 45, 501)
    assert temp(x).sum(axis=0) == pytest.approx(1.0)
    assert temp.fuzzify(14.0) == pytest.approx({"cold": 0.5, "warm": 0.5, "hot": 0.0})
    assert list(fz.partition("level", (0, 1), ["low", "mid", "high"])(0.25)) == [
        0.5,
        0.5,
        0,
    ]


def test_periodic_partition_wraps_around():
    hue = fz.partition(
        "hue", (0, 360), red=(350, 10), green=120, blue=240, periodic=True
    )
    assert hue.fuzzify(0.0) == {"red": 1.0, "green": 0.0, "blue": 0.0}
    assert hue.fuzzify(295.0) == pytest.approx({"red": 0.5, "green": 0.0, "blue": 0.5})
    x = np.random.default_rng(0).uniform(-720, 720, 5000)
    assert hue(x) == pytest.approx(hue(x % 360))
    assert hue(x).sum(axis=0) == pytest.approx(1.0)
    compass = fz.partition("wind", (0, 360), ["n", "e", "s", "w"], periodic=True)
    assert compass.fuzzify(315.0) == pytest.approx({"n": 0.5, "e": 0, "s": 0, "w": 0.5})


def test_hedges_and_norms():
    tall = fz.rising(160, 200)
    assert fz.very(tall)(180.0) == 0.25
    assert fz.extremely(tall)(180.0) == 0.125
    assert fz.somewhat(tall)(170.0) == 0.5
    a, b = np.array([0.2, 0.9]), np.array([0.5, 0.6])
    assert fz.probor(a, b) == pytest.approx([0.6, 0.96])
    assert fz.bounded_sum(a, b) == pytest.approx([0.7, 1.0])
    assert fz.bounded_diff(a, b) == pytest.approx([0.0, 0.5])


def test_defuzzifiers():
    x = np.linspace(0, 10, 11)
    hill = fz.trapezoid(2, 4, 6, 9)(x)
    assert fz.centroid(x, fz.triangle(2, 4, 9)(x)) == pytest.approx(
        5.0
    )  # (2 + 4 + 9) / 3
    assert fz.bisector(x, hill) == 5.0
    assert (fz.som(x, hill), fz.mom(x, hill), fz.lom(x, hill)) == (4.0, 5.0, 6.0)
    both = np.stack(
        [hill, np.zeros(11)]
    )  # the last axis is the grid; an empty set has no value
    for defuzz in (fz.centroid, fz.bisector, fz.mom, fz.som, fz.lom):
        value, empty = defuzz(x, both)
        assert np.isfinite(value)
        assert np.isnan(empty)


# The tipping problem: two inputs, three rules.
QUALITY = fz.partition("quality", (0, 10), ["poor", "average", "good"])
SERVICE = fz.partition("service", (0, 10), ["poor", "average", "good"])
TIP = fz.partition("tip", (0, 25), low=0, medium=13, high=25)
RULES = (
    QUALITY.poor | SERVICE.poor >> TIP.low,
    SERVICE.average >> TIP.medium,
    SERVICE.good | QUALITY.good >> TIP.high,
)


def test_mamdani_matches_the_centre_of_gravity_found_by_hand():
    # The rules fire at 0, 0.04 and 0.96, which cuts the output set into a triangle,
    # a rectangle, a trapezoid and a rectangle with these areas and moments.
    area = 0.0104 + 0.5184 + 5.52 + 0.4608
    moment = 0.0104 * 0.52 * 2 / 3 + 0.5184 * 7 + 5.52 * 20.6928 + 0.4608 * 24.76
    tipper = fz.Mamdani(*RULES, points=626)  # a step of 0.04 holds every corner
    assert tipper(quality=6.5, service=9.8) == pytest.approx(moment / area)
    assert fz.Mamdani(*RULES)(quality=6.5, service=9.8) == pytest.approx(
        moment / area, abs=1e-3
    )
    assert tipper.aggregate(quality=6.5, service=9.8).max() == pytest.approx(0.96)


def test_mamdani_broadcasts_and_marks_silence():
    tipper = fz.Mamdani(*RULES)
    grid = tipper(quality=np.linspace(0, 10, 3)[:, None], service=np.linspace(0, 10, 4))
    assert grid.shape == (3, 4)
    assert grid[1, 2] == pytest.approx(tipper(quality=5.0, service=20 / 3))
    assert tipper.aggregate(quality=[1.0, 9.0], service=5.0).shape == (2, 101)
    lazy = fz.Mamdani(SERVICE.poor >> TIP.low)
    assert np.isnan(lazy(service=[2.0, 7.0])).tolist() == [False, True]


def test_mamdani_options():
    quality = np.random.default_rng(0).uniform(0, 10, 300)
    service = np.random.default_rng(1).uniform(0, 10, 300)
    expected = fz.Mamdani(*RULES)(quality=quality, service=service)
    for split in (
        QUALITY.poor >> TIP.low,
        SERVICE.poor >> TIP.low,
    ):  # the same first rule
        rules = (split, *RULES)
        ungrouped = fz.Mamdani(*rules, aggregation=lambda a, b: np.maximum(a, b))
        assert fz.Mamdani(*rules)(quality=quality, service=service) == pytest.approx(
            expected
        )
        assert ungrouped(quality=quality, service=service) == pytest.approx(expected)
    larsen = fz.Mamdani(*RULES, implication=np.multiply, defuzz=fz.mom)
    assert larsen(quality=6.5, service=9.8) == 25.0
    half = fz.Mamdani(fz.Rule(SERVICE.good, TIP.high, weight=0.5))
    assert half.aggregate(service=10.0).max() == 0.5


def test_sugeno_is_a_weighted_average():
    x = fz.partition("x", (0, 10), ["low", "high"])
    flat = fz.Sugeno(x.low >> 10, x.high >> 20.0)
    assert flat(x=2.5) == 12.5
    assert flat(x=[0.0, 5.0, 10.0]).tolist() == [10.0, 15.0, 20.0]
    linear = fz.Sugeno(x.low >> 10, x.high >> (lambda x, gain: gain * x))
    assert linear(x=2.5, gain=2.0) == 8.75
    assert np.isnan(fz.Sugeno(fz.very(x.low) >> 1.0)(x=10.0))


def test_systems_reject_wrong_rules_and_inputs():
    with pytest.raises(ValueError, match="one variable"):
        fz.Mamdani(SERVICE.good >> TIP.high, SERVICE.poor >> QUALITY.poor)
    with pytest.raises(ValueError, match="one variable"):
        fz.Mamdani(SERVICE.good >> 3.0)
    with pytest.raises(TypeError, match="number or a function"):
        fz.Sugeno(SERVICE.good >> TIP.high)
    with pytest.raises(TypeError, match="missing input 'service'"):
        fz.Mamdani(*RULES)(quality=5)
