import numpy as np
import pytest

import fuzlog as fz


def test_norms():
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


def test_labels_describe_inputs_by_their_strongest_terms():
    temp = fz.partition("temp", (0, 40), cold=10, warm=20, hot=30)
    humid = fz.partition("humid", (0, 100), ["dry", "comfortable", "wet"])
    words = fz.Labels(temp, humid)
    assert words(temp=27, humid=40) == "hot_comfortable"
    assert isinstance(words(temp=27, humid=40), str)
    assert words(temp=[[5.0], [27.0]], humid=[10.0, 50.0, 90.0]).tolist() == [
        ["cold_dry", "cold_comfortable", "cold_wet"],
        ["hot_dry", "hot_comfortable", "hot_wet"],
    ]
    assert words(temp=15, humid=0) == "cold_dry"  # the first of equal terms wins
    assert repr(words) == "Labels(temp, humid)"
    assert list(words.inputs) == ["temp", "humid"]


def test_labels_join_terms_with_any_separator():
    temp = fz.partition("temp", (0, 40), cold=10, warm=20, hot=30)
    wind = fz.partition(
        "wind", (0, 360), ["north", "east", "south", "west"], periodic=True
    )
    assert fz.Labels(temp, wind, sep=" ")(temp=31, wind=350) == "hot north"
    assert fz.Labels(wind, temp, sep=", ")(temp=31, wind=100) == "east, hot"
    assert fz.Labels(temp, sep="")(temp=[1, 39]).tolist() == ["cold", "hot"]


def test_labels_are_empty_where_no_term_holds():
    level = fz.Variable("level", (0, 10), low=fz.falling(2, 4), high=fz.rising(6, 8))
    other = fz.partition("other", (0, 1), ["a", "b"])
    words = fz.Labels(level, other)
    assert words(level=[1, 5, 9, np.nan], other=0).tolist() == [
        "low_a",
        "",
        "high_a",
        "",
    ]
    assert words(level=5, other=0) == ""


def test_labels_need_variables_and_inputs():
    temp = fz.partition("temp", (0, 40), cold=10, warm=20, hot=30)
    with pytest.raises(TypeError, match="describe variables"):
        fz.Labels()
    with pytest.raises(TypeError, match="describe variables"):
        fz.Labels(temp, temp.hot)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="must differ"):
        fz.Labels(temp, temp)
    with pytest.raises(TypeError, match="missing input 'temp'"):
        fz.Labels(temp)(humid=3)


def test_sugeno_conclusion_counts_only_where_its_rule_fires():
    x = fz.partition("x", (0, 10), low=2, high=8)
    system = fz.Sugeno(x.low >> 1.0, x.high >> (lambda x: 1.0 / (x - 1.0)))
    with np.errstate(divide="ignore"):
        assert system(x=[1.0, 8.0]).tolist() == [1.0, 1.0 / 7.0]


def test_labels_separator_is_a_string():
    with pytest.raises(TypeError, match="sep must be a string"):
        fz.Labels(QUALITY, sep=1)  # ty: ignore[invalid-argument-type]
