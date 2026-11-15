import numpy as np
import pytest

import fuzlog as fz

TEMP = fz.Variable("temp", (0, 40), cold=fz.falling(10, 20), hot=fz.rising(20, 30))
HUMID = fz.Variable(
    "humid", (0, 1), {"dry": fz.falling(0.2, 0.6), "very wet": fz.rising(0.6, 1)}
)


def test_variable_fuzzifies():
    assert TEMP.fuzzify(12.5) == {"cold": 0.75, "hot": 0.0}
    assert TEMP([[5, 25], [15, 35]]).tolist() == [
        [[1, 0], [0.5, 0]],
        [[0, 0.5], [0, 1]],
    ]
    cold, hot = TEMP(np.array([12.5, 27.5]))
    assert (cold.tolist(), hot.tolist()) == ([0.75, 0.0], [0.0, 0.75])
    assert TEMP.grid(5).tolist() == [0, 10, 20, 30, 40]
    assert (
        repr(TEMP)
        == "Variable('temp', (0, 40), {'cold': falling(10, 20), 'hot': rising(20, 30)})"
    )


def test_peaks_sit_in_the_middle_of_the_cores():
    assert fz.partition("x", (0, 10), a=(0, 2), b=5, c=10).peaks() == pytest.approx(
        [1, 5, 10]
    )
    hue = fz.partition(
        "hue", (0, 360), red=(350, 10), green=120, blue=240, periodic=True
    )
    assert hue.peaks() == pytest.approx([0, 120, 240], abs=1e-6)


def test_terms_are_reached_by_attribute_or_item():
    assert repr(TEMP.cold) == repr(TEMP["cold"]) == "temp is cold"
    assert repr(HUMID["very wet"]) == "humid is very wet"
    with pytest.raises(AttributeError, match="no term 'warm', only"):
        _ = TEMP.warm
    with pytest.raises(TypeError, match="needs terms"):
        fz.Variable("x", (0, 1), big=lambda x: x)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="finite low < high"):
        fz.Variable("x", (1, 1), big=fz.rising(0, 1))


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


BAD = {
    "single term": lambda: fz.partition("x", (0, 1), ["only"]),
    "touching cores": lambda: fz.partition("x", (0, 10), a=(1, 5), b=5),
    "core outside": lambda: fz.partition("x", (0, 10), a=1, b=12),
    "circle out of order": lambda: fz.partition(
        "x", (0, 360), a=10, b=200, c=100, periodic=True
    ),
}


@pytest.mark.parametrize("build", BAD.values(), ids=BAD.keys())
def test_bad_partitions_are_rejected(build):
    with pytest.raises(ValueError, match=r"must|needs"):
        build()


def test_a_circle_has_no_place_for_infinity():
    wind = fz.partition("wind", (0, 360), ["n", "e", "s", "w"], periodic=True)
    assert np.isnan(wind([np.inf, -np.inf, np.nan])).all()
    assert wind.fuzzify(np.inf).keys() == wind.terms.keys()


def test_terms_of_a_partition_differ():
    with pytest.raises(ValueError, match="must differ"):
        fz.partition("x", (0, 1), ["low", "low", "high"])
