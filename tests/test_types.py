import numpy as np
import pytest

import fuzlog as fz

TEMP = fz.Variable("temp", (0, 40), cold=fz.falling(10, 20), hot=fz.rising(20, 30))
HUMID = fz.Variable(
    "humid", (0, 1), {"dry": fz.falling(0.2, 0.6), "very wet": fz.rising(0.6, 1)}
)
FAN = fz.partition("fan", (0, 100), ["slow", "fast"])


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


def test_propositions_hold_to_a_degree():
    hot_and_dry = TEMP.hot & HUMID.dry
    assert hot_and_dry(temp=27.5, humid=0.3) == 0.75
    assert (TEMP.hot | HUMID.dry)(temp=27.5, humid=0.2) == 1.0
    assert (~TEMP.hot)(temp=27.5) == 0.25
    assert fz.very(TEMP.hot)(temp=25) == 0.25
    grid = hot_and_dry(temp=np.array([[22.5], [30.0]]), humid=np.array([0.2, 0.4, 0.6]))
    assert grid.tolist() == [[0.25, 0.25, 0.0], [1.0, 0.5, 0.0]]
    with pytest.raises(TypeError, match="no truth value"):
        _ = TEMP.hot and HUMID.dry


def test_rules_need_no_brackets():
    hot, dry, wet = TEMP.hot, HUMID.dry, HUMID["very wet"]
    rule = hot & dry >> FAN.fast
    assert isinstance(rule, fz.Rule)
    assert (
        repr(rule)
        == repr((hot & dry) >> FAN.fast)
        == "if temp is hot and humid is dry then fan is fast"
    )
    mixed = ~hot | fz.very(dry) & wet >> FAN.slow
    assert repr(mixed) == (
        "if not (temp is hot) or very (humid is dry) and humid is very wet "
        "then fan is slow"
    )
    assert repr((hot | dry) & wet >> 3.5) == (
        "if (temp is hot or humid is dry) and humid is very wet then 3.5"
    )
    assert (
        repr(fz.Rule(hot, FAN.fast, weight=0.5))
        == "if temp is hot then fan is fast [0.5]"
    )
    assert (hot & dry).variables == (TEMP, HUMID)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fz.Rule(hot, FAN.fast, weight=2)
