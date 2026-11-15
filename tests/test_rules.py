import numpy as np
import pytest

import fuzlog as fz

TEMP = fz.Variable("temp", (0, 40), cold=fz.falling(10, 20), hot=fz.rising(20, 30))
HUMID = fz.Variable(
    "humid", (0, 1), {"dry": fz.falling(0.2, 0.6), "very wet": fz.rising(0.6, 1)}
)
FAN = fz.partition("fan", (0, 100), ["slow", "fast"])


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
    assert (hot & dry).inputs == {"temp": TEMP, "humid": HUMID}
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fz.Rule(hot, FAN.fast, weight=2)


def test_hedges_change_sets_and_propositions():
    tall = fz.rising(160, 200)
    assert fz.very(tall)(180.0) == 0.25
    assert fz.extremely(tall)(180.0) == 0.125
    assert fz.somewhat(tall)(170.0) == 0.5
    assert fz.extremely(TEMP.hot)(temp=25) == 0.125
    assert repr(fz.somewhat(TEMP.hot)) == "somewhat (temp is hot)"


def test_propositions_take_positive_powers_only():
    with pytest.raises(ValueError, match="positive power"):
        TEMP.hot**-2
