import numpy as np
import pytest

import fuzlog as fz

QUALITY = fz.partition("quality", (0, 10), ["poor", "average", "good"])


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


def test_labels_separator_is_a_string():
    with pytest.raises(TypeError, match="sep must be a string"):
        fz.Labels(QUALITY, sep=1)  # ty: ignore[invalid-argument-type]
