import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import fuzl as fz  # noqa: E402

TEMP = fz.partition("temp", (0, 40), cold=10, warm=20, hot=30)
FAN = fz.partition("fan", (0, 100), ["slow", "medium", "fast"])
RULES = (TEMP.cold >> FAN.slow, TEMP.warm >> FAN.medium, TEMP.hot >> FAN.fast)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def names(ax):
    return [label.get_text() for label in ax.child_axes[0].get_xticklabels()]


def test_variable_plot_names_every_term():
    ax = TEMP.plot()
    assert len(ax.lines) == 3
    assert names(ax) == ["cold", "warm", "hot"]
    assert ax.get_xlabel() == "temp"
    hue = fz.partition(
        "hue", (0, 360), red=(350, 10), green=120, blue=240, periodic=True
    )
    assert names(hue.plot()) == ["red", "green", "blue", "red"]  # named at both ends
    _, given = plt.subplots()
    assert TEMP.plot(given) is given


def test_inference_plot_marks_the_crisp_value():
    system = fz.Mamdani(*RULES)
    ax = system.plot(temp=27.0)
    assert len(ax.lines) == 5  # three terms, the output set and the crisp value
    assert np.asarray(ax.lines[-1].get_xdata())[0] == system(temp=27.0)
    assert ax.texts[0].get_text() == f"fan = {system(temp=27.0):.3g}"
    with pytest.raises(ValueError, match="scalar inputs"):
        system.plot(temp=[20.0, 27.0])  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "system",
    [fz.Mamdani(*RULES), fz.Sugeno(TEMP.cold >> 10, TEMP.hot >> 90)],
    ids=["mamdani", "sugeno"],
)
def test_surface_plot_spans_two_inputs(system):
    humid = fz.partition("humid", (0, 1), ["dry", "wet"])
    rules = (*system.rules, humid.wet >> system.rules[-1].then)
    ax = type(system)(*rules).surface("temp", "humid", points=8)
    mesh = ax.collections[0]
    assert np.asarray(mesh.get_array()).size == 64
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("temp", "humid")
    assert len(ax.figure.axes) == 2  # the plot and its colour bar
