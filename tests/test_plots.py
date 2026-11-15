import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import fuzlog as fz

TEMP = fz.partition("temp", (0, 40), cold=10, warm=20, hot=30)
FAN = fz.partition("fan", (0, 100), ["slow", "medium", "fast"])
RULES = (TEMP.cold >> FAN.slow, TEMP.warm >> FAN.medium, TEMP.hot >> FAN.fast)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def names(ax):
    """The term names in the legend of a plot as they read: along the rows."""
    ax.figure.canvas.draw()
    legend = ax.get_legend()
    assert legend is not None
    marks = [mark.get_window_extent() for mark in legend.legend_handles]
    places = [(-round(box.y0), round(box.x0)) for box in marks]  # a row stands level
    terms = [text.get_text() for text in legend.get_texts()]
    return [term for _, term in sorted(zip(places, terms, strict=True))]


def rows(ax):
    """How many rows the legend of a plot takes, once drawn."""
    ax.figure.canvas.draw()
    legend = ax.get_legend()
    assert legend is not None
    return len({round(mark.get_window_extent().y0) for mark in legend.legend_handles})


def test_variable_plot_names_every_term_below():
    ax = fz.plot(TEMP)
    assert len(ax.lines) == 3
    assert names(ax) == ["cold", "warm", "hot"]
    assert rows(ax) == 1
    legend, label = ax.get_legend(), ax.xaxis.label
    assert legend is not None
    top = max(text.get_window_extent().y1 for text in legend.get_texts())
    assert top < label.get_window_extent().y0  # under the axis label
    assert ax.get_xlabel() == "temp"
    hue = fz.partition(
        "hue", (0, 360), red=(350, 10), green=120, blue=240, periodic=True
    )
    assert names(fz.plot(hue)) == ["red", "green", "blue"]  # once, seam or not
    _, given = plt.subplots()
    assert fz.plot(TEMP, given) is given


def test_many_terms_take_rows_of_five_and_their_own_colors():
    many = fz.partition("x", (0, 1), [f"t{i}" for i in range(10)])
    ax = fz.plot(many)
    assert rows(ax) == 2  # five to a row
    assert names(ax) == [f"t{i}" for i in range(10)]  # in order along the rows
    colors = [line.get_color() for line in ax.lines]
    assert len(set(colors[:8])) == 8  # eight hues, then over again
    assert colors[8:] == colors[:2]


def test_inference_plot_marks_the_crisp_value():
    system = fz.Mamdani(*RULES)
    ax = fz.plot(system, temp=27.0)
    assert len(ax.lines) == 5  # three terms, the output set and the crisp value
    assert np.asarray(ax.lines[-1].get_xdata())[0] == system(temp=27.0)
    assert ax.texts[-1].get_text() == f"fan = {system(temp=27.0):.3g}"
    with pytest.raises(ValueError, match="scalar inputs"):
        fz.plot(system, temp=[20.0, 27.0])  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "system",
    [fz.Mamdani(*RULES), fz.Sugeno(TEMP.cold >> 10, TEMP.hot >> 90)],
    ids=["mamdani", "sugeno"],
)
def test_surface_plot_spans_two_inputs(system):
    humid = fz.partition("humid", (0, 1), ["dry", "wet"])
    rules = (*system.rules, humid.wet >> system.rules[-1].then)
    ax = fz.surface(type(system)(*rules), "temp", "humid", points=8)
    mesh = ax.collections[0]
    assert np.asarray(mesh.get_array()).size == 64
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("temp", "humid")
    assert len(ax.figure.axes) == 2  # the plot and its colour bar
