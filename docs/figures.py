"""Draw the figures of the documentation: ``uv run docs/figures.py``."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fuzlog as fz

IMAGES = Path(__file__).parent / "img"
BLUE, ORANGE, GREEN, INK = "#2a78d6", "#eb6834", "#1baf7a", "#333333"
SHAPES = [
    "triangle(2, 5, 8)",
    "trapezoid(1, 3, 6, 9)",
    "rising(3, 7)",
    "falling(3, 7)",
    "rectangle(3, 7)",
    "singleton(5)",
    "piecewise([(1, 0), (3, 1), (6, 0.4), (9, 0)])",
    "gauss(5, 1.5)",
    "gauss2(4, 1, 6, 2)",
    "bell(5, 2, 3)",
    "sigmoid(5, 2)",
    "dsigmoid(3, 4, 7, 4)",
    "psigmoid(3, 4, 7, -4)",
    "s_shape(2, 8)",
    "z_shape(2, 8)",
    "pi_shape(1, 4, 6, 9)",
]

TEMP = fz.Variable(
    "temp",
    (0, 40),
    cold=fz.falling(10, 20),
    warm=fz.triangle(10, 20, 30),
    hot=fz.rising(20, 30),
)
HUMID = fz.partition("humid", (0, 100), ["dry", "comfortable", "wet"])
FAN = fz.partition("fan", (0, 100), ["slow", "medium", "fast"])
COOLER = fz.Mamdani(
    TEMP.cold >> FAN.slow,
    TEMP.warm >> FAN.medium,
    TEMP.hot | HUMID.wet >> FAN.fast,
)


def save(figure, name):
    figure.savefig(IMAGES / name, dpi=144)
    plt.close(figure)


def bare(ax, title):
    ax.set_title(title, fontsize=8, family="monospace")
    ax.set(ylim=(0, 1.05), yticks=(0, 1))
    ax.spines[["top", "right"]].set_visible(False)


def shapes():
    """The gallery of shapes.md."""
    figure, axes = plt.subplots(
        4, 4, figsize=(9, 5.6), sharex=True, sharey=True, layout="constrained"
    )
    x = np.linspace(0, 10, 2001)
    for ax, call in zip(axes.flat, SHAPES, strict=True):
        degrees = eval(f"fz.{call}")(x)
        ax.plot(x, degrees, color=BLUE, linewidth=2)
        ax.fill_between(x, degrees, color=BLUE, alpha=0.1, linewidth=0)
        bare(ax, call if len(call) < 30 else "piecewise([(1, 0), (3, 1), ...])")
        ax.set(xlim=(0, 10), xticks=(0, 5, 10))
    save(figure, "shapes.png")


def sets():
    """Two sets and what the operators make of them."""
    warm, hot = fz.triangle(10, 20, 30), fz.rising(20, 40)
    made = {
        "warm | hot": warm | hot,
        "warm & hot": warm & hot,
        "~hot": ~hot,
        "fz.very(hot)": fz.very(hot),
    }
    figure, axes = plt.subplots(
        1, 4, figsize=(9, 1.9), sharex=True, sharey=True, layout="constrained"
    )
    x = np.linspace(0, 40, 801)
    for ax, (title, result) in zip(axes, made.items(), strict=True):
        ax.plot(x, warm(x), color=ORANGE, linewidth=1, linestyle="--")
        ax.plot(x, hot(x), color=GREEN, linewidth=1, linestyle="--")
        ax.plot(x, result(x), color=BLUE, linewidth=2)
        ax.fill_between(x, result(x), color=BLUE, alpha=0.1, linewidth=0)
        bare(ax, title)
        ax.set(xlim=(0, 40), xticks=(0, 20, 40))
    save(figure, "sets.png")


def variable():
    """The terms of a variable."""
    figure, ax = plt.subplots(figsize=(7, 2.4), layout="constrained")
    fz.plot(TEMP, ax)
    save(figure, "variable.png")


def partitions():
    """The three ways to give a partition."""
    made = [
        fz.partition("level", (0, 1), ["low", "medium", "high"]),
        fz.partition("age", (0, 100), young=(0, 25), adult=(35, 55), old=(70, 100)),
        fz.partition(
            "hue", (0, 360), red=(350, 10), green=120, blue=240, periodic=True
        ),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(9, 2.2), layout="constrained")
    for ax, made_variable in zip(axes, made, strict=True):
        fz.plot(made_variable, ax)
        ax.set_ylabel("")
    save(figure, "partitions.png")


def inference(system, name, **inputs):
    """An inferred output set beside the surface over the two inputs."""
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(9, 3.2), width_ratios=(3, 2), layout="constrained"
    )
    fz.plot(system, left, **inputs)
    fz.surface(system, *inputs, right)
    save(figure, name)


def tipper():
    """The system of README.md."""
    service = fz.partition("service", (0, 10), ["poor", "good", "excellent"])
    food = fz.Variable("food", (0, 10), rancid=fz.z_shape(1, 4), tasty=fz.s_shape(6, 9))
    tip = fz.partition("tip", (0, 30), cheap=5, average=15, generous=25)
    system = fz.Mamdani(
        service.poor | food.rancid >> tip.cheap,
        service.good >> tip.average,
        service.excellent | food.tasty >> tip.generous,
    )
    inference(system, "tipper.png", service=3, food=8)


def defuzz():
    """One output set and the value each defuzzifier takes from it."""
    x, mu = COOLER.x, COOLER.aggregate(temp=27, humid=40)
    figure, ax = plt.subplots(figsize=(7, 2.4), layout="constrained")
    ax.fill_between(x, mu, color=INK, alpha=0.15, linewidth=0)
    ax.plot(x, mu, color=INK, linewidth=1.5)
    methods = (fz.centroid, fz.bisector, fz.som, fz.mom, fz.lom)
    for method in methods:
        value = float(method(x, mu))
        ax.axvline(value, color=BLUE, linewidth=1)
        ax.annotate(
            method.__name__,
            (value, 1.03),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="top",
            rotation=90,
            fontsize=8,
            color=BLUE,
        )
    ax.set(xlim=(0, 100), ylim=(0, 1.05), yticks=(0, 0.5, 1), xlabel="fan")
    ax.spines[["top", "right"]].set_visible(False)
    save(figure, "defuzz.png")


def tables():
    """A system, a coarse table of it and a table of labels."""
    figure, axes = plt.subplots(1, 3, figsize=(9, 2.7), layout="constrained")
    temp, humid = np.linspace(0, 40, 161), np.linspace(0, 100, 161)
    grid = {"temp": temp[None, :], "humid": humid[:, None]}
    coarse = fz.tabulate(COOLER, points=9, method="nearest")
    words = fz.tabulate(fz.Labels(TEMP, HUMID, sep=chr(10)))
    codes = words.codes(**grid)
    assert words.labels is not None
    order = [*TEMP.terms, *HUMID.terms]
    checks = np.array([sum(map(order.index, w.split())) % 2 for w in words.labels])
    views = {
        "cooler": COOLER(**grid),
        'tabulate(cooler, points=9, method="nearest")': coarse(**grid),
        "tabulate(Labels(temp, humid))": checks[codes],
    }
    for ax, (title, z) in zip(axes, views.items(), strict=True):
        ax.pcolormesh(temp, humid, z, cmap="Blues", vmax=1.5 * z.max(), rasterized=True)
        ax.set_title(title, fontsize=8, family="monospace")
        ax.set(xlabel="temp", ylabel="humid", xticks=(0, 20, 40), yticks=(0, 50, 100))
    for code, label in enumerate(words.labels):
        rows, columns = np.nonzero(codes == code)
        at = (temp[columns].mean(), humid[rows].mean())
        axes[2].annotate(label, at, ha="center", va="center", fontsize=7)
    save(figure, "tables.png")


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)
    for draw in (shapes, sets, variable, partitions, tipper, defuzz, tables):
        draw()
    inference(COOLER, "cooler.png", temp=27, humid=40)
