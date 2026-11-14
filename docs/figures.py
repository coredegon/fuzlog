"""Draw the figures of the documentation: ``uv run docs/figures.py``."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fuzl as fz

IMAGES = Path(__file__).parent / "img"
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


def shapes() -> None:
    """The gallery of shapes.md."""
    figure, axes = plt.subplots(
        4, 4, figsize=(9, 5.6), sharex=True, sharey=True, layout="constrained"
    )
    x = np.linspace(0, 10, 2001)
    for ax, call in zip(axes.flat, SHAPES, strict=True):
        degrees = eval(f"fz.{call}")(x)
        ax.plot(x, degrees, color="#2a78d6", linewidth=2)
        ax.fill_between(x, degrees, color="#2a78d6", alpha=0.1, linewidth=0)
        title = call if len(call) < 30 else "piecewise([(1, 0), (3, 1), ...])"
        ax.set_title(title, fontsize=8, family="monospace")
        ax.set(xlim=(0, 10), ylim=(0, 1.05), xticks=(0, 5, 10), yticks=(0, 1))
        ax.spines[["top", "right"]].set_visible(False)
    figure.savefig(IMAGES / "shapes.png", dpi=144)


def inference(system: fz.Mamdani, name: str, **inputs: float) -> None:
    """An inferred output set beside the surface over the two inputs."""
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(9, 3.2), width_ratios=(3, 2), layout="constrained"
    )
    system.plot(left, **inputs)
    system.surface(*inputs, right)
    figure.savefig(IMAGES / name, dpi=144)


def readme() -> None:
    """The system of README.md."""
    service = fz.partition("service", (0, 10), ["poor", "good", "excellent"])
    food = fz.Variable("food", (0, 10), rancid=fz.z_shape(1, 4), tasty=fz.s_shape(6, 9))
    tip = fz.partition("tip", (0, 30), cheap=5, average=15, generous=25)
    tipper = fz.Mamdani(
        service.poor | food.rancid >> tip.cheap,
        service.good >> tip.average,
        service.excellent | food.tasty >> tip.generous,
    )
    inference(tipper, "tipper.png", service=3, food=8)


def guide() -> None:
    """The variable and the system of guide.md."""
    temp = fz.Variable(
        "temp",
        (0, 40),
        cold=fz.falling(10, 20),
        warm=fz.triangle(10, 20, 30),
        hot=fz.rising(20, 30),
    )
    humid = fz.partition("humid", (0, 100), ["dry", "comfortable", "wet"])
    fan = fz.partition("fan", (0, 100), ["slow", "medium", "fast"])
    cooler = fz.Mamdani(
        temp.cold >> fan.slow,
        temp.warm >> fan.medium,
        temp.hot | humid.wet >> fan.fast,
    )
    figure, ax = plt.subplots(figsize=(7, 2.6), layout="constrained")
    temp.plot(ax)
    figure.savefig(IMAGES / "variable.png", dpi=144)
    inference(cooler, "cooler.png", temp=27, humid=40)


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)
    shapes()
    readme()
    guide()
