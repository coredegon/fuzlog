"""Plots: matplotlib views of variables and systems.

Every function draws on the Axes it is given, or on a new figure, and returns the
Axes. Fonts, sizes and the colour of text follow matplotlib's configuration; only
data wears colour. Matplotlib itself is imported by the first plot.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from fuzlog.systems import Mamdani, Sugeno
from fuzlog.types import Array
from fuzlog.variables import Variable

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Eight hues that stay apart under colour blindness when neighbours take
# neighbouring slots: one row for light surfaces, one for dark. Terms take them in
# order; with more terms than hues every curve wears the first.
_HUES = (
    ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
     "#e87ba4", "#008300", "#6250d6", "#e34948"),
    ("#3987e5", "#d95926", "#199e70", "#c98500",
     "#d55181", "#0ca30c", "#9085e9", "#e66767"),
)  # fmt: skip
# One hue for magnitudes, from the surface towards a blue that text stays legible on:
# again a row for light surfaces and a row for dark.
_RAMPS = (
    ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef",
     "#6da7ec", "#5598e7", "#3987e5", "#2a78d6"),
    ("#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6"),
)  # fmt: skip
_SAMPLES = 1001


def plot(what: Variable | Mamdani, ax: Axes | None = None, /, **inputs: float) -> Axes:
    """Draw a variable, or what a Mamdani system infers for scalar inputs.

    ```python
    fz.plot(temp)                          # every term, named above its peak
    fz.plot(cooler, temp=27, humid=40)     # the output set and its crisp value
    ```
    """
    ax, ink = _canvas(ax, (7, 2.6))
    if isinstance(what, Variable):
        _terms(what, ax, linewidth=2.0, wash=0.1)
        return ax
    degrees = what.aggregate(**inputs)
    if degrees.ndim != 1:
        raise ValueError("an inference plot shows one sample: pass scalar inputs")
    _terms(what.output, ax, linewidth=1.0, wash=0.0)
    ax.fill_between(what.x, degrees, color=ink, alpha=0.2, linewidth=0)
    ax.plot(what.x, degrees, color=ink, linewidth=2)
    crisp = what(**inputs)
    if not math.isnan(crisp):
        ax.axvline(crisp, color=ink, linewidth=1)
        left = crisp > sum(what.output.universe) / 2  # write on the roomier side
        ax.annotate(
            f"{what.output.name} = {crisp:.3g}",
            (crisp, 0.5),
            xytext=(-6 if left else 6, 0),
            textcoords="offset points",
            ha="right" if left else "left",
            va="center",
            bbox={"facecolor": ax.get_facecolor(), "edgecolor": "none", "pad": 1.5},
        )
    return ax


def surface(
    system: Mamdani | Sugeno,
    x: str,
    y: str,
    /,
    ax: Axes | None = None,
    *,
    points: int = 64,
    **fixed: float,
) -> Axes:
    """Draw the output of a system over two of its inputs, as a labelled heat map.

    ```python
    fz.surface(cooler, "temp", "humid")
    fz.surface(cooler, "temp", "humid", wind=3)    # other inputs are fixed
    ```
    """
    from matplotlib.colors import LinearSegmentedColormap

    ax, ink = _canvas(ax, (5, 4))
    xs, ys = (np.linspace(*system.inputs[name].universe, points) for name in (x, y))
    z = system(**{x: xs[None, :], y: ys[:, None]}, **fixed)
    colors = LinearSegmentedColormap.from_list("fuzlog", _RAMPS[_dark(ax)])
    mesh = ax.pcolormesh(xs, ys, z, cmap=colors, shading="gouraud", rasterized=True)
    levels = ax.contour(xs, ys, z, levels=6, colors=ink, linewidths=0.5)
    ax.clabel(levels, fmt="%.3g", fontsize="small")
    name = system.output.name if isinstance(system, Mamdani) else "output"
    ax.figure.colorbar(mesh, ax=ax, label=name)
    ax.set(xlabel=x, ylabel=y)
    return ax


def _canvas(ax: Axes | None, size: tuple[float, float]) -> tuple[Axes, str]:
    """The Axes to draw on, new if none is given, and the colour of its text."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=size, layout="constrained")
    return ax, plt.rcParams["text.color"]


def _terms(variable: Variable, ax: Axes, *, linewidth: float, wash: float) -> None:
    """Draw the curve of every term and write the term names along the top."""
    low, high = variable.universe
    x = np.linspace(low, high, _SAMPLES)
    degrees = variable(x)
    hues = _HUES[_dark(ax)]
    for i, curve in enumerate(degrees):
        hue = hues[i] if len(degrees) <= len(hues) else hues[0]
        ax.plot(x, curve, color=hue, linewidth=linewidth)
        if wash:
            ax.fill_between(x, curve, color=hue, alpha=wash, linewidth=0)
    places, terms = _peaks(variable).tolist(), list(variable.terms)
    owner = int(degrees[:, 0].argmax())
    if variable.periodic and degrees[owner, 0] > 0.5:
        # The term that owns the seam is named at the far end too.
        places.append(low if places[owner] > (low + high) / 2 else high)
        terms.append(terms[owner])
    names = ax.secondary_xaxis("top")
    crowded = len(terms) > 6
    names.set_xticks(
        places,
        labels=terms,
        rotation=45 if crowded else 0,
        ha="left" if crowded else "center",
        rotation_mode="anchor",
    )
    names.tick_params(length=0)
    names.spines["top"].set_visible(False)
    ax.set(xlim=(low, high), ylim=(0.0, 1.05), yticks=(0.0, 0.5, 1.0))
    ax.set(xlabel=variable.name, ylabel="membership")
    ax.spines[["top", "right"]].set_visible(False)


def _peaks(variable: Variable) -> Array:
    """The most typical value of every term: the middle of where it is highest.

    On a periodic universe the middle is found going around the circle.
    """
    x = variable.grid(2001)
    degrees = variable(x)
    core = (degrees >= degrees.max(axis=1, keepdims=True) - 1e-9).astype(np.float64)
    if not variable.periodic:
        return (core @ x) / core.sum(axis=1)
    low, high = variable.universe
    angle = (x - low) * (math.tau / (high - low))
    middle = np.arctan2(core @ np.sin(angle), core @ np.cos(angle)) % math.tau
    return low + middle * ((high - low) / math.tau)


def _dark(ax: Axes) -> bool:
    """Tell whether the Axes has a dark surface, which takes its own colour steps."""
    from matplotlib.colors import to_rgba

    red, green, blue, alpha = to_rgba(ax.get_facecolor())
    if alpha == 0.0:
        red, green, blue, _ = to_rgba(ax.figure.get_facecolor())
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5
