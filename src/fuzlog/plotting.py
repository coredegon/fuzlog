"""Matplotlib views of linguistic variables and inference systems.

Every function draws on the Axes it is given, or on a new figure, and returns the
Axes, so it works the same in scripts and in notebooks. Fonts, sizes and the colour
of text and axes follow matplotlib's configuration; only data wears colour.
"""

from __future__ import annotations

import math

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as error:  # pragma: no cover
    hint = 'plotting needs matplotlib: pip install "fuzlog[plot]"'
    raise ModuleNotFoundError(hint) from error
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_rgba

from fuzlog.inference import Mamdani, Sugeno
from fuzlog.variables import Variable

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


def plot_variable(variable: Variable, ax: Axes | None = None) -> Axes:
    """Draw the membership function of every term, named above its peak.

    A term that wraps around a periodic universe is named at both ends.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 2.6), layout="constrained")
    _draw_terms(variable, ax, linewidth=2.0, wash=0.1)
    return ax


def plot_inference(system: Mamdani, ax: Axes | None = None, **inputs: float) -> Axes:
    """Draw the output set a Mamdani system infers for scalar inputs.

    The terms of the output variable are outlined, the inferred set is shaded and a
    vertical line marks the crisp value.
    """
    degrees = system.aggregate(**inputs)
    if degrees.ndim != 1:
        raise ValueError("an inference plot shows one sample: pass scalar inputs")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 2.6), layout="constrained")
    _draw_terms(system.output, ax, linewidth=1.0, wash=0.0)
    ink = plt.rcParams["text.color"]
    ax.fill_between(system.x, degrees, color=ink, alpha=0.2, linewidth=0)
    ax.plot(system.x, degrees, color=ink, linewidth=2)
    crisp = system(**inputs)
    if not math.isnan(crisp):
        ax.axvline(crisp, color=ink, linewidth=1)
        low, high = system.output.universe
        left = crisp > (low + high) / 2  # write on the roomier side of the line
        ax.annotate(
            f"{system.output.name} = {crisp:.3g}",
            (crisp, 0.5),
            xytext=(-6 if left else 6, 0),
            textcoords="offset points",
            ha="right" if left else "left",
            va="center",
            bbox={"facecolor": ax.get_facecolor(), "edgecolor": "none", "pad": 1.5},
        )
    return ax


def plot_surface(
    system: Mamdani | Sugeno,
    x: str,
    y: str,
    ax: Axes | None = None,
    *,
    points: int = 64,
    **fixed: float,
) -> Axes:
    """Draw the output of a system over two of its inputs, as a labelled heat map.

    ``x`` and ``y`` name the inputs on the axes, sampled at ``points`` values each
    over their universes; all other inputs are ``fixed`` to the given values.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    xs, ys = (np.linspace(*system.inputs[name].universe, points) for name in (x, y))
    z = system(**{x: xs[None, :], y: ys[:, None]}, **fixed)
    colors = LinearSegmentedColormap.from_list("fuzlog", _RAMPS[_dark(ax)])
    mesh = ax.pcolormesh(xs, ys, z, cmap=colors, shading="gouraud", rasterized=True)
    ink = plt.rcParams["text.color"]
    levels = ax.contour(xs, ys, z, levels=6, colors=ink, linewidths=0.5)
    ax.clabel(levels, fmt="%.3g", fontsize="small")
    name = system.output.name if isinstance(system, Mamdani) else "output"
    ax.figure.colorbar(mesh, ax=ax, label=name)
    ax.set(xlabel=x, ylabel=y)
    return ax


def _draw_terms(variable: Variable, ax: Axes, *, linewidth: float, wash: float) -> None:
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
    places, terms = variable.peaks().tolist(), list(variable.terms)
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


def _dark(ax: Axes) -> bool:
    """Tell whether the Axes has a dark surface, which takes its own colour steps."""
    red, green, blue, alpha = to_rgba(ax.get_facecolor())
    if alpha == 0.0:
        red, green, blue, _ = to_rgba(ax.figure.get_facecolor())
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5
