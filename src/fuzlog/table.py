"""Compiled systems: lookup tables that are built once, saved and loaded.

``compile`` evaluates a system on a grid of its inputs and keeps the results in a
:class:`Table`. The table is called like the system and answers by looking the
inputs up, so its speed does not depend on the rules, and it is saved to disk as
plain numbers: loading one needs neither the rules nor the code that made it.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import zipfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, NamedTuple, overload

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from fuzlog.types import Array, Prop, Variable

type Span = int | tuple[float, float] | tuple[float, float, int]
type Method = Literal["linear", "nearest"]

_FORMAT = 1  # version of the file layout
_CHUNK = 1 << 14  # grid nodes evaluated in one call of the system


class Axis(NamedTuple):
    """One input of a table: ``points`` evenly spaced values from ``low`` to ``high``.

    A ``periodic`` axis is a circle on which ``high`` is the same point as ``low``,
    so its last node lies one step before ``high``.
    """

    name: str
    low: float
    high: float
    points: int
    periodic: bool = False

    def grid(self) -> Array:
        """Return the values of the nodes."""
        return np.linspace(self.low, self.high, self.points, endpoint=not self.periodic)

    def locate(self, x: Array) -> tuple[Array, NDArray[np.bool_] | None]:
        """Return the fractional node index of ``x``, and where ``x`` is NaN.

        Values outside a plain axis are moved to its nearest end; on a periodic
        axis they are wrapped around.
        """
        if self.periodic:
            period = self.high - self.low
            with np.errstate(invalid="ignore"):  # an infinite value becomes NaN
                at = np.mod(x - self.low, period)
            at *= self.points / period
        else:
            at = (x - self.low) * ((self.points - 1) / (self.high - self.low))
            np.clip(at, 0.0, self.points - 1.0, out=at)
        missing = np.isnan(at)
        if not missing.any():
            return at, None
        at[missing] = 0.0
        return at, missing


class Table:
    """A compiled system: its output on a grid of the inputs.

    Calling a table takes the inputs by name, as scalars or as arrays that
    broadcast against each other, like the system it was compiled from. With the
    ``linear`` method the output is interpolated between the surrounding nodes;
    ``nearest`` takes the closest node, which is faster and exact when the inputs
    only take the values of the nodes.

    Tables are made by :func:`compile` and :func:`load`. ``axes`` describe the
    inputs, ``values`` holds the output with one dimension per axis, and ``source``
    names what was compiled.
    """

    __slots__ = ("_holes", "axes", "method", "source", "values")

    def __init__(
        self,
        axes: Iterable[Axis],
        values: np.ndarray,
        method: Method = "linear",
        source: str = "",
    ) -> None:
        self.axes = tuple(axes)
        if method not in ("linear", "nearest"):
            raise ValueError(f"method is 'linear' or 'nearest', got {method!r}")
        if values.shape != tuple(axis.points for axis in self.axes) or not self.axes:
            raise ValueError(
                f"values of shape {values.shape} do not fit the axes {self.axes}"
            )
        self.values = values
        self.method: Method = method
        self.source = source
        self._holes = bool(np.isnan(values).any())  # nodes where no rule fires

    @overload
    def __call__(self, **inputs: float) -> float: ...
    @overload
    def __call__(self, **inputs: ArrayLike) -> Array: ...
    def __call__(self, **inputs: ArrayLike) -> Array | float:
        """Look the output up for the given inputs."""
        names = [axis.name for axis in self.axes]
        if set(inputs) != set(names):
            raise TypeError(f"the table takes {names}, got {list(inputs)}")
        given = [np.asarray(inputs[name], dtype=np.float64) for name in names]
        shape = np.broadcast_shapes(*(x.shape for x in given))
        places, missing = [], []
        for axis, x in zip(self.axes, given, strict=True):
            at, lost = axis.locate(np.atleast_1d(x))
            places.append(at)
            if lost is not None:
                missing.append(lost)
        nearest = self.method == "nearest"
        out = self._nearest(places) if nearest else self._linear(places)
        for lost in missing:
            out[np.broadcast_to(lost, out.shape)] = np.nan
        return out.reshape(shape) if shape else out.item()

    def _strides(self) -> list[int]:
        """Distance between neighbours along every axis in the flattened values."""
        sizes = self.values.shape
        return [math.prod(sizes[i + 1 :]) for i in range(len(sizes))]

    def _nearest(self, places: list[Array]) -> Array:
        """The value of the node closest to every place."""
        cell: NDArray[np.intp] | int = 0
        for axis, at, stride in zip(self.axes, places, self._strides(), strict=True):
            node = np.rint(at).astype(np.intp)
            if axis.periodic:
                node %= axis.points
            node *= stride
            cell = cell + node
        return self.values.reshape(-1)[cell].astype(np.float64)

    def _linear(self, places: list[Array]) -> Array:
        """The values of the nodes around every place, weighted by closeness."""
        sides = []  # per axis: the node below and the node above, with their weights
        for axis, at, stride in zip(self.axes, places, self._strides(), strict=True):
            below = at.astype(np.intp)
            if axis.periodic:
                share = at - below
                below %= axis.points
                above = (below + 1) % axis.points
            else:
                np.minimum(below, axis.points - 2, out=below)
                share = at - below
                above = below + 1
            sides.append(((below * stride, 1.0 - share), (above * stride, share)))
        cells = self.values.reshape(-1)
        out: Array | None = None
        for corner in itertools.product(*sides):
            cell, weight = corner[0]
            for other, share in corner[1:]:
                cell = cell + other
                weight = weight * share
            term = cells[cell] * weight
            if self._holes:  # a hole must not leak into where it has no weight
                term[np.broadcast_to(weight == 0.0, term.shape)] = 0.0
            out = term if out is None else out + term
        assert out is not None
        return out

    def save(self, path: str | os.PathLike[str], *, compress: bool = False) -> None:
        """Write the table to ``path`` as a NumPy ``.npz`` archive.

        ``compress`` makes the file smaller and slower to write and load.
        """
        meta = {
            "format": _FORMAT,
            "axes": self.axes,
            "method": self.method,
            "source": self.source,
        }
        write = np.savez_compressed if compress else np.savez
        with open(path, "wb") as file:
            write(file, values=self.values, meta=np.array(json.dumps(meta)))

    def __repr__(self) -> str:
        axes = " x ".join(f"{axis.name}[{axis.points}]" for axis in self.axes)
        source = f", from {self.source}" if self.source else ""
        return f"Table({axes}, {self.method}{source})"


def load(path: str | os.PathLike[str]) -> Table:
    """Read a table that :meth:`Table.save` wrote."""
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"][()]))
            values = data["values"]
    except (KeyError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(f"{Path(path).name} is not a fuzlog table") from error
    if meta.get("format") != _FORMAT:
        raise ValueError(
            f"{Path(path).name} has table format {meta.get('format')}, "
            f"this fuzlog reads format {_FORMAT}"
        )
    axes = [Axis(*axis) for axis in meta["axes"]]
    return Table(axes, values, meta["method"], meta["source"])


def _variables(system: object) -> dict[str, Variable]:
    """The input variables of a system or of a proposition, by name."""
    if isinstance(system, Prop):
        return {variable.name: variable for variable in system.variables}
    return dict(getattr(system, "inputs", {}))


def _axes(
    spans: Mapping[str, Span] | None, known: Mapping[str, Variable], points: int
) -> list[Axis]:
    if spans is None:
        if not known:
            raise TypeError("give the axes: the inputs of the system are not known")
        spans = dict.fromkeys(known, points)
    axes = []
    for name, span in spans.items():
        if isinstance(span, int):
            if name not in known:
                raise ValueError(
                    f"{name!r} is not an input of the system: "
                    f"give its range as (low, high, points)"
                )
            variable = known[name]
            axis = Axis(name, *variable.universe, span, variable.periodic)
        else:
            low, high, *rest = span
            axis = Axis(name, float(low), float(high), int(rest[0]) if rest else points)
        if not (math.isfinite(axis.low) and math.isfinite(axis.high)):
            raise ValueError(f"the axis {name!r} needs finite bounds, got {span}")
        if axis.low >= axis.high or axis.points < 2:
            raise ValueError(
                f"the axis {name!r} needs low < high and at least 2 points, got {span}"
            )
        axes.append(axis)
    if not axes:
        raise ValueError("a table needs at least one axis")
    return axes


def compile(
    system: Callable[..., Any],
    axes: Mapping[str, Span] | None = None,
    /,
    *,
    transform: Callable[..., Mapping[str, ArrayLike]] | None = None,
    points: int = 101,
    method: Method = "linear",
    dtype: DTypeLike = np.float64,
) -> Table:
    """Evaluate a system on a grid of its inputs and keep the results in a table.

    The system is a ``Mamdani``, a ``Sugeno``, a proposition or any function of
    keyword arrays. Without ``axes`` every input variable is sampled at ``points``
    values of its universe::

        table = compile(cooler)
        table(temp=27, humid=40)

    ``axes`` name the inputs of the table and give each a number of points, to
    sample the universe of the variable of that name, or a range ``(low, high)``
    or ``(low, high, points)``.

    ``transform`` makes a table over other quantities than the system reads. It is
    called with the inputs of the table as keyword arrays and returns the inputs
    of the system by name. It runs only here, so the table needs no transformation
    when it is used::

        def polar(x, y):
            return {"distance": np.hypot(x, y), "angle": np.arctan2(y, x)}

        table = compile(system, {"x": (-1, 1), "y": (-1, 1)}, transform=polar)
        table(x=0.3, y=-0.2)

    Args:
        system: What to compile.
        axes: The inputs of the table; by default the inputs of the system.
        transform: Maps the inputs of the table to the inputs of the system.
        points: Number of points of every axis that does not give its own.
        method: How the table answers between its nodes: ``linear`` or ``nearest``.
        dtype: Type of the stored numbers; ``np.float32`` halves a large table.
    """
    if transform is not None and axes is None:
        raise TypeError("a transform needs axes: the inputs of the table")
    table_axes = _axes(axes, _variables(system), points)
    grids = [axis.grid() for axis in table_axes]
    shape = tuple(axis.points for axis in table_axes)
    values = np.empty(shape, dtype=dtype)
    cells = values.reshape(-1)

    for start in range(0, cells.size, _CHUNK):
        stop = min(start + _CHUNK, cells.size)
        nodes = np.unravel_index(np.arange(start, stop), shape)
        inputs: Mapping[str, ArrayLike] = {
            axis.name: grid[node]
            for axis, grid, node in zip(table_axes, grids, nodes, strict=True)
        }
        if transform is not None:
            inputs = transform(**inputs)
            if not isinstance(inputs, Mapping):
                raise TypeError(
                    f"a transform returns the inputs of the system by name, "
                    f"got {type(inputs).__name__}"
                )
        cells[start:stop] = np.broadcast_to(system(**inputs), stop - start)
    source = getattr(system, "__name__", None) or repr(system)
    return Table(table_axes, values, method, source)
