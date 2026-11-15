"""Compiled systems: lookup tables that are built once, saved and loaded.

``compile`` evaluates a system on a grid of its inputs and keeps the results in a
:class:`Table`. The table is called like the system and answers by looking the
inputs up, so its speed does not depend on the rules, and it is saved to disk as
plain numbers: loading one needs neither the rules nor the code that made it.

A system that answers with strings, such as ``Labels``, compiles into a table of
labels, which keeps the distinct strings once and a small number per node.
"""

from __future__ import annotations

import json
import math
import os
import zipfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from fuzlog.math import grid
from fuzlog.rules import Prop
from fuzlog.types import Array
from fuzlog.variables import Variable

type Span = int | np.integer | tuple[float, float] | tuple[float, float, int]
type Method = Literal["linear", "nearest"]

_FORMATS = (1, 2)  # versions of the file layout: 2 adds the labels
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


def _checked(axis: Axis) -> Axis:
    """Return the axis if a table can be laid along it."""
    finite = math.isfinite(axis.low) and math.isfinite(axis.high)
    if not (finite and axis.low < axis.high and axis.points >= 2):
        raise ValueError(
            f"the axis {axis.name!r} needs finite low < high and at least 2 points, "
            f"got ({axis.low:g}, {axis.high:g}, {axis.points})"
        )
    return axis


class Table:
    """A compiled system: its output on a grid of the inputs.

    Calling a table takes the inputs by name, as scalars or as arrays that
    broadcast against each other, like the system it was compiled from. With the
    ``linear`` method the output is interpolated between the surrounding nodes;
    ``nearest`` takes the closest node, which is faster and exact when the inputs
    only take the values of the nodes.

    A table of labels has ``labels``: its values are whole numbers that count
    into them, and it answers with the label of the closest node.

    Tables are made by :func:`compile` and :func:`load`. ``axes`` describe the
    inputs, ``values`` holds the output with one dimension per axis, and ``source``
    names what was compiled.
    """

    __slots__ = ("_holes", "_words", "axes", "labels", "method", "source", "values")

    def __init__(
        self,
        axes: Iterable[Axis],
        values: ArrayLike,
        method: Method = "linear",
        source: str = "",
        labels: Iterable[str] | None = None,
    ) -> None:
        self.axes = tuple(_checked(axis) for axis in axes)
        values = np.asarray(values)
        if method not in ("linear", "nearest"):
            raise ValueError(f"method is 'linear' or 'nearest', got {method!r}")
        if values.shape != tuple(axis.points for axis in self.axes) or not self.axes:
            raise ValueError(
                f"values of shape {values.shape} do not fit the axes {self.axes}"
            )
        self.labels = None if labels is None else tuple(labels)
        if self.labels is not None:
            whole = np.issubdtype(values.dtype, np.integer)
            if method != "nearest" or not whole or values.min() < 0:
                raise ValueError(
                    "a table of labels needs whole numbers and the nearest method"
                )
            if values.max() >= len(self.labels):
                raise ValueError(
                    f"values up to {values.max()} do not fit {len(self.labels)} labels"
                )
        self.values = values
        self.method: Method = method
        self.source = source
        self._holes = bool(np.isnan(values).any())  # nodes where no rule fires
        # The empty label closes the list: it is what the number -1 counts to.
        self._words = None if self.labels is None else np.array([*self.labels, ""])

    def __call__(self, **inputs: ArrayLike) -> Any:
        """Look the output up for the given inputs.

        A table of numbers gives a float for scalars and an array otherwise, NaN
        where an input is NaN. A table of labels gives strings the same way,
        empty where an input is NaN.
        """
        if self._words is not None:
            words = self._words[self.codes(**inputs)]
            return words if isinstance(words, np.ndarray) else str(words)
        places, missing, shape = self._locate(inputs)
        nearest = self.method == "nearest"
        out = (
            self._nearest(places)
            if nearest
            else grid.linear(self.values, places, self._wraps(), self._holes)
        )
        out = out.astype(np.float64, copy=False)
        for lost in missing:
            out[np.broadcast_to(lost, out.shape)] = np.nan
        return out.reshape(shape) if shape else out.item()

    def codes(self, **inputs: ArrayLike) -> Any:
        """Look up the position in ``labels`` instead of the label itself.

        Numbers are cheaper to compare and to store than strings; -1 stands where
        an input is NaN.
        """
        if self.labels is None:
            raise TypeError("only a table of labels has codes")
        places, missing, shape = self._locate(inputs)
        out = self._nearest(places).astype(np.intp)
        for lost in missing:
            out[np.broadcast_to(lost, out.shape)] = -1
        return out.reshape(shape) if shape else out.item()

    def _wraps(self) -> list[bool]:
        return [axis.periodic for axis in self.axes]

    def _nearest(self, places: list[Array]) -> np.ndarray:
        return np.atleast_1d(grid.nearest(self.values, places, self._wraps()))

    def _locate(
        self, inputs: Mapping[str, ArrayLike]
    ) -> tuple[list[Array], list[NDArray[np.bool_]], tuple[int, ...]]:
        """Fractional node indices along every axis, NaN masks, and the shape."""
        names = [axis.name for axis in self.axes]
        if set(inputs) != set(names):
            raise TypeError(f"the table takes {names}, got {list(inputs)}")
        given = [np.asarray(inputs[name], dtype=np.float64) for name in names]
        shape = np.broadcast_shapes(*(x.shape for x in given))
        places, missing = [], []
        for axis, x in zip(self.axes, given, strict=True):
            at = grid.locate(
                np.atleast_1d(x), axis.low, axis.high, axis.points, axis.periodic
            )
            lost = np.isnan(at)
            if lost.any():
                at[lost] = 0.0
                missing.append(lost)
            places.append(at)
        return places, missing, shape

    def save(self, path: str | os.PathLike[str], *, compress: bool = False) -> None:
        """Write the table to ``path`` as a NumPy ``.npz`` archive.

        ``compress`` makes the file smaller and slower to write and load.
        """
        meta = {
            "format": 1 if self.labels is None else 2,
            "axes": self.axes,
            "method": self.method,
            "source": self.source,
            "labels": self.labels,
        }
        write = np.savez_compressed if compress else np.savez
        with open(path, "wb") as file:
            write(file, values=self.values, meta=np.array(json.dumps(meta)))

    def __repr__(self) -> str:
        axes = " x ".join(f"{axis.name}[{axis.points}]" for axis in self.axes)
        source = f", from {self.source}" if self.source else ""
        kind = self.method if self.labels is None else f"{len(self.labels)} labels"
        return f"Table({axes}, {kind}{source})"


def load(path: str | os.PathLike[str]) -> Table:
    """Read a table that :meth:`Table.save` wrote."""
    try:
        data = np.load(path, allow_pickle=False)
        if isinstance(data, np.ndarray):
            raise ValueError("a single array")
        with data:
            meta = json.loads(str(data["meta"][()]))
            values = data["values"]
    except (KeyError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(f"{Path(path).name} is not a fuzlog table") from error
    if meta.get("format") not in _FORMATS:
        raise ValueError(
            f"{Path(path).name} has table format {meta.get('format')}, "
            f"this fuzlog reads formats {_FORMATS}"
        )
    axes = [Axis(*axis) for axis in meta["axes"]]
    return Table(axes, values, meta["method"], meta["source"], meta.get("labels"))


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
        if isinstance(span, int | np.integer):
            if name not in known:
                raise ValueError(
                    f"{name!r} is not an input of the system: "
                    f"give its range as (low, high, points)"
                )
            variable = known[name]
            axis = Axis(name, *variable.universe, int(span), variable.periodic)
        else:
            low, high, *rest = span
            axis = Axis(name, float(low), float(high), int(rest[0]) if rest else points)
        axes.append(_checked(axis))
    if not axes:
        raise ValueError("a table needs at least one axis")
    return axes


def _answer(given: object) -> np.ndarray:
    """What a system gave, as an array: of strings if it gave text of any kind."""
    out = np.asarray(given)
    if out.dtype.kind == "S":
        return np.char.decode(out)
    if out.dtype.kind == "O" and all(isinstance(item, str) for item in out.flat):
        return out.astype(np.str_)
    return out


def compile(
    system: Callable[..., Any],
    axes: Mapping[str, Span] | None = None,
    /,
    *,
    transform: Callable[..., Mapping[str, ArrayLike]] | None = None,
    points: int = 101,
    method: Method | None = None,
    dtype: DTypeLike | None = None,
) -> Table:
    """Evaluate a system on a grid of its inputs and keep the results in a table.

    The system is anything that is called with keyword arrays: a ``Mamdani``, a
    ``Sugeno``, a ``Labels``, a proposition or a function of your own. Without
    ``axes`` every input variable is sampled at ``points`` values of its universe::

        table = compile(cooler)
        table(temp=27, humid=40)

    A system that answers with strings gives a table of labels: it keeps the
    distinct strings once and a small whole number per node.

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
        method: How the table answers between its nodes: ``linear``, the default,
            or ``nearest``. A table of labels is always ``nearest``.
        dtype: Type of the stored numbers. By default ``np.float64``, of which
            ``np.float32`` takes half, and for labels the smallest type that
            counts them all.
    """
    if transform is not None and axes is None:
        raise TypeError("a transform needs axes: the inputs of the table")
    table_axes = _axes(axes, _variables(system), points)
    grids = [axis.grid() for axis in table_axes]
    shape = tuple(axis.points for axis in table_axes)
    size = math.prod(shape)
    cells: np.ndarray | None = None  # made once the first results tell their kind
    worded = False
    known: dict[str, int] = {}  # every label and its number, in order of appearance

    for start in range(0, size, _CHUNK):
        stop = min(start + _CHUNK, size)
        nodes = np.unravel_index(np.arange(start, stop), shape)
        inputs: Mapping[str, ArrayLike] = {
            axis.name: values[node]
            for axis, values, node in zip(table_axes, grids, nodes, strict=True)
        }
        if transform is not None:
            inputs = transform(**inputs)
            if not isinstance(inputs, Mapping):
                raise TypeError(
                    f"a transform returns the inputs of the system by name, "
                    f"got {type(inputs).__name__}"
                )
        out = _answer(system(**inputs))
        strings = out.dtype.kind == "U"
        if cells is None:
            worded = strings
            if worded and method == "linear":
                raise ValueError("labels cannot be interpolated: use method='nearest'")
            kind = np.uint32 if worded else np.float64 if dtype is None else dtype
            cells = np.empty(size, dtype=kind)
        elif strings != worded:
            raise TypeError("a system gives labels or numbers, not both")
        out = np.broadcast_to(out, stop - start)
        if worded:
            found, where = np.unique(out, return_inverse=True)
            numbers = [known.setdefault(str(label), len(known)) for label in found]
            out = np.array(numbers, dtype=np.uint32)[where]
        cells[start:stop] = out

    assert cells is not None
    source = getattr(system, "__name__", None) or repr(system)
    if not worded:
        return Table(table_axes, cells.reshape(shape), method or "linear", source)
    kind = np.min_scalar_type(len(known) - 1) if dtype is None else np.dtype(dtype)
    if not np.issubdtype(kind, np.integer) or len(known) - 1 > np.iinfo(kind).max:
        raise ValueError(f"{len(known)} labels cannot be counted in {kind}")
    values = cells.astype(kind).reshape(shape)
    return Table(table_axes, values, "nearest", source, known)
