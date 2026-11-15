"""Tables: a system evaluated once on a grid, then answered by lookup."""

from __future__ import annotations

import json
import math
import os
import zipfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from fuzlog.math import grid
from fuzlog.types import Array
from fuzlog.variables import Variable

type Span = int | np.integer | tuple[float, float] | tuple[float, float, int]
type Method = Literal["linear", "nearest"]

_FORMATS = (1, 2)  # versions of the file layout: 2 adds the labels
_BLOCK = 1 << 14  # grid nodes evaluated in one call of the system


class Axis(NamedTuple):
    """One input of a table: `points` evenly spaced nodes from `low` to `high`.

    A `periodic` axis is a circle, so its last node lies one step before `high`.
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
    """The output of a system on a grid of its inputs; `tabulate` makes one.

    ```python
    table = fz.tabulate(cooler, points=201)
    table(temp=27.3, humid=41)     # looked up, not inferred
    table.save("cooler.npz")
    table = fz.Table.load("cooler.npz")
    ```

    `axes` describe the inputs, `values` holds one number per node, `method` is
    how it answers between nodes and `source` names what was tabulated. A table
    of `labels` holds their positions and answers with the label.
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
        self.values = values = np.asarray(values)
        self.method: Method = method
        self.source = source
        self.labels = None if labels is None else tuple(labels)
        if method not in ("linear", "nearest"):
            raise ValueError(f"method is 'linear' or 'nearest', got {method!r}")
        if values.shape != tuple(axis.points for axis in self.axes) or not self.axes:
            raise ValueError(
                f"values of shape {values.shape} do not fit the axes {self.axes}"
            )
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
        self._holes = bool(np.isnan(values).any())  # nodes where no rule fires
        # The empty label closes the list: it is what the number -1 counts to.
        self._words = None if self.labels is None else np.array([*self.labels, ""])

    def __call__(self, **inputs: ArrayLike) -> Any:
        """Look the output up for inputs given by name.

        Inputs outside an axis take its nearest end, or wrap around if it is
        periodic. A NaN input gives NaN, or an empty label.
        """
        if self._words is not None:
            words = self._words[self.codes(**inputs)]
            return words if isinstance(words, np.ndarray) else str(words)
        return self._read(inputs, self.method, np.float64, np.nan)

    def codes(self, **inputs: ArrayLike) -> Any:
        """Look up the position in `labels` instead of the label; -1 for a NaN input."""
        if self.labels is None:
            raise TypeError("only a table of labels has codes")
        return self._read(inputs, "nearest", np.intp, -1)

    def _read(
        self, inputs: Mapping[str, ArrayLike], method: Method, dtype: type, lost: float
    ) -> Any:
        """Look the inputs up; `lost` stands where one of them is NaN."""
        names = [axis.name for axis in self.axes]
        if set(inputs) != set(names):
            raise TypeError(f"the table takes {names}, got {list(inputs)}")
        given = [np.asarray(inputs[name], dtype=np.float64) for name in names]
        shape = np.broadcast_shapes(*(x.shape for x in given))
        places, missing = [], []
        for axis, x in zip(self.axes, given, strict=True):
            at = grid.locate(np.atleast_1d(x), *axis[1:])
            nan = np.isnan(at)
            if nan.any():
                at[nan] = 0.0
                missing.append(nan)
            places.append(at)
        wraps = [axis.periodic for axis in self.axes]
        if method == "nearest":
            out = grid.nearest(self.values, places, wraps)
        else:
            out = grid.linear(self.values, places, wraps, self._holes)
        out = out.astype(dtype, copy=False)
        for nan in missing:
            out[np.broadcast_to(nan, out.shape)] = lost
        return out.reshape(shape) if shape else out.item()

    def save(self, path: str | os.PathLike[str], *, compress: bool = False) -> None:
        """Write the table to `path` as a NumPy `.npz` archive.

        The file holds the numbers and the axes, not the rules that made them.
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

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Table:
        """Read a table that `save` wrote."""
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
        return cls(axes, values, meta["method"], meta["source"], meta.get("labels"))

    def __repr__(self) -> str:
        axes = " x ".join(f"{axis.name}[{axis.points}]" for axis in self.axes)
        source = f", from {self.source}" if self.source else ""
        kind = self.method if self.labels is None else f"{len(self.labels)} labels"
        return f"Table({axes}, {kind}{source})"


def tabulate(
    system: Callable[..., Any],
    axes: Mapping[str, Span] | None = None,
    /,
    *,
    points: int = 101,
    method: Method | None = None,
    dtype: DTypeLike | None = None,
) -> Table:
    """Evaluate a system on a grid of its inputs and keep the results in a `Table`.

    ```python
    fz.tabulate(cooler)                              # 101 points per input
    fz.tabulate(cooler, {"temp": 41, "humid": (20, 80, 61)})
    fz.tabulate(lambda x, y: np.hypot(x, y), {"x": (-1, 1), "y": (-1, 1)})
    ```

    The system is anything called with keyword arrays: a `Mamdani`, a `Sugeno`,
    a proposition, a `Labels` or a function of your own. One that answers with
    strings gives a table of labels.

    Args:
        system: What to tabulate.
        axes: The inputs of the table by name: a number of points over the
            universe of that variable, `(low, high)` or `(low, high, points)`.
            By default every input variable of the system.
        points: Number of points of every axis that does not give its own.
        method: How the table answers between its nodes: `"linear"`, the
            default, or `"nearest"`, as a table of labels always does.
        dtype: Type of the stored values: by default `np.float64`, and for
            labels the smallest whole type that counts them.
    """
    table_axes = _axes(axes, getattr(system, "inputs", {}), points)
    nodes = [axis.grid() for axis in table_axes]
    shape = tuple(axis.points for axis in table_axes)
    size = math.prod(shape)
    cells: np.ndarray | None = None  # made once the first results tell their kind
    worded = False
    known: dict[str, int] = {}  # every label and its number

    for start in range(0, size, _BLOCK):
        stop = min(start + _BLOCK, size)
        at = np.unravel_index(np.arange(start, stop), shape)
        inputs = {a.name: n[i] for a, n, i in zip(table_axes, nodes, at, strict=True)}
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
    return Table(
        table_axes, cells.astype(kind).reshape(shape), "nearest", source, known
    )


def _axes(
    spans: Mapping[str, Span] | None, known: Mapping[str, Variable], points: int
) -> list[Axis]:
    """The axes of a table, from what was asked for and the variables of the system."""
    if spans is None:
        if not known:
            raise TypeError("give the axes: the inputs of the system are not known")
        spans = dict.fromkeys(known, points)
    axes = []
    for name, span in spans.items():
        if not isinstance(span, int | np.integer):
            low, high, *rest = span
            axes.append(
                Axis(name, float(low), float(high), int(rest[0]) if rest else points)
            )
        elif name in known:
            variable = known[name]
            axes.append(Axis(name, *variable.universe, int(span), variable.periodic))
        else:
            raise ValueError(
                f"{name!r} is not an input of the system: "
                f"give its range as (low, high, points)"
            )
    if not axes:
        raise ValueError("a table needs at least one axis")
    return [_checked(axis) for axis in axes]


def _answer(given: object) -> np.ndarray:
    """What a system gave, as an array: of strings if it gave text of any kind."""
    out = np.asarray(given)
    if out.dtype.kind == "S":
        return np.char.decode(out)
    if out.dtype.kind == "O" and all(isinstance(item, str) for item in out.flat):
        return out.astype(np.str_)
    return out
