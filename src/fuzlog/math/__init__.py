"""The mathematics of fuzzy logic as plain NumPy functions.

Nothing here knows about sets, variables or rules, so every part works on its own:

- `curves`: the shapes of membership functions;
- `norms`: how degrees combine;
- `defuzz`: how a fuzzy set becomes a number;
- `grid`: lookup in a table of values.
"""

from fuzlog.math import curves, defuzz, grid, norms

__all__ = ["curves", "defuzz", "grid", "norms"]
