# Reference

Everything is reached from the package: `import fuzlog as fz`. The code lives in
three modules: `fuzlog.types` (the data model), `fuzlog.core` (everything that
computes) and `fuzlog.plotting` (matplotlib views, imported on first use).

## Sets

**`MF(fn, label="mf")`** is a fuzzy set. `fn` maps a float64 array of at least one
dimension to degrees in [0, 1] of the same shape, without changing its argument.

| Expression | Result |
|---|---|
| `mf(x)` | degrees of `x`: a float for a scalar, an array otherwise |
| `a & b`, `a \| b`, `~a` | minimum, maximum, complement |
| `a ** p` | degrees to the power `p` |
| `very(a)`, `extremely(a)`, `somewhat(a)` | `a ** 2`, `a ** 3`, `a ** 0.5`; also for propositions |

Shapes, each returning an `MF` ([definitions](shapes.md)):
`triangle(a, b, c)`, `trapezoid(a, b, c, d)`, `rising(a, b)`, `falling(a, b)`,
`rectangle(a, b)`, `singleton(a)`, `piecewise(points)`, `gauss(c, sigma)`,
`gauss2(c1, sigma1, c2, sigma2)`, `bell(c, width, slope)`, `sigmoid(c, slope)`,
`dsigmoid(c1, slope1, c2, slope2)`, `psigmoid(c1, slope1, c2, slope2)`,
`s_shape(a, b)`, `z_shape(a, b)`, `pi_shape(a, b, c, d)`.

## Variables

**`Variable(name, universe, terms=None, *, periodic=False, **more_terms)`** is a
linguistic variable. `universe` is `(low, high)`; terms are sets given as a mapping,
as keyword arguments, or both. On a `periodic` universe `high` is the same point as
`low`, and values are wrapped into `[low, high)`.

| Member | Meaning |
|---|---|
| `name`, `universe`, `terms`, `periodic` | as given; `terms` is a read-only mapping of names to sets |
| `v.term`, `v["term"]` | the proposition "v is term" |
| `v(x)` | degrees of `x` in every term: shape `(len(terms), *x.shape)` |
| `v.fuzzify(x)` | the same by term name, as a dict |
| `v.grid(points=101)` | evenly spaced values that cover the universe |
| `v.peaks(points=2001)` | the middle of the core of every term |
| `v.plot(ax=None)` | draws the terms and returns the Axes |

**`partition(name, universe, cores=None, *, periodic=False, **more_cores)`** builds
a variable whose terms sum to 1. A core is a value or a `(low, high)` range, listed
in increasing order with gaps between neighbours; a list of names instead spreads
the cores evenly. On a `periodic` universe one core may wrap around.

## Propositions and rules

**`Prop`** is a proposition; `variable.term` makes one.

| Expression | Result |
|---|---|
| `p & q`, `p \| q`, `~p`, `p ** n` | and, or, not, hedge |
| `p >> then` | a `Rule` |
| `p(**inputs)` | the degree to which `p` holds, with minimum for and, maximum for or |
| `p.variables` | the variables `p` talks about |
| `p.truth(degree, tnorm, snorm)` | evaluation with other norms, given `degree(variable, term)` |

**`Rule(when, then, weight=1.0)`** is a rule; `when >> then` makes one with
weight 1. `then` is a proposition about the output variable for `Mamdani`, a number
or a function of the inputs for `Sugeno`. `weight` in [0, 1] scales the firing
strength.

## Inference

**`Mamdani(*rules, tnorm=np.minimum, snorm=np.maximum, implication=np.minimum,
aggregation=np.maximum, defuzz=centroid, points=101)`**

| Member | Meaning |
|---|---|
| `system(**inputs)` | the crisp output: a float for scalars, else an array of the broadcast shape; NaN where no rule fires |
| `system.aggregate(**inputs)` | the output set before defuzzification, with a last axis over `system.x` |
| `system.plot(ax=None, **inputs)` | draws the output set for scalar inputs and its crisp value |
| `system.surface(x, y, ax=None, *, points=64, **fixed)` | draws the output over the inputs named `x` and `y` |
| `rules`, `inputs`, `output`, `x` | the rules, the input variables by name, the output variable, its grid |

`implication` must not decrease as the firing strength grows, as every t-norm does.

**`Sugeno(*rules, tnorm=np.minimum, snorm=np.maximum)`** has `system(**inputs)`,
`system.surface(...)`, `rules` and `inputs` with the same meaning. A function in a
conclusion is called as `then(**inputs)`.

### Norms

A norm is a function of two broadcastable arrays of degrees.

| | t-norm (and) | s-norm (or) |
|---|---|---|
| Zadeh | `np.minimum` | `np.maximum` |
| probabilistic | `np.multiply` | `probor`: `a + b - ab` |
| ЕЃukasiewicz | `bounded_diff`: `max(a + b - 1, 0)` | `bounded_sum`: `min(a + b, 1)` |

### Defuzzifiers

`centroid(x, mu)`, `bisector(x, mu)`, `mom(x, mu)`, `som(x, mu)`, `lom(x, mu)` take
a grid `x` and degrees `mu` whose last axis runs over the grid, and return one value
per set: NaN for an empty one. [Meanings](guide.md#defuzzification).

## Plotting

`fuzlog.plotting` holds what the `plot` and `surface` methods call:
`plot_variable(variable, ax=None)`, `plot_inference(system, ax=None, **inputs)` and
`plot_surface(system, x, y, ax=None, *, points=64, **fixed)`. Each returns the Axes.

## Types

`Array` is a float64 NumPy array, `Kernel` the function inside an `MF`, `Norm` a
t-norm or s-norm and `Defuzzifier` a function `(x, mu)`; all four are aliases for
annotations.
