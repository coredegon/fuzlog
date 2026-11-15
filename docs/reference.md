# Reference

Everything is reached from the package: `import fuzlog as fz`.

| Module | Holds |
|---|---|
| [`fuzlog.sets`](#sets) | `Set` and the shapes |
| [`fuzlog.variables`](#variables) | `Variable`, `partition` |
| [`fuzlog.rules`](#rules) | `Prop`, `Rule`, the hedges |
| [`fuzlog.systems`](#systems) | `Mamdani`, `Sugeno` |
| [`fuzlog.features`](#features) | `labels`, `tables`, `plots` |
| [`fuzlog.math`](#math) | `curves`, `norms`, `defuzz`, `grid` |

## Sets

**`Set(curve, name="set")`**: `curve` maps a float64 array to degrees of the same
shape.

| Expression | Result |
|---|---|
| `s(x)` | degrees of `x`: a float for a scalar, else an array |
| `a & b`, `a \| b`, `~a`, `a ** p` | minimum, maximum, complement, power |
| `s.curve`, `s.name` | as given |

Shapes ([definitions](shapes.md)): `triangle(a, b, c)`, `trapezoid(a, b, c, d)`,
`rising(a, b)`, `falling(a, b)`, `rectangle(a, b)`, `singleton(a)`,
`piecewise(points)`, `gauss(c, sigma)`, `gauss2(c1, sigma1, c2, sigma2)`,
`bell(c, width, slope)`, `sigmoid(c, slope)`, `dsigmoid(c1, slope1, c2, slope2)`,
`psigmoid(c1, slope1, c2, slope2)`, `s_shape(a, b)`, `z_shape(a, b)`,
`pi_shape(a, b, c, d)`.

## Variables

**`Variable(name, universe, terms=None, *, periodic=False, **more_terms)`**

| Member | Meaning |
|---|---|
| `v.term`, `v["term"]` | the proposition "v is term" |
| `v(x)` | degrees of `x` in every term: shape `(len(terms), *x.shape)` |
| `v.grid(points=101)` | evenly spaced values over the universe |
| `v.wrap(x)` | `x` brought into `[low, high)` if the variable is periodic |
| `name`, `universe`, `terms`, `periodic` | as given; `terms` maps names to sets |

**`partition(name, universe, cores=None, *, periodic=False, **more_cores)`**
returns a `Variable` whose terms sum to 1. A core is a value or a `(low, high)`
range, in increasing order with gaps between neighbours; a list of names spreads
the cores evenly.

## Rules

**`Prop`**: a proposition; `variable.term` makes one.

| Expression | Result |
|---|---|
| `p & q`, `p \| q`, `~p`, `p ** n` | and, or, not, hedge |
| `very(p)`, `extremely(p)`, `somewhat(p)` | `p ** 2`, `p ** 3`, `p ** 0.5`; also for sets |
| `p >> then` | a `Rule` |
| `p(**inputs)` | the degree to which `p` holds: minimum for and, maximum for or |
| `p.inputs` | the variables `p` talks about, by name |
| `p.truth(inputs, tnorm, snorm)` | the same with other norms, given `rules.Inputs(values)` |

**`Rule(when, then, weight=1.0)`**: `when >> then` with a weight in [0, 1]. `then`
is a proposition for `Mamdani`, a number or a function for `Sugeno`.
`rule.fire(inputs, tnorm, snorm)` is its firing strength.

## Systems

**`Mamdani(*rules, tnorm=np.minimum, snorm=np.maximum, implication=np.minimum,
aggregation=np.maximum, defuzz=centroid, points=101)`**

| Member | Meaning |
|---|---|
| `system(**inputs)` | the crisp output: a float for scalars, else an array of the broadcast shape; NaN where no rule fires |
| `system.aggregate(**inputs)` | the output set before defuzzification: a last axis over `system.x` |
| `rules`, `inputs`, `output`, `x` | the rules, the input variables by name, the output variable, its grid |

**`Sugeno(*rules, tnorm=np.minimum, snorm=np.maximum)`** has `system(**inputs)`,
`rules` and `inputs`. A function in a conclusion receives the inputs it names, or
all of them if it takes `**kwargs`.

Norms: `probor`, `bounded_sum`, `bounded_diff`, and NumPy's `minimum`, `maximum`,
`multiply`. Defuzzifiers: `centroid`, `bisector`, `mom`, `som`, `lom`.

## Features

**`Labels(*variables, sep="_")`**: `labels(**inputs)` is the strongest term of
every variable, joined by `sep`: a string for scalars, else an array of strings.

**`tabulate(system, axes=None, *, points=101, method=None, dtype=None)`** returns
a `Table` of anything called with keyword arrays.

| Member of `Table` | Meaning |
|---|---|
| `table(**inputs)` | the looked-up output. Inputs outside an axis take its nearest end, or wrap around if it is periodic; NaN gives NaN, or an empty label |
| `table.codes(**inputs)` | for a table of labels, the position in `labels`; -1 for NaN |
| `table.save(path, *, compress=False)` | writes a NumPy `.npz` archive |
| `Table.load(path)` | reads one |
| `axes`, `values`, `method`, `source`, `labels` | the axes, one value per node, `"linear"` or `"nearest"`, what was tabulated, the strings of a table of labels |

**`plot(variable, ax=None)`** draws the terms; **`plot(system, ax=None,
**inputs)`** the output set a `Mamdani` infers and its crisp value;
**`surface(system, x, y, ax=None, *, points=64, **fixed)`** the output over two
inputs. Each returns the Axes. Matplotlib is imported by the first plot.

## Math

Plain NumPy functions; nothing here knows about the classes above.

| Module | Functions |
|---|---|
| `math.curves` | the sixteen shapes: each returns a function of a float64 array |
| `math.norms` | `probor(a, b)`, `bounded_sum(a, b)`, `bounded_diff(a, b)` |
| `math.defuzz` | `centroid(x, mu)`, `bisector`, `mom`, `som`, `lom`: one value per set along the last axis of `mu`, NaN for an empty set |
| `math.grid` | `locate(x, low, high, points, periodic)`, `nearest(values, places, periodic)`, `linear(values, places, periodic, holes=False)` |

`fuzlog.types` has the aliases the library is annotated with: `Array`, `Curve`,
`Norm`, `Defuzzifier`.
