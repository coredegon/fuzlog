# Guide

```python
import numpy as np
import fuzlog as fz
```

## Sets

A fuzzy set maps a value to its degree of membership, from 0 to 1. A set is built
from a [shape](shapes.md) and called like a function: a scalar gives a float, an
array gives an array of the same shape.

```python
warm = fz.triangle(10, 20, 30)
print(warm(15))  # 0.5
print(warm([5, 15, 20, 27.5]).tolist())  # [0.0, 0.5, 1.0, 0.25]
```

Sets combine pointwise: `&` is the minimum, `|` the maximum, `~` the complement and
`**` a power. The hedges `very`, `extremely` and `somewhat` are the powers 2, 3
and 0.5.

```python
hot = fz.rising(20, 40)
print((warm | hot)(25), (warm & hot)(25), (~hot)(25), fz.very(hot)(30))
# 0.5 0.25 0.75 0.25
print(warm | hot)  # (triangle(10, 20, 30) | rising(20, 40))
```

Any function of an array makes a set. It receives a float64 array and returns the
degrees:

```python
spike = fz.MF(lambda x: np.exp(-np.abs(x - 20)), "spike")
print(spike(20))  # 1.0
```

## Variables

A linguistic variable is a quantity, the range of its values and the terms that
describe them.

```python
temp = fz.Variable(
    "temp",
    (0, 40),
    cold=fz.falling(10, 20),
    warm=fz.triangle(10, 20, 30),
    hot=fz.rising(20, 30),
)
print(temp.fuzzify(17.5))  # {'cold': 0.25, 'warm': 0.75, 'hot': 0.0}
```

Calling the variable gives the degrees as one array with a row per term, which
unpacks into the terms:

```python
cold, warm, hot = temp([5.0, 17.5, 30.0])
print(warm.tolist())  # [0.0, 0.75, 0.0]
```

Term names that are not Python identifiers go in a mapping:
`fz.Variable("light", (0, 1), {"very dim": fz.falling(0.1, 0.3)})`.

![The terms of a variable](img/variable.png)

### Partitions

`partition` builds the terms from their cores, the values that belong to a term
completely. Between neighbouring cores membership passes linearly from one term to
the next, so the degrees always sum to 1.

```python
print(fz.partition("temp", (0, 40), cold=10, warm=20, hot=30))
# Variable('temp', (0, 40), {'cold': falling(10, 20), 'warm': triangle(10, 20, 30), 'hot': rising(20, 30)})
print(fz.partition("age", (0, 100), young=(0, 25), adult=(35, 55), old=(70, 100)))
# Variable('age', (0, 100), {'young': falling(25, 35), 'adult': trapezoid(25, 35, 55, 70), 'old': rising(55, 70)})

# a list of names spreads the cores evenly
print(fz.partition("level", (0, 1), ["low", "medium", "high"]))
# Variable('level', (0, 1), {'low': falling(0, 0.5), 'medium': triangle(0, 0.5, 1), 'high': rising(0.5, 1)})
```

A `periodic` universe is a circle: the last term blends back into the first, a core
may wrap around, and values are wrapped into the universe before the terms see them.

```python
wind = fz.partition("wind", (0, 360), ["north", "east", "south", "west"], periodic=True)
print(wind.fuzzify(315))  # {'north': 0.5, 'east': 0.0, 'south': 0.0, 'west': 0.5}
print(wind.fuzzify(-45) == wind.fuzzify(315))  # True
hue = fz.partition("hue", (0, 360), red=(350, 10), green=120, blue=240, periodic=True)
print(hue.fuzzify(355))  # {'red': 1.0, 'green': 0.0, 'blue': 0.0}
```

`peaks` gives the most typical value of every term, the middle of its core:

```python
print(temp.peaks().round(2).tolist())  # [5.0, 20.0, 35.0]
```

## Propositions and rules

`temp.hot`, or `temp["hot"]` for any name, is the proposition "temp is hot".
Propositions combine with `&` (and), `|` (or), `~` (not) and the hedges, and calling
one tells to what degree it holds for the given inputs.

```python
humid = fz.partition("humid", (0, 100), ["dry", "comfortable", "wet"])
muggy = temp.hot & humid.wet
print(muggy)  # temp is hot and humid is wet
print(muggy(temp=28, humid=90))  # 0.8
print(muggy(temp=[22, 28, 35], humid=90).tolist())  # [0.2, 0.8, 0.8]
```

`>>` turns a proposition into a rule. Python binds `>>` tighter than `&` and `|`,
and fuzlog accounts for it, so the condition needs no brackets. `Rule` adds a weight
that scales how strongly the rule fires.

```python
fan = fz.partition("fan", (0, 100), ["slow", "medium", "fast"])
print(temp.hot & humid.wet >> fan.fast)
# if temp is hot and humid is wet then fan is fast
print(fz.Rule(temp.cold, fan.slow, weight=0.5))
# if temp is cold then fan is slow [0.5]
```

Python's own `and`, `or` and `not` cannot be redefined, so using them on
propositions or sets raises a `TypeError` instead of silently dropping a condition.

## Mamdani inference

Every rule concludes "output is term" about the same output variable. The system
fires the rules, shapes the concluded sets by the firing strengths, merges them and
reduces the result to one number.

```python
cooler = fz.Mamdani(
    temp.cold >> fan.slow,
    temp.warm >> fan.medium,
    temp.hot | humid.wet >> fan.fast,
)
print(cooler)  # Mamdani(3 rules: temp, humid -> fan)
print(round(cooler(temp=27, humid=40), 2))  # 62.44
```

Inputs are named after the variables and broadcast against each other, so a column
of temperatures and a row of humidities give a table of fan speeds. Where no rule
fires the result is NaN.

```python
print(cooler(temp=[[15], [27]], humid=[20, 60, 100]).round(2).tolist())
# [[44.05, 45.22, 54.63], [62.44, 62.44, 64.27]]
print(fz.Mamdani(temp.cold >> fan.slow)(temp=30))  # nan
```

Each step can be replaced. A norm is any function of two arrays of degrees; NumPy's
`minimum`, `multiply` and `maximum` are norms already, and fuzlog adds `probor`,
`bounded_sum` and `bounded_diff`.

| Argument | Default | Meaning |
|---|---|---|
| `tnorm` | `np.minimum` | how `&` combines degrees |
| `snorm` | `np.maximum` | how `\|` combines degrees |
| `implication` | `np.minimum` | how a firing strength shapes the concluded set |
| `aggregation` | `np.maximum` | how the shaped sets merge |
| `defuzz` | `fz.centroid` | how the merged set becomes a number |
| `points` | `101` | size of the grid the output universe is sampled on |

```python
larsen = fz.Mamdani(
    *cooler.rules, tnorm=np.multiply, snorm=fz.probor, implication=np.multiply
)
print(round(larsen(temp=27, humid=40), 2))  # 67.23
```

`aggregate` returns the merged set itself, sampled on the grid `cooler.x`:

```python
print(cooler.aggregate(temp=27, humid=40).shape)  # (101,)
```

### Defuzzification

A defuzzifier is a function `(x, mu)` of a grid and of degrees whose last axis runs
over that grid.

| Function | Result |
|---|---|
| `centroid` | centre of gravity of the area under the set |
| `bisector` | value that splits the area in two halves |
| `mom` | mean of the values where the set is highest |
| `som` | smallest value where the set is highest |
| `lom` | largest value where the set is highest |

```python
for defuzz in (fz.centroid, fz.bisector, fz.mom, fz.som, fz.lom):
    system = fz.Mamdani(*cooler.rules, defuzz=defuzz)
    print(defuzz.__name__, round(system(temp=27, humid=40), 2))
# centroid 62.44
# bisector 70.0
# mom 92.5
# som 85.0
# lom 100.0
```

`centroid` treats the set as straight lines between its samples, so it is exact
when the grid holds every corner of the set and close otherwise. The other four
answer with a grid value.

## Sugeno inference

A Sugeno rule concludes a number, or a function that is called with the inputs as
keyword arguments. The output is the average of the conclusions, weighted by the
firing strengths.

```python
heater = fz.Sugeno(
    temp.cold >> 10,
    temp.warm >> 50,
    temp.hot >> (lambda temp, **others: 60 + temp),
)
print(heater(temp=17.5))  # 40.0
print(heater(temp=[5, 17.5, 28]).tolist())  # [10.0, 40.0, 80.4]
```

## Labels

`Labels` describes inputs in words: the strongest term of every variable, joined
by `sep` in the order of the variables.

```python
words = fz.Labels(temp, humid)
print(words(temp=27, humid=40))  # hot_comfortable
print(words(temp=[5, 18, 33], humid=90).tolist())
# ['cold_wet', 'warm_wet', 'hot_wet']
print(fz.Labels(humid, temp, sep=" and ")(temp=27, humid=40))  # comfortable and hot
```

Of terms that hold equally the first one wins. The label is empty where an input
is NaN or belongs to no term of its variable.

## Tables

`compile` evaluates a system on a grid of its inputs and returns a `Table`. The
table is called like the system, but answers by looking the inputs up, so its speed
does not depend on the number of rules. Between the nodes of the grid it
interpolates, so the answer is close to the system's and gets closer with more
`points`.

```python
table = fz.compile(cooler, points=201)
print(table)
# Table(temp[201] x humid[201], linear, from Mamdani(3 rules: temp, humid -> fan))
print(round(cooler(temp=27.3, humid=41), 2))  # 63.74
print(round(table(temp=27.3, humid=41), 2))  # 63.75
```

A table is saved as numbers, so loading it needs neither the rules nor the code
that built them:

```python
table.save("cooler.npz")
table = fz.load("cooler.npz")
print(round(table(temp=27.3, humid=41), 2))  # 63.75
```

By default every input variable is sampled at `points` values of its universe. A
mapping of axes sets them one by one: a number of points for a variable, or a range
`(low, high)` or `(low, high, points)`.

`method="nearest"` answers with the closest node instead of interpolating. It is
faster, and exact when the inputs only take the values of the nodes, like the
levels of an 8-bit channel on the axis `(0, 255, 256)`.

### Transforming the inputs

A `transform` makes a table over other quantities than the system reads. It
receives the inputs of the table as arrays and returns the inputs of the system by
name. It runs only while the table is compiled, so its cost is paid once.

```python
def metric(fahrenheit, moisture):
    return {"temp": (fahrenheit - 32) / 1.8, "humid": 100 * moisture}


axes = {"fahrenheit": (32, 104, 289), "moisture": (0, 1, 201)}
imperial = fz.compile(cooler, axes, transform=metric)
print(round(imperial(fahrenheit=81.14, moisture=0.41), 2))  # 63.75
```

### Tables of labels

A system that answers with strings, such as `Labels` or a function of your own,
compiles into a table of labels. It keeps the distinct strings once, in
`table.labels`, and a small whole number per node; both are saved together.

```python
named = fz.compile(words)
print(named)  # Table(temp[101] x humid[101], 9 labels, from Labels(temp, humid))
print(named(temp=27, humid=40))  # hot_comfortable
```

`codes` looks up the position of the label in `table.labels` instead of the
string, which is cheaper to compare and to store for large arrays:

```python
code = named.codes(temp=27, humid=40)
print(named.labels[code])  # hot_comfortable
```

Labels cannot be interpolated, so such a table always answers with the closest
node.

A table holds one number per node: `points` to the power of the number of inputs.
`dtype=np.float32` halves it.

## Plots

Plots need matplotlib: `pip install "fuzlog[plot]"`. Each call draws on the Axes it is
given, or on a new figure, and returns the Axes, so it works the same in a script
and in a notebook.

```python
import matplotlib.pyplot as plt

temp.plot()  # the terms of a variable
cooler.plot(temp=27, humid=40)  # the inferred output set and its crisp value
cooler.surface("temp", "humid")  # the output over two inputs

fig, ax = plt.subplots()
cooler.surface("temp", "humid", ax)  # on an Axes of your own
plt.show()
```

![The fan speed inferred for 27 degrees and 40 percent, and over both inputs](img/cooler.png)

Plots follow the active matplotlib style. Terms are coloured from a palette of
eight colour-blind-safe hues; a variable with more terms is drawn in one hue, with
the term names above the peaks.
