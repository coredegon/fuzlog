# fuzlog

Fuzzy logic for Python: sets, variables, rules, Mamdani and Sugeno inference.
NumPy inside, typed, Python 3.12+.

```sh
pip install fuzlog
```

```python
import fuzlog as fz

service = fz.partition("service", (0, 10), ["poor", "good", "excellent"])
food = fz.Variable("food", (0, 10), rancid=fz.z_shape(1, 4), tasty=fz.s_shape(6, 9))
tip = fz.partition("tip", (0, 30), cheap=5, average=15, generous=25)

tipper = fz.Mamdani(
    service.poor | food.rancid >> tip.cheap,
    service.good >> tip.average,
    service.excellent | food.tasty >> tip.generous,
)
print(round(tipper(service=3, food=8), 2))  # 17.03
print(tipper(service=[1, 5, 9], food=[2, 5, 8]).round(2).tolist())  # [8.2, 15.0, 21.8]

fz.plot(tipper, service=3, food=8)
fz.surface(tipper, "service", "food")
```

![The output set inferred for service 3 and food 8, and the tip over both inputs](https://raw.githubusercontent.com/coredegon/fuzlog/main/docs/img/tipper.png)

## Inside

| | |
|---|---|
| **Sets** | 16 shapes, from `triangle` to `pi_shape`, or any function; combined with `&`, `\|`, `~`, `**` |
| **Variables** | `Variable` names the terms of a quantity; `partition` builds terms that sum to 1, also around a circle |
| **Rules** | `temp.hot & ~humid.dry >> fan.fast`, weights, the hedges `very`, `extremely`, `somewhat` |
| **Systems** | `Mamdani` with any norms and defuzzifier; `Sugeno` with numbers or functions |
| **Features** | `Labels` puts inputs in words, `tabulate` turns a system into a lookup table that saves to disk, `plot` and `surface` draw |
| **Math** | `fuzlog.math`: curves, norms, defuzzifiers and grid lookup as plain NumPy functions |

A scalar gives a float, arrays give arrays, and inputs broadcast against each other.

## Documentation

- [Guide](https://github.com/coredegon/fuzlog/blob/main/docs/guide.md): every part, with pictures
- [Shapes](https://github.com/coredegon/fuzlog/blob/main/docs/shapes.md): the sixteen membership functions
- [Reference](https://github.com/coredegon/fuzlog/blob/main/docs/reference.md): every public name on one page

## Development

```sh
uv sync                 # environment with the dev tools
uv run pytest           # tests, including every example in the documentation
uv run ruff check       # lint; `uv run ruff format` formats
uv run ty check         # types
```

MIT license.
