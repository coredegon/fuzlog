import threading

import numpy as np
import pytest

import fuzlog as fz

TEMP = fz.partition("temp", (0, 40), cold=10, warm=20, hot=30)
HUMID = fz.partition("humid", (0, 100), ["dry", "comfortable", "wet"])
FAN = fz.partition("fan", (0, 100), ["slow", "medium", "fast"])
COOLER = fz.Mamdani(
    TEMP.cold >> FAN.slow,
    TEMP.warm >> FAN.medium,
    TEMP.hot | HUMID.wet >> FAN.fast,
)
HEATER = fz.Sugeno(TEMP.cold >> 10, TEMP.warm >> 50, TEMP.hot >> 90)
WIND = fz.partition("wind", (0, 360), ["north", "east", "south", "west"], periodic=True)


def plane(x, y):
    return 2.0 * x - 3.0 * y + 1.0


def test_table_equals_the_system_at_its_nodes():
    table = fz.compile(COOLER, points=21)
    temp, humid = np.meshgrid(TEMP.grid(21), HUMID.grid(21), indexing="ij")
    assert table.values.shape == (21, 21)
    np.testing.assert_allclose(table.values, COOLER(temp=temp, humid=humid))
    np.testing.assert_allclose(table(temp=temp, humid=humid), table.values)


def test_table_stays_close_to_the_system_between_nodes():
    rng = np.random.default_rng(0)
    temp, humid = rng.uniform(0, 40, 2000), rng.uniform(0, 100, 2000)
    exact = COOLER(temp=temp, humid=humid)
    coarse = np.abs(fz.compile(COOLER, points=51)(temp=temp, humid=humid) - exact)
    fine = np.abs(fz.compile(COOLER, points=401)(temp=temp, humid=humid) - exact)
    assert fine.mean() < coarse.mean() < 0.2
    assert fine.max() < 0.5


def test_linear_method_is_exact_for_a_linear_function():
    table = fz.compile(plane, {"x": (0, 1, 5), "y": (-2, 2, 9)})
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 1, 100), rng.uniform(-2, 2, 100)
    np.testing.assert_allclose(table(x=x, y=y), plane(x, y), atol=1e-12)


def test_nearest_method_answers_with_the_closest_node():
    table = fz.compile(plane, {"x": (0, 4, 5), "y": (0, 4, 5)}, method="nearest")
    assert table(x=1.4, y=2.6) == plane(1.0, 3.0)
    assert table(x=3, y=0) == plane(3.0, 0.0)


def test_table_is_called_like_the_system():
    table = fz.compile(HEATER, points=41)
    assert isinstance(table(temp=17.5), float)
    assert table(temp=17.5) == pytest.approx(HEATER(temp=17.5))
    assert table(temp=[[5.0], [17.5]]).shape == (2, 1)
    wide = fz.compile(COOLER, points=11)(temp=[[15], [27]], humid=[20, 60, 100])
    assert wide.shape == (2, 3)
    with pytest.raises(TypeError, match="takes"):
        table(temp=1, humid=2)
    with pytest.raises(TypeError, match="takes"):
        table()


def test_values_outside_an_axis_take_its_end_and_nan_stays_nan():
    table = fz.compile(plane, {"x": (0, 1, 3), "y": (0, 1, 3)})
    assert table(x=-5, y=0.5) == table(x=0, y=0.5)
    assert table(x=np.inf, y=0.5) == table(x=1, y=0.5)
    out = table(x=[0.5, np.nan], y=[[0.5], [np.nan]])
    assert np.isnan(out).tolist() == [[False, True], [True, True]]


def test_periodic_axis_wraps_around():
    table = fz.compile(WIND.north, points=8)
    assert table.axes[0].periodic
    angles = np.array([0.0, 30.0, 350.0, 359.9, 720.0 + 45.0, -45.0])
    np.testing.assert_allclose(table(wind=angles), WIND.north(wind=angles), atol=1e-9)
    nearest = fz.compile(WIND.north, points=8, method="nearest")
    assert nearest(wind=359.0) == nearest(wind=0.0) == 1.0


def test_holes_do_not_leak_to_the_nodes_next_to_them():
    system = fz.Mamdani(TEMP.cold >> FAN.slow)  # nothing fires from 20 degrees on
    table = fz.compile(system, points=5)
    assert np.isnan(table.values).tolist() == [False, False, True, True, True]
    assert table(temp=10.0) == pytest.approx(system(temp=10.0))
    assert np.isnan(table(temp=15.0))
    assert np.isnan(table(temp=40.0))


def test_transform_runs_at_compile_time_only():
    calls = []

    def fahrenheit(f, rh):
        calls.append(f.size)
        return {"temp": (f - 32.0) / 1.8, "humid": rh * 100.0}

    axes = {"f": (32, 104, 73), "rh": (0, 1, 51)}
    table = fz.compile(COOLER, axes, transform=fahrenheit)
    compiled = len(calls)
    assert sum(calls) == 73 * 51
    assert table(f=80.6, rh=0.4) == pytest.approx(COOLER(temp=27, humid=40), abs=0.1)
    assert len(calls) == compiled
    assert [axis.name for axis in table.axes] == ["f", "rh"]


def test_axes_take_points_or_ranges():
    table = fz.compile(COOLER, {"temp": 5, "humid": (20, 80)}, points=7)
    assert table.axes == (
        fz.table.Axis("temp", 0.0, 40.0, 5),
        fz.table.Axis("humid", 20.0, 80.0, 7),
    )
    with pytest.raises(ValueError, match="not an input"):
        fz.compile(COOLER, {"temp": 5, "speed": 5})
    with pytest.raises(ValueError, match="low < high"):
        fz.compile(COOLER, {"temp": (5, 5), "humid": 3})
    with pytest.raises(ValueError, match="at least 2 points"):
        fz.compile(COOLER, points=1)
    with pytest.raises(TypeError, match="give the axes"):
        fz.compile(plane)
    with pytest.raises(TypeError, match="needs axes"):
        fz.compile(COOLER, transform=lambda **inputs: inputs)
    with pytest.raises(TypeError, match="by name"):
        fz.compile(COOLER, {"temp": 3}, transform=lambda temp: temp)
    with pytest.raises(TypeError, match="missing input"):
        fz.compile(COOLER, {"temp": 3})
    with pytest.raises(ValueError, match="method"):
        fz.compile(COOLER, points=3, method="cubic")  # ty: ignore[invalid-argument-type]


def test_workers_give_the_same_table():
    seen = set()

    def system(x, y):
        seen.add(threading.get_ident())
        return plane(x, y)

    axes = {"x": (0, 1, 300), "y": (0, 1, 300)}
    alone = fz.compile(system, axes, workers=1)
    assert seen == {threading.get_ident()}
    together = fz.compile(system, axes, workers=4)
    assert len(seen) > 1
    np.testing.assert_array_equal(alone.values, together.values)
    with pytest.raises(ZeroDivisionError):
        fz.compile(lambda x, y: 1 / 0, axes, workers=4)


@pytest.mark.parametrize("compress", [False, True])
def test_table_survives_saving_and_loading(tmp_path, compress):
    table = fz.compile(WIND.north & TEMP.hot, points=16, method="nearest")
    path = tmp_path / "model.table"
    table.save(path, compress=compress)
    loaded = fz.load(path)
    assert loaded.axes == table.axes
    assert loaded.method == "nearest"
    assert loaded.source == "wind is north and temp is hot"
    assert repr(loaded) == repr(table)
    np.testing.assert_array_equal(loaded.values, table.values)
    assert loaded(wind=350, temp=28) == table(wind=350, temp=28)


def test_small_dtype_is_kept_on_disk_and_gives_float64(tmp_path):
    table = fz.compile(COOLER, points=11, dtype=np.float32)
    table.save(tmp_path / "small.npz")
    loaded = fz.load(tmp_path / "small.npz")
    assert loaded.values.dtype == np.float32
    assert loaded(temp=[27.0], humid=[40.0]).dtype == np.float64


def test_load_refuses_other_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        fz.load(tmp_path / "absent.npz")
    (tmp_path / "text.npz").write_text("not a table")
    with pytest.raises(ValueError, match="not a fuzlog table"):
        fz.load(tmp_path / "text.npz")
    np.savez(tmp_path / "arrays.npz", values=np.zeros(3))
    with pytest.raises(ValueError, match="not a fuzlog table"):
        fz.load(tmp_path / "arrays.npz")


def test_table_describes_itself():
    table = fz.compile(COOLER, points=11)
    assert repr(table) == (
        "Table(temp[11] x humid[11], linear, from Mamdani(3 rules: temp, humid -> fan))"
    )
    assert fz.compile(plane, {"x": (0, 1), "y": (0, 1)}, points=3).source == "plane"
