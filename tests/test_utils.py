# -*- coding: utf-8 -*-
from datetime import datetime
from functools import reduce
from operator import mul

import cf_units
import iris
import iris.cube  # noqa
import numpy as np
import pytest
from dateutil.relativedelta import relativedelta
from numpy.testing import assert_allclose, assert_array_equal

from python_inferno.py_gpu_inferno import (
    GPUConsAvg,
    GPUConsAvgNoMask,
    cpp_cons_avg_no_mask_inplace,
)
from python_inferno.utils import (
    ConsAvg3NoMask,
    ConsMonthlyAvg,
    _cons_avg,
    _cons_avg2,
    _cons_avg2_no_mask,
    dict_match,
    expand_pft_params,
    exponential_average,
    get_pft_group_index,
    linspace_no_endpoint,
    memoize,
    monthly_average_data,
    moving_sum,
    temporal_nearest_neighbour_interp,
    temporal_processing,
    wrap_phase_diffs,
)

N_ITER = 50
N_ROUNDS = 500


def test_temporal_nearest_neighbour_interp_0d_centre():
    initial = np.array([10, 11, 12])
    interpolated = temporal_nearest_neighbour_interp(initial, 3, "centre")
    assert_allclose(
        np.array([10, 10, 11, 11, 11, 12, 12, 12, 12]),
        interpolated.data,
    )
    assert interpolated.mask == False


def test_temporal_nearest_neighbour_interp_0d_start():
    initial = np.array([10, 11, 12])
    interpolated = temporal_nearest_neighbour_interp(initial, 3, "start")
    assert_allclose(
        np.array([10, 10, 10, 11, 11, 11, 12, 12, 12]),
        interpolated.data,
    )
    assert interpolated.mask == False


def test_temporal_nearest_neighbour_interp_0d_end():
    initial = np.array([10, 11, 12])
    interpolated = temporal_nearest_neighbour_interp(initial, 3, "end")
    assert_allclose(
        np.array([10, 11, 11, 11, 12, 12, 12, 12, 12]),
        interpolated.data,
    )
    assert interpolated.mask == False


def test_temporal_nearest_neighbour_interp_Nd():
    initial = np.array([10, 11, 12]).reshape(3, 1)
    assert_allclose(
        np.array([10, 10, 11, 11, 11, 12, 12, 12, 12]).reshape(9, 1),
        temporal_nearest_neighbour_interp(initial, 3, "centre").data,
    )


def test_temporal_nearest_neighbour_interp_Nd_mask():
    initial = np.ma.MaskedArray(
        np.array([10, 11, 12]).reshape(3, 1), mask=np.array([0, 1, 0], dtype=np.bool_)
    )
    interpolated = temporal_nearest_neighbour_interp(initial, 3, "centre")
    assert_allclose(
        np.array([10, 10, 11, 11, 11, 12, 12, 12, 12]).reshape(9, 1),
        interpolated.data,
    )
    assert np.all(
        interpolated.mask
        == np.array([0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=np.bool_).reshape(9, 1)
    )


def test_exponential_average_1d():
    data = np.arange(1, 4)
    assert exponential_average(data, 0.1).shape == data.shape
    assert_allclose(exponential_average(data, 1), data)
    assert_allclose(exponential_average(data, 0), np.zeros_like(data))


def test_exponential_average_1d_mask():
    data = np.ma.MaskedArray(np.arange(1, 4), mask=True)
    assert exponential_average(data, 0.1).shape == data.shape
    assert np.all(exponential_average(data, 0.1).mask)

    data = np.ma.MaskedArray(np.arange(1, 4), mask=np.array([0, 1, 0], dtype=np.bool_))
    assert exponential_average(data, 0.1).shape == data.shape
    assert np.all(exponential_average(data, 0.1).mask)


def test_exponential_average_Nd_mask():
    data = np.ma.MaskedArray(
        np.arange(0, 6).reshape(2, 3),
        mask=np.array([[0, 1, 0], [0, 0, 0]], dtype=np.bool_),
    )
    assert exponential_average(data, 0.1).shape == data.shape
    assert np.all(
        exponential_average(data, 0.1).mask
        == np.array([[0, 1, 0], [0, 1, 0]], dtype=np.bool_)
    )

    assert_allclose(exponential_average(data[:, [0, 2]], 1), data[:, [0, 2]])


def test_moving_sum():
    data = np.arange(3)
    assert_allclose(moving_sum(data, 1), data)
    assert_allclose(moving_sum(data, 2), [0, 1, 3])
    assert_allclose(moving_sum(data, 3), [0, 1, 3])

    data = np.arange(5)
    assert_allclose(moving_sum(data, 1), data)
    assert_allclose(moving_sum(data, 2), [0, 1, 3, 5, 7])
    assert_allclose(moving_sum(data, 3), [0, 1, 3, 6, 9])

    assert_allclose(moving_sum(data[:, None], 3), np.array([0, 1, 3, 6, 9])[:, None])


def test_expand_pft_params():
    assert tuple(expand_pft_params([1, 2, 3])) == (
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
    )


def get_daily_time_coord(n):
    units = cf_units.Unit("days since 1970-01-01", calendar="gregorian")
    num_dates = units.date2num(
        [datetime(1970, 1, 1) + relativedelta(days=days) for days in range(n)]
    )
    return iris.coords.DimCoord(
        num_dates, standard_name="time", var_name="time", units=units
    )


def test_temporal_processing_no_agg():
    data_dict = {
        "a": np.arange(10 * 13 * 10).reshape(10, 13, 10),
        "b": np.arange(10 * 10).reshape(10, 10),
    }
    shifts = np.array([1, 2, 3])
    antecedent_shifts_dict = {"a": shifts}
    out, time_coord = temporal_processing(
        data_dict=data_dict.copy(),
        antecedent_shifts_dict=antecedent_shifts_dict,
        average_samples=0,
        time_coord=get_daily_time_coord(10),
    )
    assert np.allclose(out["a"][:, :5], data_dict["a"][2:-1, :5])
    assert np.allclose(out["a"][:, 5:11], data_dict["a"][1:-2, 5:11])
    assert np.allclose(out["a"][:, 11:13], data_dict["a"][:-3, 11:13])
    assert np.allclose(out["b"], data_dict["b"][3:])

    assert time_coord.cell(0).point == datetime(1970, 1, 4)
    assert time_coord.cell(-1).point == datetime(1970, 1, 10)


@pytest.mark.parametrize(
    "aggregator, agg_method",
    [
        ("MEAN", np.mean),
        ("MIN", np.min),
        ("MAX", np.max),
    ],
)
def test_temporal_processing_agg(aggregator, agg_method):
    data_dict = {
        "a": np.arange(10 * 13 * 10).reshape(10, 13, 10),
        "b": np.arange(10 * 10).reshape(10, 10),
    }
    shifts = np.array([1, 2, 3])
    antecedent_shifts_dict = {"a": shifts}
    out, time_coord = temporal_processing(
        data_dict=data_dict.copy(),
        antecedent_shifts_dict=antecedent_shifts_dict,
        average_samples=2,
        aggregator=aggregator,
        time_coord=get_daily_time_coord(10),
    )
    assert np.allclose(out["a"][0, :5], agg_method(data_dict["a"][2:4, :5], axis=0))
    assert np.allclose(out["a"][0, 5:11], agg_method(data_dict["a"][1:3, 5:11], axis=0))
    assert np.allclose(
        out["a"][0, 11:13], agg_method(data_dict["a"][:2, 11:13], axis=0)
    )
    assert np.allclose(out["b"][0], agg_method(data_dict["b"][3:5], axis=0))

    assert_allclose(time_coord.points[0], 3.5)
    assert_allclose(time_coord.points[-1], 9)
    assert time_coord.cell(0).bound == (datetime(1970, 1, 4), datetime(1970, 1, 5))
    assert time_coord.cell(-1).bound == (datetime(1970, 1, 10), datetime(1970, 1, 10))


def test_temporal_processing_multi_agg():
    data_dict = {
        "a": np.arange(10 * 13 * 10).reshape(10, 13, 10),
        "b": np.arange(10 * 10).reshape(10, 10),
    }
    shifts = np.array([1, 2, 3])
    antecedent_shifts_dict = {"a": shifts}
    out, time_coord = temporal_processing(
        data_dict=data_dict.copy(),
        antecedent_shifts_dict=antecedent_shifts_dict,
        average_samples=2,
        aggregator={"a": "MIN", "b": "MAX"},
        time_coord=get_daily_time_coord(10),
    )
    assert np.allclose(out["a"][0, :5], np.min(data_dict["a"][2:4, :5], axis=0))
    assert np.allclose(out["a"][0, 5:11], np.min(data_dict["a"][1:3, 5:11], axis=0))
    assert np.allclose(out["a"][0, 11:13], np.min(data_dict["a"][:2, 11:13], axis=0))
    assert np.allclose(out["b"][0], np.max(data_dict["b"][3:5], axis=0))

    assert_allclose(time_coord.points[0], 3.5)
    assert_allclose(time_coord.points[-1], 9)
    assert time_coord.cell(0).bound == (datetime(1970, 1, 4), datetime(1970, 1, 5))
    assert time_coord.cell(-1).bound == (datetime(1970, 1, 10), datetime(1970, 1, 10))


def test_get_pft_group_index():
    assert get_pft_group_index(0) == 0
    assert get_pft_group_index(3) == 0
    assert get_pft_group_index(4) == 0
    assert get_pft_group_index(5) == 1
    assert get_pft_group_index(10) == 1
    assert get_pft_group_index(11) == 2
    assert get_pft_group_index(12) == 2


def test_monthly_average_data():
    data = np.random.default_rng(0).random((2, 2))

    # Define the time coordinate.
    units = cf_units.Unit("days since 1970-01-01", calendar="gregorian")
    num_dates = units.date2num(
        [
            datetime(1970, 1, 1) + relativedelta(months=months)
            for months in range(data.shape[0] + 1)
        ]
    )
    time_coord = iris.coords.DimCoord(
        (num_dates[1:] + num_dates[:-1]) / 2.0,
        bounds=np.hstack((num_dates[:-1].reshape(-1, 1), num_dates[1:].reshape(-1, 1))),
        standard_name="time",
        var_name="time",
        units=units,
    )

    mon_avg = monthly_average_data(data, time_coord=time_coord)
    mon_avg_con = ConsMonthlyAvg(time_coord, L=data.shape[1]).cons_monthly_average_data(
        data
    )

    assert np.allclose(mon_avg, mon_avg_con)


def test_monthly_average_data_days():
    shape = (4, 2)
    data = np.arange(reduce(mul, shape)).reshape(shape)

    # Define the time coordinate.
    units = cf_units.Unit("days since 1970-01-01", calendar="gregorian")
    num_dates = units.date2num(
        [datetime(1970, 1, 1) + relativedelta(days=days) for days in range(0, 49, 12)]
    )
    time_coord = iris.coords.DimCoord(
        (num_dates[1:] + num_dates[:-1]) / 2.0,
        bounds=np.hstack((num_dates[:-1].reshape(-1, 1), num_dates[1:].reshape(-1, 1))),
        standard_name="time",
        var_name="time",
        units=units,
    )

    mon_avg = monthly_average_data(data, time_coord=time_coord)
    mon_avg_con = ConsMonthlyAvg(time_coord, L=data.shape[1]).cons_monthly_average_data(
        data
    )

    assert np.allclose(mon_avg, np.vstack((data[1][np.newaxis], data[-1][np.newaxis])))

    # Width of sample bounds in seconds.
    bound_width = (
        time_coord.cell(0).bound[1] - time_coord.cell(0).bound[0]
    ).total_seconds()

    # The 3rd sample overlaps both months, so determine the contributions to each
    # month.
    contribs = np.zeros(2)
    contribs[0] = (datetime(1970, 2, 1) - time_coord.cell(2).bound[0]).total_seconds()
    contribs[1] = (time_coord.cell(2).bound[1] - datetime(1970, 2, 1)).total_seconds()
    assert np.isclose(np.sum(contribs), bound_width)

    # Calculate the expected conservative average.
    weights = np.zeros((4, 2))
    comp_avg_con = np.zeros((2, 2))

    weights[0:2, 0] = bound_width
    weights[2, 0] = contribs[0]
    weights[2, 1] = contribs[1]
    weights[3, 1] = bound_width

    for i in range(2):
        comp_avg_con[i] = np.sum(weights[:, i][:, np.newaxis] * data, axis=0) / np.sum(
            weights[:, i]
        )

    assert np.allclose(mon_avg_con, comp_avg_con)


@pytest.mark.slow
def test_monthly_average_data_rand():
    data = np.ma.MaskedArray(np.random.default_rng(0).random((100, 10000)), mask=False)
    # Get mask array.
    data.mask = np.ma.getmaskarray(data.mask)
    data.mask[:-1, 0] = True
    data.mask[:-2, 1] = True
    data.mask[1:, 2] = True
    data.mask[2:, 3] = True

    # Define the time coordinate.
    units = cf_units.Unit("days since 1970-01-01", calendar="gregorian")
    s_per_day = 24 * 60 * 60
    num_dates = units.date2num(
        [
            datetime(1970, 1, 1) + relativedelta(seconds=int(seconds))
            for seconds in np.linspace(
                0, 80 * s_per_day, data.shape[0] + 1, dtype=np.int64
            )
        ]
    )
    time_coord = iris.coords.DimCoord(
        (num_dates[1:] + num_dates[:-1]) / 2.0,
        bounds=np.hstack((num_dates[:-1].reshape(-1, 1), num_dates[1:].reshape(-1, 1))),
        standard_name="time",
        var_name="time",
        units=units,
    )

    mon_avg_con = ConsMonthlyAvg(time_coord, L=data.shape[1]).cons_monthly_average_data(
        data
    )

    assert np.all(mon_avg_con <= np.max(data))
    assert np.all(mon_avg_con >= np.min(data))

    assert np.isclose(mon_avg_con[-1, 0], data[-1, 0])
    assert np.isclose(mon_avg_con[-1, 1], np.mean(data[-2:, 1]))

    assert np.isclose(mon_avg_con[0, 2], data[0, 2])
    assert np.isclose(mon_avg_con[0, 3], np.mean(data[:2, 3]))


def test_dict_match():
    assert dict_match({"a": 1}, {"a": 1})
    assert dict_match({"a": 1, "b": 2.0}, {"a": 1, "b": 2.0})
    assert dict_match({"a": 1, "b": 2.123}, {"a": 1, "b": 2.123})
    assert not dict_match({"a": 1 - 1e-4, "b": 2.123}, {"a": 1, "b": 2.123})
    assert not dict_match({"b": 2.123}, {"a": 1, "b": 2.123})


def test_memoize():
    def fn_side_effects():
        if not hasattr(fn_side_effects, "counter"):
            fn_side_effects.counter = 0
        fn_side_effects.counter += 1

        return fn_side_effects.counter

    assert (fn_side_effects(), fn_side_effects()) == (1, 2)

    @memoize
    def memo_fn_side_effects():
        if not hasattr(memo_fn_side_effects, "counter"):
            memo_fn_side_effects.counter = 0
        memo_fn_side_effects.counter += 1

        return memo_fn_side_effects.counter

    assert (memo_fn_side_effects(), memo_fn_side_effects()) == (1, 1)


def test_inactive_memoize(monkeypatch):
    @memoize
    def memo_fn_side_effects():
        if not hasattr(memo_fn_side_effects, "counter"):
            memo_fn_side_effects.counter = 0
        memo_fn_side_effects.counter += 1

        return memo_fn_side_effects.counter

    monkeypatch.setattr(memoize, "active", False)

    assert (memo_fn_side_effects(), memo_fn_side_effects()) == (1, 2)


@pytest.mark.parametrize("args", [(0, 1, 12), (-10, 20, 4)])
def test_linspace_no_endpoint(args):
    assert_allclose(np.linspace(*args, endpoint=False), linspace_no_endpoint(*args))


def test_wrap_phase_diffs():
    assert_allclose(
        wrap_phase_diffs([-12, 12, 0, -6, 6, -10, 10, 8, -8, -4, 4]),
        [0, 0, 0, -6, -6, 2, -2, -4, 4, -4, 4],
    )


@pytest.fixture
def get_cons_avg_data():
    def get_data(seed=0):
        rng = np.random.default_rng(seed)

        Nt = 22
        Nout = 12
        L = 7771

        weights = rng.random((Nt, Nout), dtype=np.float32)
        weights[weights < 0.5] = 0
        in_data = rng.random((Nt, L), dtype=np.float32)
        in_mask = rng.random((Nt, L)) < 0.1
        out_data = np.zeros((Nout, L), dtype=np.float32)
        out_mask = np.ones((Nout, L), dtype=np.bool_)
        cum_weights = np.zeros((Nout, L), dtype=np.float32)

        return Nt, Nout, weights, in_data, in_mask, out_data, out_mask, cum_weights

    return get_data


def test_cons_avg_benchmark(benchmark, get_cons_avg_data):
    benchmark(_cons_avg, *get_cons_avg_data())


@pytest.mark.parametrize("seed", list(range(100)))
def test_cons_avg_implementations(get_cons_avg_data, seed):
    (
        Nt,
        Nout,
        weights,
        in_data,
        in_mask,
        out_data,
        out_mask,
        cum_weights,
    ) = get_cons_avg_data(seed=seed)
    data1, mask1 = _cons_avg(
        Nt, Nout, weights, in_data, in_mask, out_data, out_mask, cum_weights
    )
    data2, mask2 = _cons_avg2(Nt, Nout, weights, in_data, in_mask)
    data3, mask3 = GPUConsAvg(L=7771, weights=weights).run(in_data, in_mask)

    assert_array_equal(mask1, mask2)
    assert_array_equal(mask1, mask3)

    assert_allclose(data1[~mask1], data2[~mask1], rtol=5e-7, atol=1e-7)
    assert_allclose(data1[~mask1], data3[~mask1], rtol=5e-7, atol=1e-7)


def test_cons_avg2_benchmark(benchmark, get_cons_avg_data):
    benchmark(_cons_avg2, *get_cons_avg_data()[:-3])


@pytest.mark.parametrize("seed", list(range(100)))
def test_cons_avg2_no_mask(get_cons_avg_data, seed):
    (
        Nt,
        Nout,
        weights,
        in_data,
        in_mask,
        out_data,
        out_mask,
        cum_weights,
    ) = get_cons_avg_data(seed=seed)

    with_mask_out = _cons_avg2(
        Nt, Nout, weights, in_data, np.zeros_like(in_mask, dtype=np.bool_)
    )[0]
    no_mask_out = _cons_avg2_no_mask(Nt, Nout, weights, in_data)

    assert_allclose(no_mask_out, with_mask_out)


def test_cons_avg2_no_mask_benchmark(benchmark, get_cons_avg_data):
    benchmark(_cons_avg2_no_mask, *get_cons_avg_data()[:-4])


@pytest.mark.parametrize("seed", list(range(100)))
def test_cons_avg3_no_mask(get_cons_avg_data, seed):
    (
        Nt,
        Nout,
        weights,
        in_data,
        in_mask,
        out_data,
        out_mask,
        cum_weights,
    ) = get_cons_avg_data(seed=seed)

    with_mask_out = _cons_avg2(
        Nt, Nout, weights, in_data, np.zeros_like(in_mask, dtype=np.bool_)
    )[0]

    no_mask_out = ConsAvg3NoMask(weights=weights, out_data=out_data).cons_avg(in_data)

    assert_allclose(no_mask_out, with_mask_out)


def test_cons_avg3_no_mask_benchmark(benchmark, get_cons_avg_data):
    (
        Nt,
        Nout,
        weights,
        in_data,
        in_mask,
        out_data,
        out_mask,
        cum_weights,
    ) = get_cons_avg_data()

    benchmark.pedantic(
        ConsAvg3NoMask(weights=weights, out_data=out_data).cons_avg,
        args=(in_data,),
        iterations=N_ITER,
        rounds=N_ROUNDS,
    )


def test_gpu_cons_avg_benchmark(benchmark, get_cons_avg_data):
    (
        Nt,
        Nout,
        weights,
        in_data,
        in_mask,
        out_data,
        out_mask,
        cum_weights,
    ) = get_cons_avg_data()
    gpu_cons_avg = GPUConsAvg(L=7771, weights=weights)
    benchmark(gpu_cons_avg.run, in_data, in_mask)


@pytest.mark.parametrize("seed", list(range(100)))
def test_gpu_cons_avg_no_mask(get_cons_avg_data, seed):
    (
        Nt,
        Nout,
        weights,
        in_data,
        in_mask,
        out_data,
        out_mask,
        cum_weights,
    ) = get_cons_avg_data(seed=seed)

    gpu_cons_avg = GPUConsAvg(L=7771, weights=weights)
    gpu_cons_avg_no_mask = GPUConsAvgNoMask(L=7771, weights=weights)

    with_mask_out = gpu_cons_avg.run(in_data, np.zeros_like(in_mask, dtype=np.bool_))[0]
    no_mask_out = gpu_cons_avg_no_mask.run(in_data)

    assert_allclose(no_mask_out, with_mask_out)


def test_gpu_cons_avg_no_mask_benchmark(benchmark, get_cons_avg_data):
    (
        Nt,
        Nout,
        weights,
        in_data,
        in_mask,
        out_data,
        out_mask,
        cum_weights,
    ) = get_cons_avg_data()
    gpu_cons_avg_no_mask = GPUConsAvgNoMask(L=7771, weights=weights)
    benchmark.pedantic(
        gpu_cons_avg_no_mask.run,
        args=(in_data,),
        iterations=N_ITER,
        rounds=N_ROUNDS,
    )


@pytest.mark.parametrize("seed", list(range(100)))
def test_cons_avg_no_mask_implementations(get_cons_avg_data, seed):
    (
        Nt,
        Nout,
        weights,
        in_data,
        in_mask,
        out_data,
        out_mask,
        cum_weights,
    ) = get_cons_avg_data(seed=seed)

    ref_out = GPUConsAvgNoMask(L=7771, weights=weights).run(in_data)

    assert_allclose(
        ref_out,
        _cons_avg2_no_mask(Nt, Nout, weights, in_data),
        atol=1e-7,
        rtol=1e-7,
    )

    cpp_cons_avg_no_mask_inplace(weights=weights, data=in_data, out=out_data)

    assert_allclose(
        ref_out,
        out_data,
        atol=3e-7,
        rtol=3e-7,
    )


def test_cpp_cons_avg_no_mask_benchmark(benchmark, get_cons_avg_data):
    (
        Nt,
        Nout,
        weights,
        in_data,
        in_mask,
        out_data,
        out_mask,
        cum_weights,
    ) = get_cons_avg_data()
    benchmark(cpp_cons_avg_no_mask_inplace, weights=weights, data=in_data, out=out_data)


@pytest.mark.slow
def test_cons_avg3_no_mask_benchmark_rand(benchmark, get_cons_avg_data):
    (
        Nt,
        Nout,
        weights,
        in_data,
        in_mask,
        out_data,
        out_mask,
        cum_weights,
    ) = get_cons_avg_data()

    cons_avg3 = ConsAvg3NoMask(weights=weights, out_data=out_data).cons_avg

    rng = np.random.default_rng(0)

    def _bench():
        rng.random(dtype=np.float32, out=in_data)
        cons_avg3(in_data)

    benchmark.pedantic(_bench, iterations=N_ITER, rounds=N_ROUNDS)


@pytest.mark.slow
def test_gpu_cons_avg_no_mask_benchmark_rand(benchmark, get_cons_avg_data):
    (
        Nt,
        Nout,
        weights,
        in_data,
        in_mask,
        out_data,
        out_mask,
        cum_weights,
    ) = get_cons_avg_data()

    gpu_cons_avg = GPUConsAvgNoMask(L=7771, weights=weights).run

    rng = np.random.default_rng(0)

    def _bench():
        rng.random(dtype=np.float32, out=in_data)

        gpu_cons_avg(in_data)

    benchmark.pedantic(_bench, iterations=N_ITER, rounds=N_ROUNDS)


@pytest.mark.slow
def test_cons_avg2_no_mask_benchmark_rand(benchmark, get_cons_avg_data):
    (
        Nt,
        Nout,
        weights,
        in_data,
        in_mask,
        out_data,
        out_mask,
        cum_weights,
    ) = get_cons_avg_data()

    rng = np.random.default_rng(0)

    def _bench():
        rng.random(dtype=np.float32, out=in_data)

        _cons_avg2_no_mask(Nt, Nout, weights, in_data)

    benchmark.pedantic(_bench, iterations=N_ITER, rounds=N_ROUNDS)
