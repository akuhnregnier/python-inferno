# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy.testing import assert_allclose

from python_inferno.configuration import npft
from python_inferno.data import ConvergenceError, calc_litter_pool, calc_litter_pool_old


@pytest.fixture
def litter_pool_bench_kwargs_iter():
    def _kwargs_iter(seed=0, spinup_relative_delta=1e-2, max_spinup_cycles=int(1e3)):
        rng = np.random.default_rng(seed)

        while True:
            litter_tc = [1e-9 * 0.5 * (1 + rng.random()) for _ in range(npft)]
            leaf_f = [1e-3 * 0.5 * (1 + rng.random()) for _ in range(npft)]

            yield dict(
                litter_tc=litter_tc,
                leaf_f=leaf_f,
                Nt=None,
                spinup_relative_delta=spinup_relative_delta,
                max_spinup_cycles=max_spinup_cycles,
            )

    return _kwargs_iter


@pytest.mark.parametrize(
    "comp_delta, target_delta",
    [
        pytest.param(comp_delta, target_delta, marks=pytest.mark.slow)
        if i in (1, 2)
        else (comp_delta, target_delta)
        for (i, (comp_delta, target_delta)) in enumerate(
            [(1e-3, 5e-4), (1e-4, 4e-5), (1e-5, -1)]
        )
    ],
)
@pytest.mark.parametrize(
    "seed",
    [pytest.param(i, marks=pytest.mark.slow) if i != 0 else i for i in range(10)],
)
def test_litter_pool_versions(
    seed, comp_delta, target_delta, litter_pool_bench_kwargs_iter
):
    kwargs = {
        **next(litter_pool_bench_kwargs_iter(seed=seed)),
        **dict(max_spinup_cycles=int(1e2), spinup_relative_delta=target_delta),
    }

    funcs = (calc_litter_pool_old, calc_litter_pool)

    outputs = []
    for fn in funcs:
        if hasattr(fn, "_wrapped_func"):
            # Use unmemoized function to avoid memory buildup in the cache.
            fn = fn._wrapped_func

        outputs.append(fn(**kwargs))

    ref_output = outputs[0]

    for output in outputs[1:]:
        assert_allclose(output, ref_output, rtol=comp_delta, atol=0)


@pytest.mark.slow
@pytest.mark.parametrize("delta", (1e-2, 1e-3, 1e-4))
@pytest.mark.parametrize(
    "litter_pool_func", (calc_litter_pool, calc_litter_pool_old), ids=("new", "old")
)
def test_litter_pool_benchmark(
    benchmark, delta, litter_pool_bench_kwargs_iter, litter_pool_func
):
    if hasattr(litter_pool_func, "_wrapped_func"):
        # Use unmemoized function to avoid memory buildup in the cache.
        litter_pool_func = litter_pool_func._wrapped_func

    kwargs_iter = litter_pool_bench_kwargs_iter(spinup_relative_delta=delta)

    try:
        # Warm up (e.g. compile, memoization cache (not for benchmarked functions), etc...).
        litter_pool_func(**{**next(kwargs_iter), **dict(max_spinup_cycles=2)})
    except ConvergenceError:
        pass

    def _bench():
        litter_pool_func(**next(kwargs_iter))

    benchmark.pedantic(_bench, iterations=2, rounds=6)
