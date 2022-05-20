# -*- coding: utf-8 -*-

import numpy as np
import pytest
from numpy.testing import assert_allclose

from python_inferno.ba_model import (
    BAModel,
    GPUBAModel,
    GPUConsAvgBAModel,
    GPUConsAvgScoreBAModel,
)
from python_inferno.metrics import Metrics


def test_BAModel(params_model_ba):
    for params, expected_model_ba in params_model_ba:
        model_ba = BAModel(**params).run(**params)["model_ba"]
        assert_allclose(
            model_ba,
            expected_model_ba["python"],
            atol=1e-12,
            rtol=1e-7,
        )


@pytest.mark.parametrize("param_index", range(4))
@pytest.mark.parametrize(
    "seed",
    [pytest.param(i, marks=pytest.mark.slow) if i != 0 else i for i in range(100)],
)
def test_GPUBAModel(param_index, seed, params_model_ba):
    rng = np.random.default_rng(seed)

    params, _ = list(params_model_ba)[param_index]

    # Modify parameters using random permutations.
    mod_params = params.copy()
    for key in (
        "fapar_weight",
        "dryness_weight",
        "temperature_weight",
        "fuel_weight",
    ):
        mod_params[key] = rng.random()

    mod_params["crop_f"] = rng.random()
    mod_params["overall_scale"] = rng.random()

    gpu_model = GPUBAModel(**mod_params)
    gpu_ba = gpu_model.run(**mod_params)["model_ba"]

    py_model = BAModel(**mod_params)
    py_ba = py_model.run(**mod_params)["model_ba"]

    assert_allclose(gpu_ba, py_ba, atol=3e-11, rtol=1.2e-5)

    gpu_model.release()


# Tests to test / benchmark combined ConsAvg class.


@pytest.mark.parametrize("param_index", range(4))
@pytest.mark.parametrize(
    "seed",
    [pytest.param(i, marks=pytest.mark.slow) if i != 0 else i for i in range(100)],
)
def test_GPUInferno_cons_avg(param_index, seed, params_model_ba):
    rng = np.random.default_rng(seed)

    params, _ = list(params_model_ba)[param_index]

    # Modify parameters using random permutations.
    mod_params = params.copy()
    for key in (
        "fapar_weight",
        "dryness_weight",
        "temperature_weight",
        "fuel_weight",
    ):
        mod_params[key] = rng.random()

    gpu_cons_avg_model = GPUConsAvgBAModel(**mod_params)
    gpu_cons_avg_ba = gpu_cons_avg_model.run(**mod_params)["model_ba"]

    gpu_model = GPUBAModel(**mod_params)
    gpu_ba = gpu_model.run(**mod_params)["model_ba"]
    gpu_ba2 = gpu_model._cons_monthly_avg.cons_monthly_average_data(gpu_ba)

    py_model = BAModel(**mod_params)
    py_ba = py_model.run(**mod_params)["model_ba"]
    py_ba2 = py_model._cons_monthly_avg.cons_monthly_average_data(py_ba)

    # Compare un-averaged BA.
    assert_allclose(py_ba, gpu_ba, atol=3e-11, rtol=1e-5)

    # Compare averaged BA.
    assert_allclose(gpu_ba2, py_ba2, atol=3e-11, rtol=1e-5)
    assert_allclose(gpu_cons_avg_ba, gpu_ba2, atol=1e-10, rtol=1e-3)
    assert_allclose(gpu_cons_avg_ba, py_ba2, atol=1e-10, rtol=1e-3)

    gpu_cons_avg_model.release()
    gpu_model.release()


CONS_AVG_ITER = 10
CONS_AVG_ROUNDS = 100


@pytest.mark.parametrize("index", range(4))
def test_GPUInferno_cons_avg_combined_benchmark(index, params_model_ba, benchmark):
    params, _ = list(params_model_ba)[index]

    rng = np.random.default_rng(0)

    model1 = GPUConsAvgBAModel(**params)

    def _bench():
        mod_params = params.copy()
        mod_params["fapar_weight"] = rng.random()
        return model1.run(**mod_params)

    benchmark.pedantic(_bench, iterations=CONS_AVG_ITER, rounds=CONS_AVG_ROUNDS)

    model1.release()


@pytest.mark.parametrize("index", range(4))
def test_GPUInferno_cons_avg_separate_benchmark(index, params_model_ba, benchmark):
    params, _ = list(params_model_ba)[index]

    rng = np.random.default_rng(0)

    model2 = GPUBAModel(**params)

    def _bench():
        mod_params = params.copy()
        mod_params["fapar_weight"] = rng.random()
        return model2._cons_monthly_avg.cons_monthly_average_data(
            model2.run(**mod_params)["model_ba"]
        )

    benchmark.pedantic(_bench, iterations=CONS_AVG_ITER, rounds=CONS_AVG_ROUNDS)

    model2.release()


def test_calculate_scores_benchmark(benchmark, model_params):
    rng = np.random.default_rng(0)

    params = next(iter(model_params.values()))  # Get first value.
    ba_model = BAModel(**params)
    model_ba = ba_model.run(**params)["model_ba"]

    def _bench():
        model_ba.ravel()[0] += 1e-5 * rng.random()
        # Ensure +ve.
        model_ba.ravel()[0] = abs(model_ba.ravel()[0])

        ba_model.calc_scores(
            model_ba=model_ba,
            requested=(Metrics.MPD, Metrics.ARCSINH_NME),
        )

    benchmark(_bench)


TOTAL_ITER = 10
TOTAL_ROUNDS = 10


@pytest.mark.parametrize(
    "model_class, iter_f", ((BAModel, 1), (GPUBAModel, 10), (GPUConsAvgBAModel, 20))
)
@pytest.mark.parametrize("index", range(4))
def test_total_opt_benchmark(index, model_class, iter_f, benchmark, model_params):
    rng = np.random.default_rng(0)

    params = list(model_params.values())[index]

    ba_model = model_class(**params)

    def _bench():
        model_ba = ba_model.run(
            **{
                **params,
                **dict(
                    fapar_weight=rng.random(),
                    dryness_weight=rng.random(),
                    temperature_weight=rng.random(),
                    fuel_weight=rng.random(),
                ),
            }
        )["model_ba"]

        return ba_model.calc_scores(
            model_ba=model_ba,
            requested=(Metrics.MPD, Metrics.ARCSINH_NME),
        )

    benchmark.pedantic(_bench, iterations=iter_f * TOTAL_ITER, rounds=TOTAL_ROUNDS)


@pytest.mark.parametrize("index", range(4))
def test_gpu_score_total_opt_benchmark(index, benchmark, model_params):
    rng = np.random.default_rng(0)

    params = list(model_params.values())[index]

    ba_model = GPUConsAvgScoreBAModel(**params)

    def _bench():
        scores = ba_model.get_scores(
            requested=(Metrics.MPD, Metrics.ARCSINH_NME),
            **{
                **params,
                **dict(
                    fapar_weight=rng.random(),
                    dryness_weight=rng.random(),
                    temperature_weight=rng.random(),
                    fuel_weight=rng.random(),
                ),
            }
        )

        return scores

    benchmark.pedantic(_bench, iterations=30 * TOTAL_ITER, rounds=TOTAL_ROUNDS)
