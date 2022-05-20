# -*- coding: utf-8 -*-

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from python_inferno.ba_model import (
    BAModel,
    GPUBAModel,
    GPUConsAvgBAModel,
    GPUConsAvgScoreBAModel,
)
from python_inferno.configuration import (
    N_pft_groups,
    land_pts,
    n_cell_grp_pft,
    n_cell_nat_pft,
    n_cell_no_pft,
    n_cell_tot_pft,
)
from python_inferno.inferno import calc_flam
from python_inferno.metrics import Metrics
from python_inferno.py_gpu_inferno import (
    GPUCalculateMPD,
    GPUCalculatePhase,
    GPUFlam2,
    _GPUCompute,
    _GPUInfernoAvg,
    cpp_nme,
)


@pytest.fixture
def compute():
    compute = _GPUCompute()
    yield compute
    compute.release()


@pytest.fixture
def get_compute_data():
    def _get_compute_data(
        includeTemperature=1,
        fuelBuildUpMethod=1,
        drynessMethod=1,
        Nt=10,
    ):
        return dict(
            ignitionMethod=1,
            flammabilityMethod=2,
            drynessMethod=drynessMethod,
            fuelBuildUpMethod=fuelBuildUpMethod,
            includeTemperature=includeTemperature,
            Nt=Nt,
            t1p5m_tile=np.zeros(n_cell_tot_pft(Nt), dtype=np.float32) + 300,
            q1p5m_tile=np.zeros(n_cell_tot_pft(Nt), dtype=np.float32),
            pstar=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
            sthu_soilt_single=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
            frac=np.zeros(n_cell_tot_pft(Nt), dtype=np.float32),
            c_soil_dpm_gb=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
            c_soil_rpm_gb=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
            canht=np.zeros(n_cell_nat_pft(Nt), dtype=np.float32),
            ls_rain=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
            con_rain=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
            pop_den=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
            flash_rate=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
            fuel_build_up=np.zeros(n_cell_nat_pft(Nt), dtype=np.float32),
            fapar_diag_pft=np.zeros(n_cell_nat_pft(Nt), dtype=np.float32),
            grouped_dry_bal=np.zeros(n_cell_grp_pft(Nt), dtype=np.float32),
            litter_pool=np.zeros(n_cell_nat_pft(Nt), dtype=np.float32),
            dry_days=np.zeros(n_cell_no_pft(Nt), dtype=np.float32),
            checks_failed=np.zeros(n_cell_nat_pft(Nt), dtype=np.bool_),
        )

    return _get_compute_data


@pytest.fixture
def compute_params():
    return dict(
        fapar_factor=np.zeros(N_pft_groups, dtype=np.float32),
        fapar_centre=np.zeros(N_pft_groups, dtype=np.float32),
        fapar_shape=np.zeros(N_pft_groups, dtype=np.float32),
        fuel_build_up_factor=np.zeros(N_pft_groups, dtype=np.float32),
        fuel_build_up_centre=np.zeros(N_pft_groups, dtype=np.float32),
        fuel_build_up_shape=np.zeros(N_pft_groups, dtype=np.float32),
        temperature_factor=np.zeros(N_pft_groups, dtype=np.float32),
        temperature_centre=np.zeros(N_pft_groups, dtype=np.float32),
        temperature_shape=np.zeros(N_pft_groups, dtype=np.float32),
        dry_day_factor=np.zeros(N_pft_groups, dtype=np.float32),
        dry_day_centre=np.zeros(N_pft_groups, dtype=np.float32),
        dry_day_shape=np.zeros(N_pft_groups, dtype=np.float32),
        dry_bal_factor=np.zeros(N_pft_groups, dtype=np.float32),
        dry_bal_centre=np.zeros(N_pft_groups, dtype=np.float32),
        dry_bal_shape=np.zeros(N_pft_groups, dtype=np.float32),
        litter_pool_factor=np.zeros(N_pft_groups, dtype=np.float32),
        litter_pool_centre=np.zeros(N_pft_groups, dtype=np.float32),
        litter_pool_shape=np.zeros(N_pft_groups, dtype=np.float32),
        fapar_weight=np.ones(N_pft_groups, dtype=np.float32),
        dryness_weight=np.ones(N_pft_groups, dtype=np.float32),
        temperature_weight=np.ones(N_pft_groups, dtype=np.float32),
        fuel_weight=np.ones(N_pft_groups, dtype=np.float32),
    )


@pytest.mark.parametrize("includeTemperature", [0, 1])
@pytest.mark.parametrize("fuelBuildUpMethod", [1, 2])
@pytest.mark.parametrize("drynessMethod", [1, 2])
@pytest.mark.parametrize("Nt", [100, 10])
def test_GPUCompute_synthetic(
    get_compute_data,
    compute_params,
    compute,
    includeTemperature,
    fuelBuildUpMethod,
    drynessMethod,
    Nt,
):
    compute.set_data(
        **get_compute_data(
            includeTemperature=includeTemperature,
            fuelBuildUpMethod=fuelBuildUpMethod,
            drynessMethod=drynessMethod,
            Nt=Nt,
        )
    )

    compute.set_params(**compute_params)

    compute.run(np.empty((Nt, land_pts), dtype=np.float32))


def test_multiple_gpucompute():
    instances = []

    for i in range(10):
        compute = _GPUCompute()
        instances.append(compute)

    for instance in instances:
        instance.release()


@pytest.mark.timeout(45)
def test_many_run(compute, get_compute_data, compute_params):
    Nt = 10
    compute.set_data(**get_compute_data(Nt=Nt))
    for i in range(10000):
        compute.set_params(**compute_params)
        compute.run(np.empty((Nt, land_pts), dtype=np.float32))


def test_GPUBAModel(params_model_ba):
    for params, expected_model_ba in params_model_ba:
        rng = np.random.default_rng(0)

        rand_params = {key: rng.random(1) for key in params}

        # Set up the model.
        model = GPUBAModel(**params)
        # Initialise the parameters using random values to ensure that modifying the
        # parameters later on works as expected.
        random_model_ba = model.run(**rand_params)["model_ba"]
        assert not np.allclose(
            random_model_ba, expected_model_ba["metal"], atol=1e-12, rtol=1e-7
        )
        # Set the proper parameters and run.
        model_ba = model.run(**params)["model_ba"]
        assert_allclose(model_ba, expected_model_ba["metal"], atol=1e-12, rtol=1e-7)

        model.release()


def test_multiple_phase_instances():
    phases = []
    for i in range(10):
        phases.append(GPUCalculatePhase(i + 10))
        assert phases[-1].run(
            np.random.default_rng(0).random((12, i + 10), dtype=np.float32)
        ).shape == (i + 10,)


def test_gpu_phase(benchmark):
    gpu_calc = GPUCalculatePhase(7771)
    benchmark(
        gpu_calc.run, x=np.random.default_rng(0).random((12, 7771), dtype=np.float32)
    )


def test_gpu_mpd():
    gpu_mpd = GPUCalculateMPD(7771)
    mpd, ignored = gpu_mpd.run(
        obs=np.random.default_rng(0).random((12, 7771), dtype=np.float32),
        pred=np.random.default_rng(1).random((12, 7771), dtype=np.float32),
        return_ignored=True,
    )
    assert mpd > 0
    assert ignored >= 0


def test_multiple_mpd_instances():
    gpu_mpds = []
    for i in range(100):
        gpu_mpds.append(GPUCalculateMPD(i + 10))
        assert (
            len(
                gpu_mpds[-1].run(
                    obs=np.random.default_rng(0).random((12, i + 10), dtype=np.float32),
                    pred=np.random.default_rng(1).random(
                        (12, i + 10), dtype=np.float32
                    ),
                    return_ignored=True,
                )
            )
            == 2
        )


def test_multiple_mpd_instances2():
    gpu = GPUCalculateMPD(7771)
    mpd = gpu.run

    obs = np.random.default_rng(0).random((12, 7771))
    assert_allclose(mpd(obs=obs, pred=obs), 0)

    gpu2 = GPUCalculateMPD(100)
    mpd2 = gpu2.run

    obs = np.random.default_rng(0).random((12, 100))
    assert_allclose(mpd2(obs=obs, pred=obs), 0)


def test_cpp_nme():
    rng = np.random.default_rng(0)
    assert (
        cpp_nme(
            obs=rng.random(1000, dtype=np.float32),
            pred=rng.random(1000, dtype=np.float32),
        )
        > 0
    )


@pytest.mark.parametrize("includeTemperature", [0, 1])
@pytest.mark.parametrize("fuelBuildUpMethod", [1, 2])
@pytest.mark.parametrize("drynessMethod", [1, 2])
@pytest.mark.parametrize("Nt", [100, 10])
def test_GPUInfernoConsAvg_synthetic(
    get_compute_data,
    compute_params,
    includeTemperature,
    fuelBuildUpMethod,
    drynessMethod,
    Nt,
):
    compute = _GPUInfernoAvg(
        land_pts,
        np.random.default_rng(0).random((Nt, 12), dtype=np.float32),
    )
    compute.set_data(
        **get_compute_data(
            includeTemperature=includeTemperature,
            fuelBuildUpMethod=fuelBuildUpMethod,
            drynessMethod=drynessMethod,
            Nt=Nt,
        )
    )
    compute.set_params(**compute_params)
    compute.run(np.empty((12, land_pts), dtype=np.float32))
    compute.release()


def test_GPUInfernoConsAvg(params_model_ba):
    for params, expected_model_ba in params_model_ba:
        # Set up the model.
        model = GPUConsAvgBAModel(**params)

        test_ba = model.run(**params)["model_ba"]

        # Carry out monthly averaging of the expected output to compare to the already
        # averaged output above.
        expected = model._cons_monthly_avg.cons_monthly_average_data(
            expected_model_ba["metal"]
        )

        assert_allclose(test_ba, expected, atol=1e-12, rtol=1e-3)

        model.release()


def test_flam():
    compute = GPUFlam2()

    def gpu_calc_flam(**params):
        return compute.run(
            temp_l=params["temp_l"],
            fuel_build_up=params["fuel_build_up"],
            fapar=params["fapar"],
            dry_days=params["dry_days"],
            dryness_method=params["dryness_method"],
            fuel_build_up_method=params["fuel_build_up_method"],
            fapar_factor=params["fapar_factor"],
            fapar_centre=params["fapar_centre"],
            fapar_shape=params["fapar_shape"],
            fuel_build_up_factor=params["fuel_build_up_factor"],
            fuel_build_up_centre=params["fuel_build_up_centre"],
            fuel_build_up_shape=params["fuel_build_up_shape"],
            temperature_factor=params["temperature_factor"],
            temperature_centre=params["temperature_centre"],
            temperature_shape=params["temperature_shape"],
            dry_day_factor=params["dry_day_factor"],
            dry_day_centre=params["dry_day_centre"],
            dry_day_shape=params["dry_day_shape"],
            dry_bal=params["dry_bal"],
            dry_bal_factor=params["dry_bal_factor"],
            dry_bal_centre=params["dry_bal_centre"],
            dry_bal_shape=params["dry_bal_shape"],
            litter_pool=params["litter_pool"],
            litter_pool_factor=params["litter_pool_factor"],
            litter_pool_centre=params["litter_pool_centre"],
            litter_pool_shape=params["litter_pool_shape"],
            include_temperature=params["include_temperature"],
            fapar_weight=params["fapar_weight"],
            dryness_weight=params["dryness_weight"],
            temperature_weight=params["temperature_weight"],
            fuel_weight=params["fuel_weight"],
        )

    for seed in range(10000):
        rng = np.random.default_rng(seed)

        # Use random data and parameters to simulate variability of real nputs.
        params = dict(
            temp_l=rng.random(),
            rhum_l=0.0,
            fuel_l=0.0,
            sm_l=0.0,
            rain_l=0.0,
            fuel_build_up=rng.random(),
            fapar=rng.random(),
            dry_days=rng.random(),
            flammability_method=2,
            dryness_method=rng.integers(1, 3),
            fuel_build_up_method=rng.integers(1, 3),
            fapar_factor=rng.random(),
            fapar_centre=rng.random(),
            fapar_shape=rng.random(),
            fuel_build_up_factor=rng.random(),
            fuel_build_up_centre=rng.random(),
            fuel_build_up_shape=rng.random(),
            temperature_factor=rng.random(),
            temperature_centre=rng.random(),
            temperature_shape=rng.random(),
            dry_day_factor=rng.random(),
            dry_day_centre=rng.random(),
            dry_day_shape=rng.random(),
            dry_bal=rng.random(),
            dry_bal_factor=rng.random(),
            dry_bal_centre=rng.random(),
            dry_bal_shape=rng.random(),
            litter_pool=rng.random(),
            litter_pool_factor=rng.random(),
            litter_pool_centre=rng.random(),
            litter_pool_shape=rng.random(),
            include_temperature=rng.integers(2),
            fapar_weight=rng.random(),
            dryness_weight=rng.random(),
            temperature_weight=rng.random(),
            fuel_weight=rng.random(),
        )

        python_flam = calc_flam(**params)
        metal_flam = gpu_calc_flam(**params)

        assert_allclose(python_flam, metal_flam, atol=1e-8, rtol=1e-4)

    # Test 0 weights case.
    for key in (
        "fapar_weight",
        "dryness_weight",
        "temperature_weight",
        "fuel_weight",
    ):
        params[key] = 0.0

    python_flam = calc_flam(**params)
    metal_flam = gpu_calc_flam(**params)

    assert_allclose(python_flam, 1, atol=1e-8, rtol=1e-4)
    assert_allclose(metal_flam, 1, atol=1e-8, rtol=1e-4)

    compute.release()


def test_checks_mask(model_params):
    for _, params in model_params.items():
        metal_model = GPUBAModel(**params)
        mask = metal_model._gpu_inferno.get_checks_failed_mask()

        python_model = BAModel(**params)
        expected = python_model._get_checks_failed_mask()

        assert np.all(mask == expected)

        metal_model.release()


def test_diagnostics(model_params):
    for _, params in model_params.items():
        metal_model = GPUBAModel(**params)
        metal_diagnostics = metal_model._get_diagnostics()

        python_model = BAModel(**params)
        python_diagnostics = python_model._get_diagnostics()

        assert len(metal_diagnostics) == len(python_diagnostics) == 3

        for i in range(len(metal_diagnostics)):
            # NOTE This very high tolerance probably indicates the unreliability of
            # certain floating point calculations. Therefore, the Python (Numba)
            # version should be preferred.
            assert_allclose(metal_diagnostics[i], python_diagnostics[i], rtol=0.6)

        metal_model.release()


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("index", range(4))
def test_GPUInfernoConsAvgScore(index, seed, model_params):
    rng = np.random.default_rng(seed)

    _params = list(model_params.values())[index]

    params = {
        **_params,
        **dict(
            fapar_weight=rng.random(),
            dryness_weight=rng.random(),
            temperature_weight=rng.random(),
            fuel_weight=rng.random(),
            overall_scale=100 * rng.random(),
            crop_f=rng.random(),
        ),
    }

    # Set up the models.
    score_model = GPUConsAvgScoreBAModel(**params)
    ba_model = BAModel(**params)

    requested = (Metrics.MPD, Metrics.ARCSINH_NME)

    test_scores = score_model.get_scores(requested=requested, **params)
    assert len(test_scores) == 3

    # Compare against the normal scores.
    exp_scores = ba_model.calc_scores(
        model_ba=ba_model.run(**params)["model_ba"], requested=requested
    )["scores"]

    assert_allclose(
        exp_scores["arcsinh_nme"], test_scores["arcsinh_nme"], atol=1e-12, rtol=2e-4
    )
    assert_allclose(exp_scores["mpd"], test_scores["mpd"], atol=1e-12, rtol=4e-6)
    assert_array_equal(exp_scores["mpd_ignored"], test_scores["mpd_ignored"])

    score_model.release()
