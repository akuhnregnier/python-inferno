# -*- coding: utf-8 -*-
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from python_inferno.metrics import (
    calculate_factor,
    calculate_phase,
    loghist,
    mpd,
    nme,
    nme_simple,
    nmse,
)
from python_inferno.py_gpu_inferno import GPUCalculateMPD, GPUCalculatePhase, cpp_nme


def test_phase_calculation():
    x = np.random.default_rng(0).random((12, 300), dtype=np.float32)
    gpu_phase = GPUCalculatePhase(x.shape[1])
    assert_allclose(gpu_phase.run(x), calculate_phase(x), rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("nme_func", [nme, nme_simple, cpp_nme])
def test_nme(nme_func):
    obs = np.random.default_rng(0).random((1000,))
    assert_allclose(nme(obs=obs, pred=obs), 0)

    pred = np.random.default_rng(1).random(obs.shape)
    nme1 = nme(obs=obs, pred=pred)
    assert nme1 > 0

    nme2 = nme(obs=obs, pred=(pred + obs) / 2.0)
    assert nme2 < nme1

    assert_allclose(nme(obs=obs, pred=np.mean(obs)), 1)

    # The shape should not matter.
    assert_allclose(nme(obs=obs, pred=np.zeros_like(obs) + np.mean(obs)), 1)


@pytest.mark.parametrize("nme_func", [nme_simple, cpp_nme])
def test_nme_implementations(nme_func):
    obs = np.random.default_rng(0).random((12, 7771))
    pred = np.random.default_rng(1).random(obs.shape)
    ref_nme = nme(obs=obs, pred=pred)
    nme2 = nme_func(obs=obs, pred=pred)
    assert_allclose(ref_nme, nme2, atol=4e-6, rtol=4e-6)


def test_nme_benchmark(benchmark):
    obs = np.random.default_rng(0).random((12, 7771))
    pred = np.random.default_rng(1).random(obs.shape)
    nme1 = benchmark(nme, obs=obs, pred=pred)
    assert nme1 > 0


def test_nme_simple_benchmark(benchmark):
    obs = np.random.default_rng(0).random((12, 7771))
    pred = np.random.default_rng(1).random(obs.shape)
    nme1 = benchmark(nme_simple, obs=obs, pred=pred)
    assert nme1 > 0


def test_cpp_nme_benchmark(benchmark):
    obs = np.random.default_rng(0).random((12, 7771))
    pred = np.random.default_rng(1).random(obs.shape)
    nme1 = benchmark(cpp_nme, obs=obs, pred=pred)
    assert nme1 > 0


def test_nmse():
    obs = np.random.default_rng(0).random((1000,))
    assert_allclose(nmse(obs=obs, pred=obs), 0)

    pred = np.random.default_rng(1).random(obs.shape)
    nmse1 = nmse(obs=obs, pred=pred)
    assert nmse1 > 0

    nmse2 = nmse(obs=obs, pred=(pred + obs) / 2.0)
    assert nmse2 < nmse1

    assert_allclose(nmse(obs=obs, pred=np.mean(obs)), 1)

    # The shape should not matter.
    assert_allclose(nmse(obs=obs, pred=np.zeros_like(obs) + np.mean(obs)), 1)


def test_mpd(benchmark):
    test_mpd = GPUCalculateMPD(7771).run

    obs = np.random.default_rng(0).random((12, 7771))
    assert_allclose(test_mpd(obs=obs, pred=obs), 0)

    pred = np.random.default_rng(1).random(obs.shape)
    mpd1 = test_mpd(obs=obs, pred=pred)
    assert mpd1 > 0

    GPUCalculatePhase(obs.shape[1])

    mpd2 = benchmark(
        test_mpd,
        obs=obs,
        pred=(pred + obs) / 2.0,
    )
    assert mpd2 < mpd1


@pytest.mark.parametrize("test_mpd", [mpd, GPUCalculateMPD(1).run])
def test_mpd_sin(test_mpd):
    obs = np.sin(np.linspace(0, 1, 12) * np.pi)
    mpd_vals = np.array(
        [
            test_mpd(obs=obs.reshape(12, 1), pred=np.roll(obs, i).reshape(12, 1))
            for i in range(13)
        ]
    )
    comp = np.linspace(0, 1, 7)
    assert_allclose(mpd_vals, np.append(comp, comp[:6][::-1]), atol=1e-7)


@pytest.mark.parametrize("test_mpd", [mpd, GPUCalculateMPD(1000).run])
def test_mpd_zeros(test_mpd):
    obs = np.random.default_rng(0).random((12, 1000))

    # Set all values at a certain location to 0.
    obs[:, 0] = 0.0
    assert_allclose(test_mpd(obs=obs, pred=obs), 0)

    pred = np.random.default_rng(1).random(obs.shape)
    assert test_mpd(obs=obs, pred=pred) < 1

    assert test_mpd(obs=obs, pred=pred, return_ignored=True)[1] == 1

    pred[:, 0] = 0.0
    assert test_mpd(obs=obs, pred=pred, return_ignored=True)[1] == 1

    pred[:, 2] = 0.0
    assert test_mpd(obs=obs, pred=pred, return_ignored=True)[1] == 2


def test_loghist():
    obs = np.random.default_rng(0).normal(0.5, 0.2, (1000,))

    bndhist = partial(loghist, edges=np.linspace(0, 1, 10))

    assert_allclose(bndhist(obs=obs, pred=obs), 0)

    pred = np.random.default_rng(1).random(obs.shape)
    err1 = bndhist(obs=obs, pred=pred)
    assert err1 > 0

    err2 = bndhist(
        obs=obs,
        pred=np.concatenate(
            (
                # Get elements from both arrays.
                obs[: obs.size // 2],
                pred[: obs.size // 2],
            )
        ),
    )
    assert err2 < err1

    assert_allclose(bndhist(obs=obs, pred=obs), 0)


def test_calculate_factor():
    y_true = np.random.default_rng(0).random(100)
    y_pred = y_true / 2.0

    assert np.isclose(calculate_factor(y_true=y_true, y_pred=y_pred), 2.0)
