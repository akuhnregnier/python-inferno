# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_allclose

from python_inferno.metrics import mpd, nme, nmse


def test_mse():
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


def test_mpd():
    obs = np.random.default_rng(0).random((12, 1000))
    assert_allclose(mpd(obs=obs, pred=obs), 0)

    pred = np.random.default_rng(1).random(obs.shape)
    mpd1 = mpd(obs=obs, pred=pred)
    assert mpd1 > 0

    mpd2 = mpd(obs=obs, pred=(pred + obs) / 2.0)
    assert mpd2 < mpd1


def test_mpd_sin():
    obs = np.sin(np.linspace(0, 1, 12) * np.pi)
    mpd_vals = np.array(
        [
            mpd(obs=obs.reshape(12, 1), pred=np.roll(obs, i).reshape(12, 1))
            for i in range(13)
        ]
    )
    comp = np.linspace(0, 1, 7)
    assert_allclose(mpd_vals, np.append(comp, comp[:6][::-1]))


def test_mpd_zeros():
    obs = np.random.default_rng(0).random((12, 1000))

    # Set all values at a certain location to 0.
    obs[:, 0] = 0.0
    assert_allclose(mpd(obs=obs, pred=obs), 0)

    pred = np.random.default_rng(1).random(obs.shape)
    assert mpd(obs=obs, pred=pred) < 1

    assert mpd(obs=obs, pred=pred, return_ignored=True)[1] == 1

    pred[:, 0] = 0.0
    assert mpd(obs=obs, pred=pred, return_ignored=True)[1] == 1

    pred[:, 2] = 0.0
    assert mpd(obs=obs, pred=pred, return_ignored=True)[1] == 2
