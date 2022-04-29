# -*- coding: utf-8 -*-
import pickle

import numpy as np
import pytest

from python_inferno.ba_model import BAModel, GPUBAModel

from . import TEST_DATA_DIR


@pytest.fixture
def model_params():
    with (TEST_DATA_DIR / "best_params.pkl").open("rb") as f:
        params_dict = pickle.load(f)

    return params_dict


@pytest.mark.parametrize(
    "exp_key, expected_diffs",
    [
        (
            "dry_Dry_Day__fuel_Antec_NPP",
            {
                "np.abs(np.mean(diffs))": 3.5e-16,
                "np.mean(np.abs(diffs))": 4.0e-16,
                "np.max(np.abs(diffs))": 3.1e-13,
            },
        ),
        (
            "dry_Dry_Day__fuel_Leaf_Litter_Pool",
            {
                "np.abs(np.mean(diffs))": 2.4e-15,
                "np.mean(np.abs(diffs))": 2.6e-15,
                "np.max(np.abs(diffs))": 3.9e-13,
            },
        ),
        (
            "dry_VPD_Precip__fuel_Antec_NPP",
            {
                "np.abs(np.mean(diffs))": 4.5e-15,
                "np.mean(np.abs(diffs))": 7.8e-15,
                "np.max(np.abs(diffs))": 4.8e-13,
            },
        ),
        (
            "dry_VPD_Precip__fuel_Leaf_Litter_Pool",
            {
                "np.abs(np.mean(diffs))": 2.4e-17,
                "np.mean(np.abs(diffs))": 4.8e-17,
                "np.max(np.abs(diffs))": 1.5e-13,
            },
        ),
    ],
)
def test_GPUBAModel(model_params, exp_key, expected_diffs):
    params = model_params[exp_key]

    model_bas = {}

    for name, model_class in [
        ("python", BAModel),
        ("metal", GPUBAModel),
    ]:
        # `**params` is used twice here because the functions simply use the
        # kwargs they require, ignoring the rest.
        ba_model = model_class(**params)

        model_bas[name] = ba_model.run(**params)["model_ba"]

        if name == "metal":
            ba_model._gpu_inferno.release()

    diffs = model_bas["python"] - model_bas["metal"]

    assert np.abs(np.mean(diffs)) <= expected_diffs["np.abs(np.mean(diffs))"]
    assert np.mean(np.abs(diffs)) <= expected_diffs["np.mean(np.abs(diffs))"]
    assert np.max(np.abs(diffs)) <= expected_diffs["np.max(np.abs(diffs))"]
