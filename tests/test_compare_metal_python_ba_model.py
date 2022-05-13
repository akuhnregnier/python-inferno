# -*- coding: utf-8 -*-

import numpy as np
import pytest

from python_inferno.ba_model import BAModel, GPUBAModel


@pytest.mark.parametrize(
    "exp_key, expected_diffs",
    [
        (
            "dry_Dry_Day__fuel_Antec_NPP",
            {
                "np.abs(np.mean(diffs))": 3.5e-16,
                "np.mean(np.abs(diffs))": 5.1e-16,
                "np.max(np.abs(diffs))": 4.2e-13,
            },
        ),
        (
            "dry_Dry_Day__fuel_Leaf_Litter_Pool",
            {
                "np.abs(np.mean(diffs))": 2.4e-15,
                "np.mean(np.abs(diffs))": 3.1e-15,
                "np.max(np.abs(diffs))": 4.6e-13,
            },
        ),
        (
            "dry_VPD_Precip__fuel_Antec_NPP",
            {
                "np.abs(np.mean(diffs))": 4.9e-15,
                "np.mean(np.abs(diffs))": 9.2e-15,
                "np.max(np.abs(diffs))": 5.1e-13,
            },
        ),
        (
            "dry_VPD_Precip__fuel_Leaf_Litter_Pool",
            {
                "np.abs(np.mean(diffs))": 1.5e-16,
                "np.mean(np.abs(diffs))": 8.1e-16,
                "np.max(np.abs(diffs))": 1.5e-13,
            },
        ),
    ],
)
def test_GPUBAModel(model_params, exp_key, expected_diffs):
    _params = model_params[exp_key]

    params = {
        **dict(
            fapar_weight=1,
            dryness_weight=1,
            temperature_weight=1,
            fuel_weight=1,
        ),
        **_params,
    }

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
    assert np.allclose(diffs, 0, atol=1e-12, rtol=1e-9)
