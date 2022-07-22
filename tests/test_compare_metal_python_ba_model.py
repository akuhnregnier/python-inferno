# -*- coding: utf-8 -*-

import numpy as np
import pytest
from numpy.testing import assert_allclose

from python_inferno.ba_model import BAModel, GPUBAModel


@pytest.mark.parametrize(
    "exp_key, expected_diffs",
    [
        (
            "dry_Dry_Day__fuel_Antec_NPP",
            {
                "allclose_params": {"rtol": 1e-4, "atol": 2e-11},
                "max zero_atol": 1e-14,
            },
        ),
        (
            "dry_Dry_Day__fuel_Leaf_Litter_Pool",
            {
                "allclose_params": {"rtol": 1e-4, "atol": 1e-11},
                "max zero_atol": 1e-12,
            },
        ),
        (
            "dry_VPD_Precip__fuel_Antec_NPP",
            {
                "allclose_params": {"rtol": 1e-4, "atol": 1e-12},
                "max zero_atol": 1e-14,
            },
        ),
        (
            "dry_VPD_Precip__fuel_Leaf_Litter_Pool",
            {
                "allclose_params": {"rtol": 1.1e-5, "atol": 1e-10},
                "max zero_atol": 7e-13,
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

    python_zeros = model_bas["python"] < 1e-30
    metal_zeros = model_bas["metal"] < 1e-30

    zero_mask = python_zeros | metal_zeros

    python_nonzero = model_bas["python"][~zero_mask]
    metal_nonzero = model_bas["metal"][~zero_mask]

    max_zero_atol = max(
        np.max(np.abs(model_bas["metal"][python_zeros])),
        np.max(np.abs(model_bas["python"][metal_zeros])),
    )

    assert_allclose(metal_nonzero, python_nonzero, **expected_diffs["allclose_params"])
    assert max_zero_atol <= expected_diffs["max zero_atol"]
