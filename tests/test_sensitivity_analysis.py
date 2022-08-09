# -*- coding: utf-8 -*-
import os
from itertools import islice
from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from python_inferno.hyperopt import get_space_template
from python_inferno.iter_opt import ALWAYS_OPTIMISED, IGNORED
from python_inferno.model_params import get_model_params
from python_inferno.sensitivity_analysis import SAMetric
from python_inferno.sobol_sa import BAModelSobolSA, GPUBAModelSobolSA


@pytest.mark.parametrize("param_index", range(4))
@pytest.mark.parametrize("test_type", ["data", "params"])
@pytest.mark.parametrize(
    "land_index",
    [
        pytest.param(l, marks=pytest.mark.slow) if i > 3 else l
        for i, l in enumerate(
            [
                0,
                100,
                200,
                300,
                500,
                1000,
                2000,
                3000,
                3200,
                3500,
                4000,
                5000,
                6500,
                7000,
            ]
        )
    ],
)
@pytest.mark.parametrize("exponent", [6, pytest.param(8, marks=pytest.mark.slow)])
def test_sa_versions(param_index, test_type, land_index, exponent):
    record_dir = Path(os.environ["EPHEMERAL"]) / "opt_record"
    df, method_iter = get_model_params(
        record_dir=record_dir, progress=True, verbose=False
    )

    (
        dryness_method,
        fuel_build_up_method,
        df_sel,
        min_index,
        min_loss,
        params,
        exp_name,
        exp_key,
    ) = list(islice(method_iter(), param_index, param_index + 1))[0]
    assert int(params["include_temperature"]) == 1

    if test_type == "params":
        space_template = get_space_template(
            dryness_method=dryness_method,
            fuel_build_up_method=fuel_build_up_method,
            include_temperature=int(params["include_temperature"]),
        )

        param_names = [
            key for key in space_template if key not in ALWAYS_OPTIMISED.union(IGNORED)
        ]
        if "crop_f" in param_names:
            param_names.remove("crop_f")

        data_variables = param_names
    else:
        data_variables = None

    sa_params = dict(
        params=params,
        exponent=exponent,
        data_variables=data_variables,
        fuel_build_up_method=int(params["fuel_build_up_method"]),
        dryness_method=int(params["dryness_method"]),
    )

    sa = BAModelSobolSA(**sa_params)
    gpu_sa = GPUBAModelSobolSA(**sa_params)

    si = sa.sobol_sis(land_index=land_index)
    gpu_si = gpu_sa.sensitivity_analysis(land_index=land_index)

    for metric in SAMetric:
        if metric not in si or metric not in gpu_si:
            assert metric not in gpu_si and metric not in si
        else:
            assert_allclose(
                gpu_si[metric]["S1"], si[metric]["S1"], rtol=1e-4, atol=5e-6
            )
