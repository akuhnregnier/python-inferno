# -*- coding: utf-8 -*-
import pytest
from numpy.testing import assert_allclose

from python_inferno.sensitivity_analysis import (
    BAModelSensitivityAnalysis,
    GPUBAModelSensitivityAnalysis,
)


@pytest.mark.parametrize("param_index", range(4))
@pytest.mark.parametrize(
    "land_index",
    [0, 100, 200, 300, 500, 1000, 2000, 3000, 3200, 3500, 4000, 5000, 6500, 7000],
)
@pytest.mark.parametrize("exponent", [6, pytest.param(8, marks=pytest.mark.slow)])
def test_sa_versions(model_params, param_index, land_index, exponent):
    params = list(model_params.values())[param_index]

    sa = BAModelSensitivityAnalysis(params=params, exponent=exponent)
    gpu_sa = GPUBAModelSensitivityAnalysis(params=params, exponent=exponent)

    si = sa.sobol_sis(land_index=land_index)
    gpu_si = gpu_sa.sobol_sis(land_index=land_index)

    assert_allclose(gpu_si["S1"], si["S1"], rtol=1e-4, atol=5e-6)
