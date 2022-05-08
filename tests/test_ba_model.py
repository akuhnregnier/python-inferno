# -*- coding: utf-8 -*-

import numpy as np

from python_inferno.ba_model import BAModel
from python_inferno.metrics import Metrics


def test_BAModel(params_model_ba):
    for params, expected_model_ba in params_model_ba:
        assert np.allclose(
            BAModel(**params).run(
                **{
                    **dict(
                        fapar_weight=1,
                        dryness_weight=1,
                        temperature_weight=1,
                        fuel_weight=1,
                    ),
                    **params,
                }
            )["model_ba"],
            expected_model_ba["python"],
            atol=1e-12,
            rtol=1e-7,
        )


def test_calculate_scores_benchmark(benchmark, model_params):
    params = next(iter(model_params.values()))  # Get first value.
    ba_model = BAModel(**params)
    model_ba = ba_model.run(
        **{
            **dict(
                fapar_weight=1,
                dryness_weight=1,
                temperature_weight=1,
                fuel_weight=1,
            ),
            **params,
        }
    )["model_ba"]

    benchmark(
        ba_model.calc_scores,
        model_ba=model_ba,
        requested=(Metrics.MPD, Metrics.ARCSINH_NME),
    )
