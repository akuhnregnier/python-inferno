# -*- coding: utf-8 -*-

import numpy as np

from python_inferno.ba_model import BAModel


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
