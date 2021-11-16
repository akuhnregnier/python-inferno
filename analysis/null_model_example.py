#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from python_inferno.metrics import null_model_analysis

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    reference_data = rng.random((12, 200, 300))

    null_model_analysis(
        reference_data=reference_data,
        comp_data=dict(
            x0=reference_data + 0.2 * rng.random(reference_data.shape),
            x1=reference_data + 0.4 * rng.random(reference_data.shape),
        ),
        rng=rng,
    )
