#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from python_inferno.configuration import land_pts
from python_inferno.data import load_jules_lats_lons
from python_inferno.model_params import get_model_params
from python_inferno.sensitivity_analysis import (
    GPUBAModelSensitivityAnalysis,
    LandChecksFailed,
)

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    jules_lats, jules_lons = load_jules_lats_lons()

    record_dir = Path(os.environ["EPHEMERAL"]) / "opt_record"
    df, method_iter = get_model_params(
        record_dir=record_dir, progress=True, verbose=False
    )

    for (
        dryness_method,
        fuel_build_up_method,
        df_sel,
        min_index,
        min_loss,
        params,
        exp_name,
        exp_key,
    ) in islice(
        method_iter(),
        0,
        # XXX
        1,
    ):
        logger.info(exp_name)

        sa = GPUBAModelSensitivityAnalysis(params=params, exponent=8)

        sobol_sis = {}
        for i in tqdm(range(land_pts), desc="land"):
            try:
                sobol_sis[i] = sa.sobol_sis(land_index=i, verbose=False)
            except LandChecksFailed:
                pass

        group_names = list(next(iter(sobol_sis.values())).to_df()[0].index.values)

        data = {}
        for name in group_names:
            vals = [si.to_df()[0]["ST"][name] for si in sobol_sis.values()]
            data[name] = dict(
                mean=np.nanmean(vals),
                std=np.nanstd(vals),
            )
        df = pd.DataFrame(data).T
        print(df)
