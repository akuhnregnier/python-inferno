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

from python_inferno.data import load_jules_lats_lons
from python_inferno.model_params import get_model_params
from python_inferno.sensitivity_analysis import BAModelSensitivityAnalysis

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

        sa = BAModelSensitivityAnalysis(params=params)

        sobol_sis = [sa.sobol_sis(land_index=i) for i in tqdm(range(3), desc="land")]

        group_names = list(sobol_sis[0].to_df()[0].index.values)

        data = {}
        for name in group_names:
            vals = [si.to_df()[0]["ST"][name] for si in sobol_sis]
            data[name] = dict(
                mean=np.nanmean(vals),
                std=np.nanstd(vals),
            )
        df = pd.DataFrame(data).T
        print(df)
