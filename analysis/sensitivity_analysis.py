#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from itertools import islice
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from loguru import logger
from tqdm import tqdm
from wildfires.analysis import cube_plotting

from python_inferno.configuration import land_pts
from python_inferno.data import load_jules_lats_lons
from python_inferno.model_params import get_model_params
from python_inferno.sensitivity_analysis import sis_calc

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    sa_plot_dir = Path("~/tmp/sa").expanduser()
    sa_plot_dir.mkdir(parents=False, exist_ok=True)

    jules_lats, jules_lons = load_jules_lats_lons()

    record_dir = Path(os.environ["EPHEMERAL"]) / "opt_record"
    df, method_iter = get_model_params(
        record_dir=record_dir, progress=True, verbose=False
    )
    jules_lats, jules_lons = load_jules_lats_lons()

    for (
        dryness_method,
        fuel_build_up_method,
        df_sel,
        min_index,
        min_loss,
        params,
        exp_name,
        exp_key,
    ) in islice(method_iter(), 0, None):
        logger.info(exp_name)

        sobol_sis = sis_calc(params=params, land_points=list(range(land_pts)))

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

        # Plotting.

        save_dir = sa_plot_dir / exp_key
        save_dir.mkdir(parents=False, exist_ok=True)

        for name in tqdm(group_names, desc="plotting"):
            data = np.ma.MaskedArray(np.zeros(land_pts), mask=True)
            for land_i, val in sobol_sis.items():
                st_val = val.to_df()[0]["ST"][name]
                if not np.isnan(st_val):
                    data[land_i] = st_val

            if np.all(np.ma.getmaskarray(data)):
                logger.warning(f"{name} all masked!")
                continue

            if np.all(np.isclose(data, 0)):
                logger.info(f"{name} all close to 0.")
                continue

            cube_2d = cube_1d_to_2d(
                get_1d_data_cube(data, lats=jules_lats, lons=jules_lons)
            )

            fig = plt.figure(figsize=(6, 4), dpi=200)
            cube_plotting(cube_2d, title=name, fig=fig)
            fig.savefig(str(save_dir / name))
