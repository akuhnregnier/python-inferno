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
from python_inferno.hyperopt import get_space_template
from python_inferno.iter_opt import ALWAYS_OPTIMISED, IGNORED
from python_inferno.model_params import get_model_params
from python_inferno.sensitivity_analysis import sis_calc


def analyse_sobol_sis(*, sobol_sis, save_dir):
    jules_lats, jules_lons = load_jules_lats_lons()
    group_names = list(next(iter(sobol_sis.values())).to_df()[0].index.values)

    data = {}
    for name in group_names:
        vals = [si.to_df()[0]["ST"][name] for si in sobol_sis.values()]
        data[name] = dict(
            mean=np.nanmean(vals),
            std=np.nanstd(vals),
        )
    df = pd.DataFrame(data).T
    df["ratio"] = df["std"] / df["mean"]
    print(df.sort_values("mean", ascending=False))

    # Plotting.

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
        plt.close(fig)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    sa_plot_dir = Path("~/tmp/sa").expanduser()
    sa_plot_dir.mkdir(parents=False, exist_ok=True)

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
    ) in islice(method_iter(), 0, None):
        assert int(params["include_temperature"]) == 1

        logger.info(exp_name)

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

        save_dir = sa_plot_dir / exp_key
        save_dir.mkdir(parents=False, exist_ok=True)

        for data_variables in [param_names, None]:
            if data_variables is None:
                save_dir2 = save_dir / "data"
                save_dir2.mkdir(parents=False, exist_ok=True)
            else:
                save_dir2 = save_dir / "params"
                save_dir2.mkdir(parents=False, exist_ok=True)

            sobol_sis = sis_calc(
                params=params,
                land_points=list(range(land_pts)),
                data_variables=data_variables,
                # NOTE Derive parameter uncertainty ranges from the existing set of runs.
                df_sel=df_sel,
                fuel_build_up_method=fuel_build_up_method,
                dryness_method=dryness_method,
            )
            analyse_sobol_sis(sobol_sis=sobol_sis, save_dir=save_dir2)
