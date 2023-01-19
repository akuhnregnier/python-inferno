#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import islice
from pathlib import Path

import matplotlib.pyplot as plt
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from wildfires.analysis import cube_plotting

from python_inferno.ba_model import BAModel
from python_inferno.data import get_data_yearly_stdev, load_jules_lats_lons
from python_inferno.model_params import get_model_params
from python_inferno.plotting import use_style

if __name__ == "__main__":
    save_dir = Path("~/tmp/yearly_iav").expanduser()
    save_dir.mkdir(parents=False, exist_ok=True)

    use_style()

    df, method_iter = get_model_params()
    _n = 3

    (
        dryness_method,
        fuel_build_up_method,
        df_sel,
        min_index,
        min_loss,
        _params,
        exp_name,
        exp_key,
    ) = next(islice(method_iter(), _n, _n + 1))

    _ba_model = BAModel(**_params)
    yearly_stdev = get_data_yearly_stdev(
        params={
            "litter_pool": {
                "litter_tc": _ba_model.disc_params["litter_tc"],
                "leaf_f": _ba_model.disc_params["leaf_f"],
            },
            "grouped_dry_bal": {
                "rain_f": _ba_model.disc_params["rain_f"],
                "vpd_f": _ba_model.disc_params["vpd_f"],
            },
        }
    )

    for key, data_1d in yearly_stdev.items():
        if data_1d.ndim > 1:
            assert data_1d.ndim == 2
            assert data_1d.shape[1] == 7771
            data_1d = data_1d.mean(axis=0)

        jules_lats, jules_lons = load_jules_lats_lons()
        _1d_cube = get_1d_data_cube(data_1d, lats=jules_lats, lons=jules_lons)
        _2d_cube = cube_1d_to_2d(_1d_cube)
        assert _2d_cube.ndim == 2

        fig = cube_plotting(
            cube=_2d_cube,
            title=key,
            colorbar_kwargs=dict(label="Yearly IAV"),
            fig=plt.figure(figsize=(5, 2)),
        )
        fig.savefig(save_dir / f"{key}_yearly_iav")
        plt.close(fig)
