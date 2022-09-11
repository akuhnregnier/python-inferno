#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from loguru import logger
from wildfires.analysis import cube_plotting

from python_inferno.ba_model import GPUBAModel
from python_inferno.configuration import land_pts
from python_inferno.cv import get_ba_cv_splits
from python_inferno.data import load_jules_lats_lons
from python_inferno.model_params import get_model_params

if __name__ == "__main__":
    mpl.rc_file(Path(__file__).absolute().parent / "matplotlibrc")
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    save_dir = Path("~/tmp/ba-histograms-cat").expanduser()
    save_dir.mkdir(parents=False, exist_ok=True)

    df, method_iter = get_model_params(progress=False, verbose=False)

    method_data = next(method_iter())
    params = method_data[5]
    ba_model = GPUBAModel(**params)
    gfed_ba = ba_model.mon_avg_gfed_ba_1d
    assert not np.any(gfed_ba.mask)

    gfed_ba = np.ma.getdata(ba_model.mon_avg_gfed_ba_1d)

    plt.figure()
    plt.hist(gfed_ba.ravel(), bins=50)
    plt.yscale("log")
    plt.savefig(save_dir / "raw_ba.png")
    plt.close()

    train_grids, test_grids, test_grid_map = get_ba_cv_splits(gfed_ba)

    # Visualise splits on a map.

    jules_lats, jules_lons = load_jules_lats_lons()

    split_data_1d = np.zeros(land_pts, dtype=np.float64)
    for grid_i, split_i in test_grid_map.items():
        split_data_1d[grid_i] = split_i

    split_data_1d_cube = get_1d_data_cube(
        split_data_1d, lats=jules_lats, lons=jules_lons
    )
    split_data_2d_cube = cube_1d_to_2d(split_data_1d_cube)

    fig = plt.figure(figsize=(12, 5))
    cube_plotting(
        split_data_2d_cube,
        title="CV Splits",
        nbins=9,
        boundaries=np.arange(6) - 0.5,
        cmap="tab10",
        fig=fig,
    )
    fig.savefig(save_dir / "split_map.png")
    plt.close(fig)

    cat_boundaries = [-1, 1e-4, 1e-2, 0.1, 1]

    for test_grid in test_grids:
        sel = gfed_ba[:, test_grid].ravel()
        print(sel.size)
        print(np.bincount(np.digitize(sel, cat_boundaries)))
