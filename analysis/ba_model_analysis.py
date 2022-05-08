#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from operator import itemgetter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from loguru import logger
from tqdm import tqdm

from python_inferno.ba_model import BAModel, Status, calculate_scores
from python_inferno.cache import cache
from python_inferno.configuration import n_total_pft, npft
from python_inferno.data import load_data, load_jules_lats_lons
from python_inferno.metrics import null_model_analysis
from python_inferno.model_params import get_model_params
from python_inferno.plotting import plotting
from python_inferno.utils import (
    ConsMonthlyAvg,
    PartialDateTime,
    get_apply_mask,
    get_ba_mask,
    memoize,
    temporal_processing,
)


def frac_weighted_mean(*, data, frac):
    assert len(data.shape) == 3, "Need time, PFT, and space coords."
    assert data.shape[1] in (npft, n_total_pft)
    assert frac.shape[1] == n_total_pft

    return np.sum(data * frac[:, : data.shape[1]], axis=1) / np.sum(
        frac[:, : data.shape[1]], axis=1
    )


@cache(dependencies=[load_data, temporal_processing])
def get_processed_climatological_jules_ba():
    logger.debug("start data")
    (
        _,
        _,
        _,
        _,
        frac,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        jules_ba_gb,
        _,
        jules_time_coord,
        _,
        _,
        climatology_output,
    ) = load_data(
        filenames=(
            tuple(
                [
                    str(Path(s).expanduser())
                    for s in [
                        "~/tmp/climatology6.nc",
                    ]
                ]
            )
        ),
        N=None,
        output_timesteps=4,
        climatology_dates=(PartialDateTime(2000, 1), PartialDateTime(2016, 12)),
    )
    logger.debug("Got data")

    data_dict = dict(
        frac=frac,
        jules_ba_gb=jules_ba_gb.data,
    )

    logger.debug("Populated data_dict")

    assert jules_time_coord.cell(-1).point.month == 12
    last_year = jules_time_coord.cell(-1).point.year
    for start_i in range(jules_time_coord.shape[0]):
        if jules_time_coord.cell(start_i).point.year == last_year:
            break
    else:
        raise ValueError("Target year not encountered.")

    # Trim the data and temporal coord such that the data spans a single year.
    jules_time_coord = jules_time_coord[start_i:]
    for data_name in data_dict:
        data_dict[data_name] = data_dict[data_name][start_i:]

    assert (
        jules_time_coord.cell(0).point.year == jules_time_coord.cell(-1).point.year
        and jules_time_coord.cell(0).point.month == 1
        and jules_time_coord.cell(-1).point.month == 12
        and jules_time_coord.shape[0] >= 12
    )
    return data_dict, jules_time_coord


def plot_param_histograms(df_sel, exp_name, hist_save_dir):
    for col in [col for col in df_sel.columns if col != "loss"]:
        if df_sel[col].isna().all():
            continue

        plt.figure()
        plt.plot(df_sel[col], df_sel["loss"], linestyle="", marker="o", alpha=0.6)
        plt.xlabel(col)
        plt.ylabel("loss")
        plt.title(exp_name)
        if col in ("rain_f", "vpd_f", "litter_tc", "leaf_f"):
            plt.xscale("log")
        plt.savefig(hist_save_dir / f"{col}.png")
        plt.close()


if __name__ == "__main__":
    mpl.rc_file(Path(__file__).absolute().parent / "matplotlibrc")
    save_dir = Path("~/tmp/ba-model-analysis/").expanduser()
    save_dir.mkdir(exist_ok=True, parents=False)
    plotting = partial(plotting, save_dir=save_dir, regions="PNV")

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    jules_lats, jules_lons = load_jules_lats_lons()

    # To prevent memory accumulation during repeated calculations below.
    memoize.active = False

    # XXX - 'opt_record_bak' vs. 'opt_record'
    record_dir = Path(os.environ["EPHEMERAL"]) / "opt_record_bak"
    df, method_iter = get_model_params(
        record_dir=record_dir, progress=True, verbose=True
    )

    hist_bins = 50

    plot_data = dict()
    plot_prog = tqdm(desc="Generating plot data", total=6)

    executor = ProcessPoolExecutor(max_workers=10)
    futures = []

    for (
        dryness_method,
        fuel_build_up_method,
        df_sel,
        min_index,
        min_loss,
        params,
        exp_name,
        exp_key,
    ) in method_iter():
        logger.info(exp_name)
        logger.info(exp_key)

        hist_save_dir = save_dir / "parameter_histograms" / exp_key
        hist_save_dir.mkdir(exist_ok=True, parents=True)

        logger.info("Plotting histograms.")
        futures.append(
            executor.submit(plot_param_histograms, df_sel, exp_name, hist_save_dir)
        )

        logger.info("Predicting BA")
        ba_model = BAModel(**params)
        model_ba, mon_avg_gfed_ba_1d = itemgetter("model_ba", "mon_avg_gfed_ba_1d")(
            ba_model.run(**params)
        )
        scores, calc_factors = itemgetter("scores", "calc_factors")(
            ba_model.score(model_ba=model_ba)
        )

        model_ba *= calc_factors["adj_factor"]

        ba_mask_1d = get_ba_mask(mon_avg_gfed_ba_1d)
        apply_ba_mask_1d = get_apply_mask(ba_mask_1d)

        orig_mon_avg_gfed_ba_1d = mon_avg_gfed_ba_1d
        mon_avg_gfed_ba_1d = apply_ba_mask_1d(mon_avg_gfed_ba_1d)
        model_ba = apply_ba_mask_1d(model_ba)

        gc.collect()

        model_ba_1d = get_1d_data_cube(model_ba, lats=jules_lats, lons=jules_lons)
        logger.info("Getting 2D cube")
        model_ba_2d = cube_1d_to_2d(model_ba_1d)

        gc.collect()

        plot_data[exp_name] = dict(
            exp_key=exp_key,
            raw_data=np.ma.getdata(model_ba)[~np.ma.getmaskarray(model_ba)],
            model_ba_2d_data=model_ba_2d.data,
            hist_bins=hist_bins,
            arcsinh_adj_factor=calc_factors["arcsinh_adj_factor"],
            arcsinh_factor=calc_factors["arcsinh_factor"],
            scores=scores,
            data_params=params,
        )
        gc.collect()
        plot_prog.update()

    # TODO The use of `mon_avg_gfed_ba_1d` here makes the results dependent on the
    # averaging used to derive it (depends on `average_samples`). This should also be
    # true for other scripts following a similar structure, e.g.
    # `multi_gam_analysis.py`!

    # GFED4
    reference_obs = cube_1d_to_2d(
        get_1d_data_cube(mon_avg_gfed_ba_1d, lats=jules_lats, lons=jules_lons)
    ).data
    plot_data["GFED4"] = dict(
        raw_data=np.ma.getdata(mon_avg_gfed_ba_1d)[
            ~np.ma.getmaskarray(mon_avg_gfed_ba_1d)
        ],
        model_ba_2d_data=reference_obs,
        hist_bins=hist_bins,
        # NOTE: Assuming arcsinh_factor is constant across all experiments, which
        # should be true.
        arcsinh_adj_factor=1.0,
        arcsinh_factor=calc_factors["arcsinh_factor"],
    )
    plot_prog.update()

    # Old INFERNO BA.
    data_dict, jules_time_coord = get_processed_climatological_jules_ba()
    jules_ba_gb = data_dict.pop("jules_ba_gb")
    scores, status, avg_jules_ba, calc_factors = calculate_scores(
        model_ba=jules_ba_gb,
        cons_monthly_avg=ConsMonthlyAvg(jules_time_coord),
        mon_avg_gfed_ba_1d=mon_avg_gfed_ba_1d,
    )
    assert status is Status.SUCCESS, "Score calculation failed!"

    avg_jules_ba *= calc_factors["adj_factor"]

    plot_data["Old INFERNO BA"] = dict(
        raw_data=np.ma.getdata(avg_jules_ba)[~np.ma.getmaskarray(avg_jules_ba)],
        model_ba_2d_data=get_apply_mask(reference_obs.mask)(
            cube_1d_to_2d(
                get_1d_data_cube(avg_jules_ba, lats=jules_lats, lons=jules_lons)
            ).data
        ),
        hist_bins=hist_bins,
        arcsinh_adj_factor=calc_factors["arcsinh_adj_factor"],
        arcsinh_factor=calc_factors["arcsinh_factor"],
        scores=scores,
        # TODO Which params to use (combination of all?) - call again with actual
        # old INFERNO params to get corresponding data - is this supported?
        data_params=params,
    )
    plot_prog.update()
    plot_prog.close()

    for exp_name, data in plot_data.items():
        futures.append(
            executor.submit(
                plotting,
                exp_name=exp_name,
                ref_2d_data=(reference_obs if exp_name != "GFED4" else None),
                **data,
            )
        )

    # TODO For error calculations, do not use the version with the low BA mask?
    orig_reference_obs = cube_1d_to_2d(
        get_1d_data_cube(orig_mon_avg_gfed_ba_1d, lats=jules_lats, lons=jules_lons)
    ).data
    null_model_analysis(
        reference_data=orig_reference_obs,
        comp_data={
            key: vals["model_ba_2d_data"]
            for key, vals in plot_data.items()
            if key != "GFED4"
        },
        rng=np.random.default_rng(0),
        save_dir=save_dir,
    )

    for f in tqdm(
        as_completed(futures), total=len(futures), desc="Waiting for executor"
    ):
        # Get result here to be notified of any exceptions.
        f.result()

    executor.shutdown()
