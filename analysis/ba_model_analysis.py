#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
import sys
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from itertools import islice
from operator import itemgetter
from pathlib import Path
from warnings import filterwarnings

import numpy as np
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from loguru import logger
from tqdm import tqdm

from python_inferno.ba_model import ARCSINH_FACTOR, BAModel, calculate_scores
from python_inferno.cache import cache
from python_inferno.configuration import land_pts
from python_inferno.data import load_data, load_jules_lats_lons
from python_inferno.metrics import Metrics
from python_inferno.metrics_plotting import null_model_analysis
from python_inferno.model_params import get_model_params, plot_param_histograms
from python_inferno.plotting import (
    collated_ba_log_plot,
    phase_calc,
    plot_collated_abs_arcsinh_diffs,
    plot_collated_phase_diffs,
    plotting,
    use_style,
    wrap_phase_diffs,
)
from python_inferno.utils import (
    ConsMonthlyAvg,
    DebugExecutor,
    PartialDateTime,
    get_apply_mask,
    memoize,
    temporal_processing,
)

filterwarnings("ignore", ".*divide by zero.*")
filterwarnings("ignore", ".*invalid units.*")
filterwarnings("ignore", ".*may not be fully.*")
filterwarnings("ignore", ".*axes.*")


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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", type=int, help="number of processes", default=10)
    parser.add_argument(
        "--no-hist-plots", action="store_true", help="do not plot parameter histograms"
    )
    parser.add_argument(
        "--no-phase-diff-locs", action="store_true", help="do not plot phase diff locs"
    )
    parser.add_argument(
        "-n", type=int, help="method index (-1 selects all; default)", default=-1
    )
    args = parser.parse_args()

    assert args.p == 1, "Loc plotting always uses multiprocessing too."

    exclude_plot_keys = []
    if args.no_phase_diff_locs:
        exclude_plot_keys.append("data_params")

    use_style()
    save_dir = Path("~/tmp/ba-model-analysis/").expanduser()
    save_dir.mkdir(exist_ok=True, parents=False)
    plotting = partial(plotting, save_dir=save_dir, regions="PNV")

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    requested = (Metrics.MPD, Metrics.ARCSINH_NME)

    jules_lats, jules_lons = load_jules_lats_lons()

    # To prevent memory accumulation during repeated calculations below.
    memoize.active = False

    df, method_iter = get_model_params(progress=True, verbose=True)

    hist_bins = 50

    plot_data = dict()
    plot_prog = tqdm(desc="Generating plot data", total=6)

    executor = ProcessPoolExecutor(max_workers=10) if args.p > 1 else DebugExecutor()
    futures = []

    if args.n >= 0:
        slice_args = (args.n, args.n + 1)
    else:
        slice_args = (0, None)

    for (
        dryness_method,
        fuel_build_up_method,
        df_sel,
        min_index,
        min_loss,
        params,
        exp_name,
        exp_key,
    ) in islice(method_iter(), *slice_args):
        logger.info(exp_name)
        logger.info(exp_key)

        if not args.no_hist_plots:
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
        scores, avg_ba = itemgetter("scores", "avg_ba")(
            ba_model.calc_scores(model_ba=model_ba, requested=requested)
        )

        # NOTE Low BA mask is not used during the optimisation!
        # ba_mask_1d = get_ba_mask(mon_avg_gfed_ba_1d)
        # apply_ba_mask_1d = get_apply_mask(ba_mask_1d)

        orig_mon_avg_gfed_ba_1d = mon_avg_gfed_ba_1d
        # mon_avg_gfed_ba_1d = apply_ba_mask_1d(mon_avg_gfed_ba_1d)
        # model_ba = apply_ba_mask_1d(model_ba)  # ??
        # avg_ba = apply_ba_mask_1d(avg_ba)  # ??

        gc.collect()

        logger.info("Getting 2D cube")
        model_ba_2d = cube_1d_to_2d(
            get_1d_data_cube(avg_ba, lats=jules_lats, lons=jules_lons)
        )

        gc.collect()

        plot_data[exp_name] = dict(
            exp_key=exp_key,
            raw_data=np.ma.getdata(avg_ba)[~np.ma.getmaskarray(avg_ba)],
            model_ba_2d_data=model_ba_2d.data,
            hist_bins=hist_bins,
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
    )
    plot_prog.update()

    # Standard INFERNO BA.
    data_dict, jules_time_coord = get_processed_climatological_jules_ba()
    jules_ba_gb = data_dict.pop("jules_ba_gb")
    scores, avg_jules_ba = calculate_scores(
        model_ba=jules_ba_gb,
        cons_monthly_avg=ConsMonthlyAvg(jules_time_coord, L=land_pts),
        mon_avg_gfed_ba_1d=mon_avg_gfed_ba_1d,
    )

    plot_data["standard INFERNO"] = dict(
        raw_data=np.ma.getdata(avg_jules_ba)[~np.ma.getmaskarray(avg_jules_ba)],
        model_ba_2d_data=get_apply_mask(reference_obs.mask)(
            cube_1d_to_2d(
                get_1d_data_cube(avg_jules_ba, lats=jules_lats, lons=jules_lons)
            ).data
        ),
        hist_bins=hist_bins,
        scores=scores,
        # TODO Which params to use (combination of all?) - call again with actual
        # standard INFERNO params to get corresponding data - is this supported?
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
                **{
                    key: val
                    for key, val in data.items()
                    if key not in exclude_plot_keys
                },
            )
        )

    # Collated phase difference map.
    phase_diff_dict = {}
    for exp_name, data in plot_data.items():
        if exp_name == "GFED4":
            continue

        model_phase_2d = phase_calc(data=data["model_ba_2d_data"])
        phase_diff_dict[exp_name] = wrap_phase_diffs(
            phase_calc(data=reference_obs) - model_phase_2d
        )
    futures.append(
        executor.submit(
            plot_collated_phase_diffs,
            phase_diff_dict=phase_diff_dict,
            save_dir=save_dir,
            save_name="collated_phase_diffs",
        )
    )

    # Collated arcsinh-error map.
    abs_diff_arcsinh_dict = {}
    for exp_name, data in plot_data.items():
        if exp_name == "GFED4":
            continue
        abs_diff_arcsinh_dict[exp_name] = np.abs(
            np.arcsinh(ARCSINH_FACTOR * data["model_ba_2d_data"])
            - np.arcsinh(ARCSINH_FACTOR * reference_obs)
        )
    futures.append(
        executor.submit(
            plot_collated_abs_arcsinh_diffs,
            plot_data_dict=abs_diff_arcsinh_dict,
            save_dir=save_dir,
            save_name="collated_abs_arcsinh_diffs",
        )
    )

    # Collated log plotting of 2D BA cubes.
    futures.append(
        executor.submit(
            collated_ba_log_plot,
            ba_data_dict={
                exp_name: data_dict["model_ba_2d_data"]
                for exp_name, data_dict in plot_data.items()
            },
            plot_dir=save_dir,
            save_name="collated_log",
        )
    )

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
