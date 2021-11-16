# -*- coding: utf-8 -*-
import math

import iris
import matplotlib.pyplot as plt
import numpy as np
from jules_output_analysis.data import n96e_lats, n96e_lons
from loguru import logger
from wildfires.analysis import cube_plotting
from wildfires.data import regions_GFED, regrid

from python_inferno.cache import cache
from python_inferno.metrics import calculate_phase, calculate_phase_2d
from python_inferno.utils import memoize, wrap_phase_diffs


def lin_cube_plotting(*, data, title):
    cube_plotting(
        data,
        title=title,
        nbins=9,
        vmin_vmax_percentiles=(5, 95),
        fig=plt.figure(figsize=(12, 5)),
        colorbar_kwargs=dict(format="%.1e"),
    )


def log_cube_plotting(*, data, title, raw_data):
    cube_plotting(
        data,
        title=title,
        boundaries=np.geomspace(*np.quantile(raw_data[raw_data > 0], [0.05, 0.95]), 8),
        fig=plt.figure(figsize=(12, 5)),
        colorbar_kwargs=dict(format="%.1e"),
    )


def phase_calc(*, data):
    assert len(data.shape) == 3, "Need time, x, y"
    phase = np.ma.MaskedArray(
        calculate_phase_2d(np.ascontiguousarray(np.ma.getdata(data))),
        mask=np.all(np.ma.getmaskarray(data), axis=0),
    )
    phase += np.pi
    # Phase is now in [0, 2pi].
    phase -= calculate_phase(
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(12, 1)
    )[0]
    phase %= 2 * np.pi
    # Phase is now in [0, 2pi].
    #
    # Transform to [0.5, 12.5].
    phase *= 12 / (2 * np.pi)
    phase += 0.5
    return phase


def plot_phase_map(*, phase, title):
    cube_plotting(
        phase,
        title=title,
        boundaries=np.linspace(0.5, 12.5, 13),
        cmap="twilight",
        fig=plt.figure(figsize=(12, 5)),
        colorbar_kwargs=dict(format="%.1e"),
    )


def plot_phase_diff_map(*, phase_diff, title):
    cube_plotting(
        phase_diff,
        title=title,
        boundaries=np.linspace(-6, 6, 10),
        cmap="RdBu",
        fig=plt.figure(figsize=(12, 5)),
        colorbar_kwargs=dict(format="%.1e"),
    )


@memoize
@cache
def get_gfed_regions():
    gfed_regions = regrid(
        regions_GFED(),
        new_latitudes=n96e_lats,
        new_longitudes=n96e_lons,
        scheme=iris.analysis.Nearest(),
    )
    assert gfed_regions.attributes["regions"][0] == "Ocean"
    gfed_regions.data.mask |= np.ma.getdata(gfed_regions.data) == 0
    return gfed_regions


def plotting(
    *,
    exp_name,
    exp_key=None,
    raw_data,
    model_ba_2d_data,
    hist_bins,
    arcsinh_adj_factor,
    arcsinh_factor,
    scores=None,
    save_dir=None,
    ref_2d_data=None,
):
    if scores is not None:
        arcsinh_nme = scores["arcsinh_nme"]
        mpd = scores["mpd"]
        total = arcsinh_nme + mpd
        title = (
            f"{exp_name}, arcsinh NME: {arcsinh_nme:0.2f}, MPD: {mpd:0.2f}, "
            f"Total: {total:0.2f}"
        )
    else:
        title = exp_name

    if exp_key is None:
        exp_key = exp_name.replace(" ", "_").lower()

    # Regional plotting setup.
    gfed_regions = get_gfed_regions()
    N_plots = len(gfed_regions.attributes["regions"]) - 1  # Ignore the Ocean region
    region_nrows_ncols = dict(nrows=math.ceil(N_plots / 2), ncols=2)

    # Plotting.

    # Global BA histogram.
    logger.debug("Plotting hist")
    plt.figure()
    plt.hist(raw_data, bins=hist_bins)
    plt.yscale("log")
    plt.title(title)
    if save_dir is not None:
        ba_hist_dir = save_dir / "BA_hist"
        ba_hist_dir.mkdir(exist_ok=True, parents=False)
        plt.savefig(ba_hist_dir / f"hist_{exp_key}.png")
    plt.close()

    # Regional BA histograms.
    bin_width = 0.02

    fig, axes = plt.subplots(**region_nrows_ncols)
    for (ax, (region_code, region_name)) in zip(
        axes.ravel(),
        {
            code: name
            for code, name in gfed_regions.attributes["regions"].items()
            # Ignore the Ocean region.
            if code != 0
        }.items(),
    ):
        region_sel = (
            np.ones((12, 1, 1), dtype=np.bool_)
            & (np.ma.getdata(gfed_regions.data) == region_code)[np.newaxis]
        )
        region_data = model_ba_2d_data[region_sel]
        reg_min = np.min(region_data)
        reg_max = np.max(region_data)

        bins = np.arange(reg_min, reg_max, bin_width)
        bins = np.unique(np.append(bins, bins[-1] + bin_width))

        ax.hist(region_data, bins=bins)
        ax.set_yscale("log")
        # Show the region name and number of selected locations.
        ax.set_title(f"{region_name} (n={np.sum(region_sel) / 12})")

    fig.suptitle(title)
    plt.tight_layout()

    if save_dir is not None:
        ba_reg_hist_dir = save_dir / "BA_reg_hist"
        ba_reg_hist_dir.mkdir(exist_ok=True, parents=False)
        plt.savefig(ba_reg_hist_dir / f"reg_hists_{exp_key}.png")
    plt.close()

    # Global BA map with log bins.
    log_cube_plotting(data=model_ba_2d_data, title=title, raw_data=raw_data)
    if save_dir is not None:
        ba_map_log_dir = save_dir / "BA_map_log"
        ba_map_log_dir.mkdir(exist_ok=True, parents=False)
        plt.savefig(ba_map_log_dir / f"BA_map_log_{exp_key}.png")
    plt.close()

    # Global BA map with linear bins.
    lin_cube_plotting(data=model_ba_2d_data, title=title)
    if save_dir is not None:
        ba_map_lin_dir = save_dir / "BA_map_lin"
        ba_map_lin_dir.mkdir(exist_ok=True, parents=False)
        plt.savefig(ba_map_lin_dir / f"BA_map_lin_{exp_key}.png")
    plt.close()

    # Global BA map with linear bins and arcsinh transform.
    lin_cube_plotting(
        data=arcsinh_adj_factor * np.arcsinh(arcsinh_factor * model_ba_2d_data),
        title=title,
    )
    if save_dir is not None:
        ba_map_arcsinh_dir = save_dir / "BA_map_arcsinh"
        ba_map_arcsinh_dir.mkdir(exist_ok=True, parents=False)
        plt.savefig(ba_map_arcsinh_dir / f"BA_map_arcsinh_{exp_key}.png")
    plt.close()

    # Global phase map plot.
    plot_phase_map(
        phase=phase_calc(data=model_ba_2d_data),
        title=title,
    )
    if save_dir is not None:
        phase_map_dir = save_dir / "phase_map"
        phase_map_dir.mkdir(exist_ok=True, parents=False)
        plt.savefig(phase_map_dir / f"phase_map_{exp_key}.png")
    plt.close()

    # Phase difference (relative to GFED4) plots.
    if ref_2d_data is not None:
        phase_diff = wrap_phase_diffs(
            phase_calc(data=ref_2d_data) - phase_calc(data=model_ba_2d_data)
        )

        # Global phase difference histogram.
        plt.figure()
        plt.title(title)
        plt.hist(
            np.ma.getdata(phase_diff)[~np.ma.getmaskarray(phase_diff)],
            bins=np.linspace(-12, 12, 50),
            density=True,
        )
        if save_dir is not None:
            phase_diff_hist_dir = save_dir / "phase_diff_hist"
            phase_diff_hist_dir.mkdir(exist_ok=True, parents=False)
            plt.savefig(phase_diff_hist_dir / f"phase_diff_hist_{exp_key}.png")
        plt.close()

        # Regional phase difference histograms.
        bins = np.linspace(-6, 6, 10)

        fig, axes = plt.subplots(**region_nrows_ncols)
        for (ax, (region_code, region_name)) in zip(
            axes.ravel(),
            {
                code: name
                for code, name in gfed_regions.attributes["regions"].items()
                # Ignore the Ocean region.
                if code != 0
            }.items(),
        ):
            region_sel = np.ma.getdata(gfed_regions.data) == region_code
            ax.hist(phase_diff[region_sel])
            # Show the region name and number of selected locations.
            ax.set_title(f"{region_name} (n={np.sum(region_sel)})")

        fig.suptitle(title)
        plt.tight_layout()

        if save_dir is not None:
            reg_phase_diff_hist_dir = save_dir / "reg_phase_diff_hist"
            reg_phase_diff_hist_dir.mkdir(exist_ok=True, parents=False)
            plt.savefig(reg_phase_diff_hist_dir / f"reg_phase_diff_hists_{exp_key}.png")
        plt.close()

        # Global phase difference map.
        plot_phase_diff_map(phase_diff=phase_diff, title=title)
        if save_dir is not None:
            phase_diff_map_dir = save_dir / "phase_diff_map"
            phase_diff_map_dir.mkdir(exist_ok=True, parents=False)
            plt.savefig(phase_diff_map_dir / f"phase_diff_map_{exp_key}.png")
        plt.close()
    else:
        logger.debug("'ref_2d_data' not given.")
