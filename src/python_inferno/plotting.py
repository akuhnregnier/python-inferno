# -*- coding: utf-8 -*-
import math
from numbers import Integral

import iris
import matplotlib.pyplot as plt
import numpy as np
from jules_output_analysis.data import n96e_lats, n96e_lons
from loguru import logger
from wildfires.analysis import cube_plotting
from wildfires.data import regions_GFED, regrid

from python_inferno.cache import cache
from python_inferno.metrics import calculate_phase, calculate_phase_2d
from python_inferno.pnv import get_pnv_mega_regions
from python_inferno.utils import memoize, wrap_phase_diffs


def lin_cube_plotting(*, data, title, label="BA"):
    cube_plotting(
        data,
        title=title,
        nbins=9,
        vmin_vmax_percentiles=(5, 95),
        fig=plt.figure(figsize=(12, 5)),
        colorbar_kwargs=dict(format="%.1e", label=label),
    )


def log_cube_plotting(*, data, title, raw_data, label="log(BA)"):
    cube_plotting(
        data,
        title=title,
        boundaries=np.geomspace(*np.quantile(raw_data[raw_data > 0], [0.05, 0.95]), 8),
        fig=plt.figure(figsize=(12, 5)),
        colorbar_kwargs=dict(format="%.1e", label=label),
    )


def phase_calc(*, data):
    assert len(data.shape) == 3, "Need time, x, y"

    # Calculate phase in [-pi, pi].
    phase = np.ma.MaskedArray(
        calculate_phase_2d(np.ascontiguousarray(np.ma.getdata(data))),
        mask=np.any(np.ma.getmaskarray(data), axis=0),
    )

    # Shift s.t. phase=0 corresponds to point between Dec and Jan.
    phase -= calculate_phase(
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(12, 1)
    )[0]

    # Transform phase to [0, 2pi].
    phase %= 2 * np.pi

    # Make phase monotonically increasing.
    phase = 2 * np.pi - phase

    # Transform radians [0, 2pi] to [0, 12].
    phase *= 12 / (2 * np.pi)

    return phase


def plot_phase_map(*, phase, title, label="phase (month)"):
    cube_plotting(
        phase,
        title=title,
        boundaries=np.linspace(0.5, 12.5, 13),
        cmap="twilight",
        fig=plt.figure(figsize=(12, 5)),
        colorbar_kwargs=dict(format="%.1e", label=label),
    )


def plot_phase_diff_map(*, phase_diff, title, label):
    cube_plotting(
        phase_diff,
        title=title,
        boundaries=np.linspace(-6, 6, 10),
        cmap="RdBu",
        fig=plt.figure(figsize=(12, 5)),
        colorbar_kwargs=dict(format="%.1e", label=label),
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
    return (
        gfed_regions,
        len(gfed_regions.attributes["regions"]) - 1,
    )  # Ignore the Ocean region


def get_pnv_mega_plot_data():
    pnv_mega_regions = get_pnv_mega_regions()
    return pnv_mega_regions, len(pnv_mega_regions.attributes["regions"])


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
    regions="GFED",
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

    model_phase_2d = phase_calc(data=model_ba_2d_data)

    if isinstance(hist_bins, Integral):
        ba_bins = np.linspace(np.min(raw_data), np.max(raw_data), hist_bins)
    else:
        ba_bins = hist_bins

    # Regional plotting setup.
    if regions == "GFED":
        regions_cube, N_plots = get_gfed_regions()
    elif regions == "PNV":
        regions_cube, N_plots = get_pnv_mega_plot_data()
    else:
        raise ValueError(f"Unknown regions: '{regions}'.")
    regions_cube.data.mask |= np.any(model_ba_2d_data.mask, axis=0)

    region_nrows_ncols = dict(nrows=math.ceil(N_plots / 2), ncols=2)

    # Plotting.

    # Global BA histogram.
    logger.debug("Plotting hist")
    plt.figure()
    plt.hist(raw_data, bins=ba_bins)
    plt.yscale("log")
    plt.title(title)
    if save_dir is not None:
        ba_hist_dir = save_dir / "BA_hist"
        ba_hist_dir.mkdir(exist_ok=True, parents=False)
        plt.savefig(ba_hist_dir / f"hist_{exp_key}.png")
    plt.close()

    # Regional BA histograms.
    fig, axes = plt.subplots(
        sharex=True, sharey=True, figsize=(5, 8), **region_nrows_ncols
    )
    global_max_count = 0
    for (ax_i, (ax, (region_code, region_name))) in enumerate(
        zip(
            axes.ravel(),
            {
                code: name
                for code, name in regions_cube.attributes["short_regions"].items()
                # Ignore the Ocean region.
                if code != 0
            }.items(),
        )
    ):
        region_sel = (
            np.ones((12, 1, 1), dtype=np.bool_)
            & (np.ma.getdata(regions_cube.data) == region_code)[np.newaxis]
        )
        # NOTE hist() does not seem to handle masked arrays, so ensure that only valid
        # entries are passed to it.
        region_sel &= ~np.ma.getmaskarray(model_ba_2d_data)
        ax.hist(model_ba_2d_data[region_sel], bins=ba_bins)
        ax.set_yscale("log")
        # Show the region name and number of selected locations.
        ax.set_title(f"{region_name} (n={np.sum(region_sel) / 12})")

        max_count = np.histogram(model_ba_2d_data[region_sel], bins=ba_bins)[0].max()
        if max_count > global_max_count:
            global_max_count = max_count

    # Set on one axis - since the y-axes are shared, this affects all axes.
    ax.set_ylim(bottom=0.9, top=global_max_count * 1.5)

    for ax in axes.ravel()[ax_i + 1 :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.0, 1, 0.98])

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
        label="arcsinh(a*BA)",
    )
    if save_dir is not None:
        ba_map_arcsinh_dir = save_dir / "BA_map_arcsinh"
        ba_map_arcsinh_dir.mkdir(exist_ok=True, parents=False)
        plt.savefig(ba_map_arcsinh_dir / f"BA_map_arcsinh_{exp_key}.png")
    plt.close()

    # Global phase map plot.
    plot_phase_map(
        phase=model_phase_2d,
        title=title,
    )
    if save_dir is not None:
        phase_map_dir = save_dir / "phase_map"
        phase_map_dir.mkdir(exist_ok=True, parents=False)
        plt.savefig(phase_map_dir / f"phase_map_{exp_key}.png")
    plt.close()

    # Phase difference (relative to GFED4) plots.
    if ref_2d_data is not None:
        phase_diff = wrap_phase_diffs(phase_calc(data=ref_2d_data) - model_phase_2d)
        xlabel = "phase diff (obs - model)"
        bins = np.linspace(-6, 6, 20)
        assert not np.any(phase_diff) < bins[0]
        assert not np.any(phase_diff) > bins[-1]

        # Global phase difference histogram.
        plt.figure()
        plt.title(title)
        plt.hist(
            np.ma.getdata(phase_diff)[~np.ma.getmaskarray(phase_diff)],
            bins=bins,
            density=True,
        )
        plt.yscale("log")
        plt.xlabel(xlabel)
        if save_dir is not None:
            phase_diff_hist_dir = save_dir / "phase_diff_hist"
            phase_diff_hist_dir.mkdir(exist_ok=True, parents=False)
            plt.savefig(phase_diff_hist_dir / f"phase_diff_hist_{exp_key}.png")
        plt.close()

        # Regional phase difference histograms.

        fig, axes = plt.subplots(
            sharex=True, sharey=True, figsize=(4, 8), **region_nrows_ncols
        )
        for (ax, (region_code, region_name)) in zip(
            axes.ravel(),
            {
                code: name
                for code, name in regions_cube.attributes["short_regions"].items()
                # Ignore the Ocean region.
                if code != 0
            }.items(),
        ):
            region_sel = np.ma.getdata(regions_cube.data) == region_code
            ax.hist(phase_diff[region_sel], bins=bins, density=True)
            # Show the region name and number of selected locations.
            ax.set_title(f"{region_name} (n={np.sum(region_sel)})")

        for ax in axes[-1, :]:
            ax.set_xlabel(xlabel)

        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.0, 1, 0.98])

        if save_dir is not None:
            reg_phase_diff_hist_dir = save_dir / "reg_phase_diff_hist"
            reg_phase_diff_hist_dir.mkdir(exist_ok=True, parents=False)
            plt.savefig(reg_phase_diff_hist_dir / f"reg_phase_diff_hists_{exp_key}.png")
        plt.close()

        # Global phase difference map.
        plot_phase_diff_map(phase_diff=phase_diff, title=title, label=xlabel)
        if save_dir is not None:
            phase_diff_map_dir = save_dir / "phase_diff_map"
            phase_diff_map_dir.mkdir(exist_ok=True, parents=False)
            plt.savefig(phase_diff_map_dir / f"phase_diff_map_{exp_key}.png")
        plt.close()
    else:
        logger.debug("'ref_2d_data' not given.")
