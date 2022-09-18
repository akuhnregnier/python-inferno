# -*- coding: utf-8 -*-
import math
from contextlib import contextmanager
from itertools import product
from numbers import Integral
from pathlib import Path
from string import ascii_lowercase

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from jules_output_analysis.data import n96e_lats, n96e_lons
from jules_output_analysis.utils import PFTs, pft_acronyms
from loguru import logger
from matplotlib.colors import from_levels_and_colors
from matplotlib.transforms import offset_copy
from wildfires.analysis import cube_plotting

from .ba_model import ARCSINH_FACTOR, BAModel
from .configuration import pft_group_names, scheme_name_map
from .data import get_2d_cubes, get_gfed_regions, get_pnv_mega_plot_data
from .metrics import calculate_phase, calculate_phase_2d
from .utils import wrap_phase_diffs


def lin_cube_plotting(*, data, title, label="BA"):
    cube_plotting(
        data * 1e6,
        title=title,
        nbins=9,
        vmin_vmax_percentiles=(5, 95),
        fig=plt.figure(figsize=(12, 5)),
        colorbar_kwargs=dict(format="%.1e", label=label),
    )


def log_cube_plotting(*, data, title, raw_data, label="BA"):
    cube_plotting(
        data * 1e6,
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


def plot_phase_diff_map(*, phase_diff, title, label, figsize=(12, 5)):
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=ccrs.Robinson()))
    cube_plotting(
        phase_diff,
        title=title,
        boundaries=np.linspace(-6, 6, 10),
        cmap="twilight_shifted",
        fig=fig,
        ax=ax,
        colorbar_kwargs=dict(format="%.1e", label=label),
    )
    return fig, ax


def plot_phase_diff_locs(
    *,
    model_ba_2d_data,
    ref_2d_data,
    phase_diff,
    title,
    label,
    save_dir,
    exp_key,
    data_2d_cubes,
):
    if save_dir is not None:
        save_dir.mkdir(exist_ok=True, parents=False)
        save_dir = save_dir / exp_key
        save_dir.mkdir(exist_ok=True, parents=False)

    fig, ax = plot_phase_diff_map(
        phase_diff=phase_diff, title=title, label=label, figsize=(24, 10)
    )

    valid_phase_diffs = np.ma.getdata(phase_diff)[~np.ma.getmaskarray(phase_diff)]
    unique_phase_diffs = list(np.unique(valid_phase_diffs))

    plotted_coord_indices = []

    for i in range(40):
        target_phase_diff = unique_phase_diffs[-i]
        lat_indices, lon_indices = np.where(phase_diff == target_phase_diff)

        for lat_i, lon_i in zip(lat_indices, lon_indices):
            lat = n96e_lats[lat_i]
            lon = n96e_lons[lon_i]

            if plotted_coord_indices and (
                np.min(
                    np.sum(
                        np.abs(
                            np.array(plotted_coord_indices) - np.array([lat_i, lon_i])
                        ),
                        axis=1,
                    )
                )
                <= 3
            ):
                continue

            plotted_coord_indices.append((lat_i, lon_i))

            ax.plot(
                lon,
                lat,
                marker="o",
                markerfacecolor="None",
                markeredgecolor="red",
                markersize=8,
                transform=ccrs.PlateCarree(),
                alpha=0.7,
            )
            text_transform = offset_copy(
                ccrs.PlateCarree()._as_mpl_transform(ax), units="dots", x=40
            )
            ax.text(
                lon,
                lat,
                i,
                horizontalalignment="left",
                transform=text_transform,
                bbox=dict(
                    boxstyle="square,pad=0", facecolor="white", alpha=0.7, ec="grey"
                ),
            )

            N_plots = len(data_2d_cubes) + 1  # +1 for the BA plots.
            ncols = 5
            nrows = math.ceil(N_plots / ncols)

            loc_fig, loc_axes = plt.subplots(
                nrows, ncols, figsize=(3 * ncols, 3 * nrows)
            )
            loc_axes.ravel()[0].plot(model_ba_2d_data[:, lat_i, lon_i], label="model")
            loc_axes.ravel()[0].plot(ref_2d_data[:, lat_i, lon_i], label="ref")
            loc_axes.ravel()[0].set_title("BA")
            loc_axes.ravel()[0].legend()

            for (j, (name, cube)) in enumerate(data_2d_cubes.items()):
                loc_ax = loc_axes.ravel()[j + 1]
                loc_ax.set_title(name)

                if len(cube.shape) == 3:
                    loc_ax.plot(cube[:, lat_i, lon_i].data)
                elif len(cube.shape) == 4:
                    if cube.shape[1] == 3:
                        labels = pft_group_names
                        colors = plt.get_cmap("tab10").colors
                    else:
                        labels = pft_acronyms[PFTs.VEG13]
                        colors = plt.get_cmap("tab20").colors
                    for k in range(min(13, cube.shape[1])):
                        loc_ax.plot(
                            cube[:, k, lat_i, lon_i].data, label=labels[k], c=colors[k]
                        )
                    loc_ax.legend()
                else:
                    print(f"Unsupported shape: {cube.shape}")

            for loc_ax in loc_axes.ravel()[N_plots:]:
                loc_ax.set_axis_off()

            # TODO Maybe combine plots with plots of the model response to individual
            # variables, e.g. as aggregated as part of the GAM analysis scripts.

            if save_dir is not None:
                loc_fig.savefig(save_dir / f"{i}_{exp_key}.png")
            plt.close(loc_fig)

    if save_dir is not None:
        fig.savefig(save_dir / f"phase_diff_locs_{exp_key}.png")
    plt.close(fig)


def plotting(
    *,
    exp_name,
    exp_key=None,
    raw_data,
    model_ba_2d_data,
    hist_bins,
    scores=None,
    save_dir=None,
    ref_2d_data=None,
    regions="GFED",
    data_params=None,
):
    use_style()
    # Prep.

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

    if data_params is not None:
        data_dict = BAModel(**data_params).data_dict
        data_2d_cubes = get_2d_cubes(data_dict=data_dict)
    else:
        data_2d_cubes = None

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
    for (ax, (region_code, region_name)) in zip(
        axes.ravel(),
        {
            code: name
            for code, name in regions_cube.attributes["short_regions"].items()
            # Ignore the Ocean region.
            if code != 0
        }.items(),
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

    for ax in axes.ravel()[N_plots + 1 :]:
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
        data=np.arcsinh(ARCSINH_FACTOR * model_ba_2d_data),
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

        if data_2d_cubes is not None:
            # Global phase difference map with focus on individual locations.
            plot_phase_diff_locs(
                model_ba_2d_data=model_ba_2d_data,
                ref_2d_data=ref_2d_data,
                phase_diff=phase_diff,
                title=title,
                label=xlabel,
                save_dir=save_dir / "phase_diff_locs",
                exp_key=exp_key,
                data_2d_cubes=data_2d_cubes,
            )
    else:
        logger.debug("'ref_2d_data' not given.")


def plot_collated_phase_diffs(*, phase_diff_dict, save_dir, save_name):
    # Global phase difference maps.
    use_style()

    bin_edges = np.linspace(-6, 6, 10)

    cmap, norm = from_levels_and_colors(
        levels=list(bin_edges),
        colors=plt.get_cmap("twilight_shifted")(np.linspace(0, 1, len(bin_edges) - 1)),
        extend="neither",
    )

    ncols = 2
    nrows = math.ceil(len(phase_diff_dict) / ncols)

    with custom_axes(
        ncols=ncols,
        nrows=nrows,
        nplots=len(phase_diff_dict),
        height=1.9 * nrows,
        width=6.2,
        h_pad=0.04,
        w_pad=0.02,
        cbar_pos="bottom",
        cbar_height=0.012,
        cbar_width=0.6,
        cbar_h_pad=0.03,
        cbar_w_pad=0,
        projection=ccrs.Robinson(),
    ) as (fig, axes, cax):
        for (i, (ax, (title, phase_diff))) in enumerate(
            zip(axes, phase_diff_dict.items())
        ):
            cube_plotting(
                phase_diff,
                title="",
                fig=fig,
                ax=ax,
                cmap=cmap,
                norm=norm,
                colorbar_kwargs=dict(
                    cax=cax,
                    orientation="horizontal",
                    label="phase diff (obs - model)",
                    format="%.1f",
                )
                if i == 0
                else False,
            )

            if title in scheme_name_map:
                ax.set_title(f"SINFERNO-{scheme_name_map[title]}")
            else:
                ax.set_title(title)

        fig.savefig(save_dir / save_name)


def get_plot_name_map(*, dryness_method, fuel_build_up_method):
    name_map = {
        "t1p5m_tile": "Temperature",
        "fapar_diag_pft": "NPP",
    }

    if dryness_method == 1:
        name_map["dry_days"] = "Dry Days"
    elif dryness_method == 2:
        name_map["grouped_dry_bal"] = "Dry Bal"

    if fuel_build_up_method == 1:
        name_map["fuel_build_up"] = "Antecedent NPP"
    elif fuel_build_up_method == 2:
        name_map["litter_pool"] = "Litter Pool"

    return name_map


def get_plot_name_map_total():
    name_map = {
        "t1p5m_tile": "Temperature",
        "temperature": "Temperature",
        "fapar_diag_pft": "NPP",
        "fapar": "NPP",
        "dry_days": "Dry Days",
        "dry_day": "Dry Days",
        "dry_bal": "Dry Bal",
        "grouped_dry_bal": "Dry Bal",
        "fuel_build_up": "Antecedent NPP",
        "litter_pool": "Litter Pool",
        "obs_pftcrop_1d": "Cropland",
        "overall_scale": "Overall Scale",
        "fapar_factor": "NPP Factor",
        "fapar_centre": "NPP Centre",
        "fapar_shape": "NPP Shape",
        "fapar_weight": "NPP Weight",
        "dryness_weight": "Dryness Weight",
        "fuel_weight": "Fuel Weight",
        "average_samples": "Average Samples",
        "crop_f": "crop_f",
        "dry_day_factor": "Dry Day Factor",
        "dry_day_centre": "Dry Day Centre",
        "dry_day_shape": "Dry Day Shape",
        "rain_f": "rain_f",
        "vpd_f": "vpd_f",
        "dry_bal_factor": "Dry Bal Factor",
        "dry_bal_centre": "Dry Bal Centre",
        "dry_bal_shape": "Dry Bal Shape",
        "fuel_build_up_n_samples": "Fuel Build Up n_samples",
        "fuel_build_up_factor": "Fuel Build Up Factor",
        "fuel_build_up_centre": "Fuel Build Up Centre",
        "fuel_build_up_shape": "Fuel Build Up Shape",
        "litter_tc": "litter_tc",
        "leaf_f": "leaf_f",
        "litter_pool_factor": "Litter Pool Factor",
        "litter_pool_centre": "Litter Pool Centre",
        "litter_pool_shape": "Litter Pool Shape",
        "temperature_factor": "Temperature Factor",
        "temperature_centre": "Temperature Centre",
        "temperature_shape": "Temperature Shape",
        "temperature_weight": "Temperature Weight",
    }

    return name_map


def get_plot_units_map(*, dryness_method, fuel_build_up_method):
    name_map = {
        "t1p5m_tile": "K",
        "fapar_diag_pft": "1",
    }

    if dryness_method == 1:
        name_map["dry_days"] = "days"
    elif dryness_method == 2:
        name_map["grouped_dry_bal"] = "1"

    if fuel_build_up_method == 1:
        name_map["fuel_build_up"] = "1"
    elif fuel_build_up_method == 2:
        name_map["litter_pool"] = "1"

    return name_map


def plot_label_case(label):
    return label.lower().replace("npp", "NPP")


def use_style():
    """Use common plotting style."""
    mpl.style.use(Path(__file__).absolute().parent / "../../analysis/matplotlibrc")


@contextmanager
def custom_axes(
    *,
    ncols,
    nrows,
    nplots,
    height,
    width,
    h_pad,
    w_pad,
    cbar_pos,
    cbar_height,
    cbar_width,
    cbar_h_pad,
    cbar_w_pad,
    projection,
):
    assert cbar_pos == "bottom"

    # Including padding.
    cbar_total_height = cbar_height + cbar_h_pad
    cbar_width + cbar_w_pad

    total_ax_height = 1 - cbar_height - cbar_h_pad - (nrows - 1) * h_pad
    ax_height = total_ax_height / nrows

    total_ax_width = 1 - (ncols - 1) * w_pad
    ax_width = total_ax_width / ncols

    plt.ioff()
    fig = plt.figure(figsize=(width, height))

    cax = fig.add_axes(
        [
            # xmin
            0.5 - cbar_width / 2.0,
            # ymin
            0,
            # width (x)
            cbar_width,
            # height (y)
            cbar_height,
        ]
    )

    axes = []

    for (label, (i, j)) in zip(
        ascii_lowercase, product(range(nrows - 1, -1, -1), range(ncols))
    ):
        # Row index (i), col index (j).
        axes.append(
            fig.add_axes(
                [
                    # xmin
                    j * (ax_width + w_pad),
                    # ymin
                    cbar_total_height + i * (ax_height + h_pad),
                    # width (x)
                    ax_width,
                    # height (y)
                    ax_height,
                ],
                projection=projection,
            )
        )

        axes[-1].text(0.01, 1.05, f"({label})", transform=axes[-1].transAxes)

        if len(axes) == nplots:
            # Stop adding axes.
            break

    yield fig, axes, cax

    plt.close(fig)


def collated_ba_log_plot(*, ba_data_dict, plot_dir, save_name):
    use_style()

    bin_edges = np.geomspace(5e-6, 5e-3, 8)

    cmap, norm = from_levels_and_colors(
        levels=list(bin_edges),
        colors=plt.get_cmap("inferno")(np.linspace(0, 1, len(bin_edges) + 1)),
        extend="both",
    )

    ncols = 2
    nrows = math.ceil(len(ba_data_dict) / ncols)

    found_standard_inferno = False

    with custom_axes(
        ncols=ncols,
        nrows=nrows,
        nplots=len(ba_data_dict),
        height=1.9 * nrows,
        width=6.2,
        h_pad=0.04,
        w_pad=0.02,
        cbar_pos="bottom",
        cbar_height=0.012,
        cbar_width=0.6,
        cbar_h_pad=0.03,
        cbar_w_pad=0,
        projection=ccrs.Robinson(),
    ) as (fig, axes, cax):
        for (i, (ax, (title, data))) in enumerate(zip(axes, ba_data_dict.items())):

            if title == "standard INFERNO":
                # Multiply by number of seconds in month.
                data = 31 * 24 * 60 * 60 * data.copy()
                found_standard_inferno = True

            try:
                cube_plotting(
                    data,
                    title="",
                    fig=fig,
                    ax=ax,
                    cmap=cmap,
                    norm=norm,
                    colorbar_kwargs=dict(
                        cax=cax,
                        orientation="horizontal",
                        label="BA",
                        format="%.1e",
                    )
                    if i == 0
                    else False,
                )
            except AssertionError:
                pass

            if title in scheme_name_map:
                ax.set_title(f"SINFERNO-{scheme_name_map[title]}")
            else:
                ax.set_title(title)

        assert found_standard_inferno

        fig.savefig(plot_dir / save_name)
