# -*- coding: utf-8 -*-
import math
from string import ascii_lowercase

import matplotlib.pyplot as plt
import numpy as np

from .data import get_gfed_regions, get_pnv_mega_plot_data
from .metrics import calculate_resampled_errors, mpd, nme


def error_hist(
    *, errors, title, error_dict, save_path, title_label=None, fig=None, ax=None
):
    if fig is None and ax is None:
        fig = plt.figure()
    elif fig is None:
        fig = ax.get_figure()
    if ax is None:
        ax = plt.axes()

    ax.set_title(title)

    if title_label is not None:
        ax.text(-0.01, 1.05, title_label, transform=ax.transAxes)

    if errors is not None:
        ax.hist(errors, bins="auto", density=True)

    xmins = [np.min(errors)] if errors is not None else []
    xmaxs = [np.max(errors)] if errors is not None else []

    ax2 = ax.twinx()

    # Indicate other errors.
    prev_ylim = ax.get_ylim()
    for (i, (key, (err, std))) in enumerate(error_dict.items()):
        ax.vlines(err, *prev_ylim, color=f"C{i+1}", label=key)

        xs = np.linspace(err - std, err + std, 100)
        ax2.plot(
            xs,
            (1 / np.sqrt(2 * np.pi * std**2))
            * np.exp(-((xs - err) ** 2) / (2 * std**2)),
            c=f"C{i+1}",
            linestyle="--",
            alpha=0.8,
        )

        xmins.append(err - std)
        xmaxs.append(err + std)

    ax.set_ylim(*prev_ylim)
    ax.set_xlim(np.min(xmins), np.max(xmaxs))

    ax.legend(loc="best")
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)

    return min(xmins), max(xmaxs)


def null_model_analysis(
    reference_data,
    comp_data=None,
    rng=None,
    save_dir=None,
    N=10000,
    regions="PNV",
):
    """Data should have the 3D shape (12, M, N), i.e. map data over 12 months.

    For N96e, M=144, N=192

    Args:
        reference_data (12, lat, lon) numpy array: Reference data.
        comp_data (dict of (12, lat, lon) numpy array): Data to compare to
            `reference_data`.
        rng (numpy.random.default_rng):
        save_dir (save directory):
        N (int):  Number of resampling operations.
        regions ({None, "GFED", "PNV"}): If not None, perform a regional error
            breakdown.

    """
    total_mask = np.zeros_like(reference_data, dtype=np.bool_)

    if comp_data is None:
        comp_data = {}
    else:
        for data in comp_data.values():
            assert len(data.shape) == 3
            total_mask |= np.ma.getmaskarray(data)

    assert reference_data.ndim == 3
    total_mask |= np.ma.getmaskarray(reference_data)
    total_sel = ~total_mask

    if rng is None:
        rng = np.random.default_rng()

    valid_reference_data = np.ma.getdata(reference_data)[total_sel]
    valid_comp_data = {
        key: np.ma.getdata(data)[total_sel] for key, data in comp_data.items()
    }

    nme_errors, mpd_errors = calculate_resampled_errors(
        reference_data=reference_data,
        valid_reference_data=valid_reference_data,
        total_sel=total_sel,
        N=N,
    )

    nme_error_dict = {}
    mpd_error_dict = {}

    # Error given just the mean state.
    nme_error_dict["mean_state"] = nme(
        obs=valid_reference_data,
        pred=np.zeros_like(valid_reference_data) + np.mean(valid_reference_data),
        return_std=True,
    )
    mpd_error_dict["mean_state"] = mpd(
        obs=reference_data,
        pred=np.zeros_like(reference_data) + np.mean(valid_reference_data),
        return_std=True,
    )

    # Errors for the other data.
    for key, data in valid_comp_data.items():
        nme_error_dict[key] = nme(obs=valid_reference_data, pred=data, return_std=True)
    for key, data in comp_data.items():
        mpd_error_dict[key] = mpd(obs=reference_data, pred=data, return_std=True)

    # NME Errors.
    error_hist(
        errors=nme_errors,
        title="NME errors (with std of mean)",
        error_dict=nme_error_dict,
        save_path=save_dir / "nme_errors.png",
    )

    # MPD Errors.
    error_hist(
        errors=mpd_errors,
        title="MPD errors (with std of mean)",
        error_dict=mpd_error_dict,
        save_path=save_dir / "mpd_errors.png",
    )

    if regions is not None:
        # Regional plotting.
        if regions == "GFED":
            regions_cube, N_plots = get_gfed_regions()
        elif regions == "PNV":
            regions_cube, N_plots = get_pnv_mega_plot_data()
        else:
            raise ValueError(f"Unknown regions: '{regions}'.")

        region_nrows_ncols = dict(nrows=math.ceil(N_plots / 2), ncols=2)
        regions_cube.data.mask |= np.any(total_mask, axis=0)

        sharex = False

        nme_fig, nme_axes = plt.subplots(
            sharex=sharex, sharey=True, figsize=(9, 9), **region_nrows_ncols
        )
        mpd_fig, mpd_axes = plt.subplots(
            sharex=sharex, sharey=True, figsize=(9, 9), **region_nrows_ncols
        )

        if sharex:
            nme_xmin = np.inf
            nme_xmax = -np.inf
            mpd_xmin = np.inf
            mpd_xmax = -np.inf

        for (plot_i, (region_code, region_name)) in enumerate(
            {
                code: name
                for code, name in regions_cube.attributes["short_regions"].items()
                # Ignore the Ocean region.
                if code != 0
            }.items()
        ):
            region_sel = (
                np.ones((12, 1, 1), dtype=np.bool_)
                & (np.ma.getdata(regions_cube.data) == region_code)[np.newaxis]
            )
            # NOTE hist() does not seem to handle masked arrays, so ensure that only valid
            # entries are passed to it.

            # Regional errors.
            reg_nme_error_dict = {}
            reg_mpd_error_dict = {}

            reg_reference_data = np.ma.MaskedArray(
                np.ma.getdata(reference_data),
                mask=~(~np.ma.getmaskarray(reference_data) & region_sel),
            )
            reg_valid_reference_data = np.ma.getdata(reference_data)[region_sel]

            reg_valid_comp_data = {
                key: np.ma.getdata(data)[region_sel] for key, data in comp_data.items()
            }
            reg_comp_data = {
                key: np.ma.MaskedArray(
                    np.ma.getdata(data), mask=~(~np.ma.getmaskarray(data) & region_sel)
                )
                for key, data in comp_data.items()
            }

            for key, data in reg_valid_comp_data.items():
                reg_nme_error_dict[key] = nme(
                    obs=reg_valid_reference_data, pred=data, return_std=True
                )
            for key, data in reg_comp_data.items():
                reg_mpd_error_dict[key] = mpd(
                    obs=reg_reference_data, pred=data, return_std=True
                )

            title_label = f"({ascii_lowercase[plot_i]})"

            nme_reg_xmin, nme_reg_xmax = error_hist(
                errors=None,
                # Show the region name and number of selected locations.
                title=f"{region_name} (n={np.sum(region_sel) / 12})",
                title_label=title_label,
                error_dict=reg_nme_error_dict,
                save_path=None,
                fig=nme_fig,
                ax=nme_axes.ravel()[plot_i],
            )

            mpd_reg_xmin, mpd_reg_xmax = error_hist(
                errors=None,
                # Show the region name and number of selected locations.
                title=f"{region_name} (n={np.sum(region_sel) / 12})",
                title_label=title_label,
                error_dict=reg_mpd_error_dict,
                save_path=None,
                fig=mpd_fig,
                ax=mpd_axes.ravel()[plot_i],
            )

            if sharex:
                if nme_reg_xmin < nme_xmin:
                    nme_xmin = nme_reg_xmin
                if nme_reg_xmax > nme_xmax:
                    nme_xmax = nme_reg_xmax
                if mpd_reg_xmin < mpd_xmin:
                    mpd_xmin = mpd_reg_xmin
                if mpd_reg_xmax > mpd_xmax:
                    mpd_xmax = mpd_reg_xmax

        if sharex:
            for ax in nme_axes.ravel():
                ax.set_xlim(nme_xmin, nme_xmax)
            for ax in mpd_axes.ravel():
                ax.set_xlim(mpd_xmin, mpd_xmax)

        nme_fig.suptitle("Regional NME Errors")
        mpd_fig.suptitle("Regional MPD Errors")

        for fig in [nme_fig, mpd_fig]:
            fig.tight_layout(rect=[0, 0.0, 1, 0.98])

        nme_fig.savefig(save_dir / "regional_nme_errors.png")
        mpd_fig.savefig(save_dir / "regional_mpd_errors.png")

        for fig in [nme_fig, mpd_fig]:
            plt.close(fig)
