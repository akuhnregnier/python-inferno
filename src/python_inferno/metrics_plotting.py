# -*- coding: utf-8 -*-
import math
from string import ascii_lowercase

import matplotlib.pyplot as plt
import numpy as np

from .configuration import scheme_name_map
from .data import get_gfed_regions, get_pnv_mega_plot_data
from .metrics import calculate_resampled_errors, mpd, nme
from .plotting import broken_y_axis, get_fig_ax, use_style


def error_hist(
    *,
    errors,
    title=None,
    ylabel=None,
    error_dict,
    save_path,
    title_label=None,
    fig=None,
    ax=None,
    ylim=None,
):
    fig, ax = get_fig_ax(fig=fig, ax=ax)

    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title_label is not None:
        ax.text(-0.01, 1.05, title_label, transform=ax.transAxes)

    # NOTE Old plot.
    # if errors is not None:
    #     ax.hist(errors, bins="auto", density=True)

    # xmins = [np.min(errors)] if errors is not None else []
    # xmaxs = [np.max(errors)] if errors is not None else []

    # ax2 = ax.twinx()

    # # Indicate other errors.
    # prev_ylim = ax.get_ylim()
    # for (i, (key, (err, std))) in enumerate(error_dict.items()):
    #     ax.vlines(err, *prev_ylim, color=f"C{i+1}", label=key)

    #     xs = np.linspace(err - std, err + std, 100)
    #     ax2.plot(
    #         xs,
    #         (1 / np.sqrt(2 * np.pi * std**2))
    #         * np.exp(-((xs - err) ** 2) / (2 * std**2)),
    #         c=f"C{i+1}",
    #         linestyle="--",
    #         alpha=0.8,
    #     )

    #     xmins.append(err - std)
    #     xmaxs.append(err + std)

    # ax.set_ylim(*prev_ylim)
    # ax.set_xlim(np.min(xmins), np.max(xmaxs))

    # ax.legend(loc="best")

    # NOTE New box plot.

    # XXX TODO Use median and quantiles, not mean / std?

    if errors is not None:
        s = {}
        # NOTE Median and Q1, Q3.
        # for key, result in zip(
        #     ["q1", "med", "q3"], np.quantile(errors, [0.25, 0.5, 0.75])
        # ):
        #     s[key] = result
        # NOTE Mean and mean +- std.
        s["med"] = np.mean(errors)
        err_std = np.std(errors)
        s["q1"] = s["med"] - err_std
        s["q3"] = s["med"] + err_std

        s["label"] = "resamp"
        stats = [s]
    else:
        stats = []
    for (key, (err, std)) in error_dict.items():
        stats.append(
            {
                "q1": err - std,
                "q3": err + std,
                "med": err,
                "label": scheme_name_map.get(key, key)
                if key != "standard INFERNO"
                else "standard\nINFERNO",
            }
        )

    for s in stats:
        s["whislo"] = None
        s["whishi"] = None

    ax.bxp(stats, showfliers=False)

    if ylim is not None:
        ax.set_ylim(ylim)

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)


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
    use_style()

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
    nme_error_dict["mean"] = nme(
        obs=valid_reference_data,
        pred=np.zeros_like(valid_reference_data) + np.mean(valid_reference_data),
        return_std=True,
    )
    mpd_error_dict["mean"] = mpd(
        obs=reference_data,
        pred=np.zeros_like(reference_data) + np.mean(valid_reference_data),
        return_std=True,
    )

    # Errors for the other data.
    for key, data in valid_comp_data.items():
        nme_error_dict[key] = nme(obs=valid_reference_data, pred=data, return_std=True)
    for key, data in comp_data.items():
        mpd_error_dict[key] = mpd(obs=reference_data, pred=data, return_std=True)

    single_boxplot_figsize = (4, 2.5)

    # NME Errors.
    shared_error_hist_kwargs = dict(
        errors=nme_errors, error_dict=nme_error_dict, save_path=None
    )
    broken_y_axis(
        figsize=single_boxplot_figsize,
        ylabel="NME error (std of mean)",
        ylabelpad=30,
        height_ratio=0.6,
        ylims=[
            (0.93, 1.116),
            (0.551, 0.595),
        ],
        save_path=save_dir / "nme_errors.pdf",
        plot_func=error_hist,
        plot_func_kwargs=(
            shared_error_hist_kwargs,
            shared_error_hist_kwargs,
        ),
    )

    # NOTE Single plot.
    # error_hist(
    #     errors=nme_errors,
    #     ylabel="NME error (std of mean)",
    #     error_dict=nme_error_dict,
    #     save_path=save_dir / "nme_errors.pdf",
    #     fig=plt.figure(figsize=single_boxplot_figsize),
    # )

    # MPD Errors.
    shared_error_hist_kwargs = dict(
        errors=mpd_errors, error_dict=mpd_error_dict, save_path=None
    )
    broken_y_axis(
        figsize=single_boxplot_figsize,
        ylabel="MPD error (std of mean)",
        ylabelpad=30,
        height_ratio=0.6,
        ylims=[
            (0.273, 0.53),
            (0.201, 0.245),
        ],
        save_path=save_dir / "mpd_errors.pdf",
        plot_func=error_hist,
        plot_func_kwargs=(
            shared_error_hist_kwargs,
            shared_error_hist_kwargs,
        ),
    )

    # NOTE Single plot.
    # error_hist(
    #     errors=mpd_errors,
    #     ylabel="MPD error (std of mean)",
    #     error_dict=mpd_error_dict,
    #     save_path=save_dir / "mpd_errors.pdf",
    #     fig=plt.figure(figsize=single_boxplot_figsize),
    # )

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

        region_nme_data = {}
        region_mpd_data = {}

        n_points_map = {}

        for region_code, region_name in {
            code: name
            for code, name in regions_cube.attributes["short_regions"].items()
            # Ignore the Ocean region.
            if code != 0
        }.items():
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

            region_nme_data[region_name] = reg_nme_error_dict
            region_mpd_data[region_name] = reg_mpd_error_dict

            n_points = np.sum(region_sel) / 12
            assert (round(n_points) - n_points) < 1e-10
            n_points = round(n_points)

            n_points_map[region_name] = n_points

        # Calculate common y-limits excluding the 'desert' region for its anomalous
        # behaviour.
        common_nme_y_limit = [math.inf, -math.inf]
        common_mpd_y_limit = [math.inf, -math.inf]

        for (region_name, reg_nme_error_dict), reg_mpd_error_dict in zip(
            region_nme_data.items(), region_mpd_data.values()
        ):
            if region_name == "desert":
                continue

            for err, std in reg_nme_error_dict.values():
                if (err - std) < common_nme_y_limit[0]:
                    common_nme_y_limit[0] = err - std
                if (err + std) > common_nme_y_limit[1]:
                    common_nme_y_limit[1] = err + std

            for err, std in reg_mpd_error_dict.values():
                if (err - std) < common_mpd_y_limit[0]:
                    common_mpd_y_limit[0] = err - std
                if (err + std) > common_mpd_y_limit[1]:
                    common_mpd_y_limit[1] = err + std

        # Add padding.
        common_nme_y_limit[0] -= 0.01
        common_nme_y_limit[1] += 0.01
        common_mpd_y_limit[0] -= 0.01
        common_mpd_y_limit[1] += 0.01

        nme_fig, nme_axes = plt.subplots(figsize=(7, 8), **region_nrows_ncols)
        mpd_fig, mpd_axes = plt.subplots(figsize=(7, 8), **region_nrows_ncols)

        for (
            plot_i,
            ((region_name, reg_nme_error_dict), reg_mpd_error_dict, n_points),
        ) in enumerate(
            zip(
                region_nme_data.items(),
                region_mpd_data.values(),
                n_points_map.values(),
            )
        ):
            title_label = f"({ascii_lowercase[plot_i]})"

            title = f"{region_name} (n={n_points})"

            if region_name != "desert":
                nme_ylim = common_nme_y_limit
                mpd_ylim = common_mpd_y_limit
            else:
                nme_ylim = None
                mpd_ylim = None

            error_hist(
                errors=None,
                # Show the region name and number of selected locations.
                title=title,
                title_label=title_label,
                ylabel="NME error (std of mean)"
                if not plot_i % region_nrows_ncols["ncols"]
                else None,
                error_dict=reg_nme_error_dict,
                save_path=None,
                fig=nme_fig,
                ax=nme_axes.ravel()[plot_i],
                ylim=nme_ylim,
            )

            error_hist(
                errors=None,
                # Show the region name and number of selected locations.
                title=title,
                title_label=title_label,
                ylabel="MPD error (std of mean)"
                if not plot_i % region_nrows_ncols["ncols"]
                else None,
                error_dict=reg_mpd_error_dict,
                save_path=None,
                fig=mpd_fig,
                ax=mpd_axes.ravel()[plot_i],
                ylim=mpd_ylim,
            )

        for fig in [nme_fig, mpd_fig]:
            fig.tight_layout(rect=[0, 0.0, 1, 0.98])

        nme_fig.savefig(save_dir / "regional_nme_errors.pdf")
        mpd_fig.savefig(save_dir / "regional_mpd_errors.pdf")

        for fig in [nme_fig, mpd_fig]:
            plt.close(fig)
