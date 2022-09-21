#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from string import ascii_lowercase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from python_inferno.configuration import scheme_name_map
from python_inferno.iter_opt import iterative_ba_model_opt, vis_result
from python_inferno.model_params import get_model_params
from python_inferno.plotting import use_style
from python_inferno.sinferno_mcmc_results import get_sinferno_mcmc_results
from python_inferno.spotpy_mcmc import get_cached_mcmc_r_hat


def split_axis_setup(*, top_ax, bottom_ax, top_ylim, bottom_ylim):
    top_ax.set_ylim(top_ylim)
    bottom_ax.set_ylim(bottom_ylim)

    # Hide the spines between the axes.
    top_ax.spines["bottom"].set_visible(False)
    bottom_ax.spines["top"].set_visible(False)
    top_ax.xaxis.tick_top()
    top_ax.tick_params(labeltop=False)  # don't put tick labels at the top
    top_ax.xaxis.set_ticks_position(
        "none"
    )  # hide top ticks themselves (not just labels)

    bottom_ax.xaxis.tick_bottom()

    # Cut-out slanted lines.

    kwargs = dict(
        marker=[(-1, -0.5), (1, 0.5)],
        markersize=8,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    top_ax.plot([0, 1], [0, 0], transform=top_ax.transAxes, **kwargs)
    bottom_ax.plot([0, 1], [1, 1], transform=bottom_ax.transAxes, **kwargs)


class SplitAxes:
    def __init__(self, *, fig, top_ax, bottom_ax, top_ylim, bottom_ylim, ylabelpad):
        split_axis_setup(
            top_ax=top_ax,
            bottom_ax=bottom_ax,
            top_ylim=top_ylim,
            bottom_ylim=bottom_ylim,
        )

        self.fig = fig

        self.top_ax = top_ax
        self.bottom_ax = bottom_ax
        self.top_ylim = top_ylim
        self.bottom_ylim = bottom_ylim

        self.ylabelpad = ylabelpad

    @property
    def axes(self):
        return (self.top_ax, self.bottom_ax)

    def set_title(self, title):
        self.top_ax.set_title(title)

    def plot(self, *args, **kwargs):
        for ax in self.axes:
            ax.plot(*args, **kwargs)

    def hlines(self, *args, **kwargs):
        for ax in self.axes:
            ax.hlines(*args, **kwargs)

    def legend(self, *args, **kwargs):
        self.top_ax.legend(*args, **kwargs)

    def set_ylabel(self, *args, **kwargs):
        tl, tb, tw, th = self.top_ax.get_position().bounds
        bl, bb, bw, bh = self.bottom_ax.get_position().bounds

        # Determine centred position to place the label at.
        x = tl - self.ylabelpad
        y = (bb + tb + th) / 2.0

        self.top_ax.text(
            x,
            y,
            *args,
            **kwargs,
            transform=self.fig.transFigure,
            rotation="vertical",
            va="baseline",
            ha="center",
            rotation_mode="anchor",
        )

    def set_xlabel(self, *args, **kwargs):
        self.bottom_ax.set_xlabel(*args, **kwargs)

    def annotate(self, *args, **kwargs):
        self.bottom_ax.annotate(*args, **kwargs)


def setup_quad_split_axes(
    *,
    figsize,
    width=0.45,
    height=0.44,
    h_pad=0.008,
    height_ratio=3.0,
    top_ylim,
    bottom_ylim,
    ylabelpad=0.08,
    panel_labels=True,
):
    fig = plt.figure(figsize=figsize)
    split_axes = []

    eff_height = height - h_pad

    # Generate 4 pairs of axes.
    for (i, ((c_x, c_y), letter)) in enumerate(
        zip(
            (
                [0.25, 0.75],
                [0.75, 0.75],
                [0.25, 0.25],
                [0.75, 0.25],
            ),
            ascii_lowercase,
        )
    ):
        bottom_left = (c_x - width / 2.0, c_y - width / 2.0)

        bottom_height = eff_height * (height_ratio / (height_ratio + 1))
        top_height = eff_height * (1 / (height_ratio + 1))

        bottom_ax = fig.add_axes(
            [
                bottom_left[0],
                bottom_left[1],
                width,
                bottom_height,
            ]
        )
        top_ax = fig.add_axes(
            [
                bottom_left[0],
                bottom_left[1] + bottom_height + h_pad,
                width,
                top_height,
            ]
        )

        if panel_labels:
            tl, tb, tw, th = top_ax.get_position().bounds
            bl, bb, bw, bh = bottom_ax.get_position().bounds

            # Determine centred position to place the label at.
            x = tl
            y = tb + th + 0.018 * (th + bh)

            top_ax.text(
                x,
                y,
                f"({letter})",
                transform=fig.transFigure,
            )

        if (i + 1) % 2 == 0:
            for ax in (top_ax, bottom_ax):
                ax.tick_params(labelleft=False)

        if i // 2 == 0:
            for ax in (top_ax, bottom_ax):
                ax.tick_params(labelbottom=False)

        split_axes.append(
            SplitAxes(
                fig=fig,
                top_ax=top_ax,
                bottom_ax=bottom_ax,
                top_ylim=top_ylim,
                bottom_ylim=bottom_ylim,
                ylabelpad=ylabelpad,
            )
        )

    return fig, split_axes


if __name__ == "__main__":
    use_style()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    save_dir = Path("~/tmp/iter-opt-models").expanduser()
    save_dir.mkdir(parents=False, exist_ok=True)

    df, method_iter = get_model_params(progress=False, verbose=False)

    fig, split_axes = setup_quad_split_axes(
        figsize=(5.5, 5.5),
        top_ylim=(1.255, 1.38),
        bottom_ylim=(0.74, 0.99),
        panel_labels=True,
    )

    df_data_methods = {}

    for (method_index, (split_ax, method_data)) in enumerate(
        zip(split_axes, method_iter())
    ):
        full_opt_loss = method_data[4]
        params = method_data[5]
        method_name = method_data[6]

        mcmc_r_hat = get_cached_mcmc_r_hat(method_index=method_index)
        mcmc_min_result = abs(
            get_sinferno_mcmc_results(method_index=method_index, r_hat=mcmc_r_hat)[
                "Maximal objective value"
            ]
        )

        split_ax.set_title(f"SINFERNO-{scheme_name_map[method_name]}")

        method_dir = save_dir / method_name
        method_dir.mkdir(parents=False, exist_ok=True)

        df_data_opts = {}

        for key, marker, colors, opt_kwargs in (
            ("50,1", "^", ("C0", "C1"), dict(maxiter=50, niter_success=1)),
            ("1000,5", "x", ("C2", "C3"), dict(maxiter=1000, niter_success=5)),
        ):
            min_params_dict = {}

            results, aic_results, cv_results = iterative_ba_model_opt(
                params=params, **opt_kwargs
            )

            n_params, aics, arcsinh_aics = zip(
                *(
                    (n_params, data["aic"], data["arcsinh_aic"])
                    for (n_params, data) in aic_results.items()
                )
            )
            aics = np.asarray(aics)
            arcsinh_aics = np.asarray(arcsinh_aics)

            opt_dir = method_dir / key.replace(",", "_")
            opt_dir.mkdir(parents=False, exist_ok=True)

            for n, result in tqdm(results.items(), desc="Plotting model vis"):
                vis_result(
                    result=result,
                    dryness_method=int(params["dryness_method"]),
                    fuel_build_up_method=int(params["fuel_build_up_method"]),
                    save_key=str(n),
                    save_dir=opt_dir,
                )

            n_params, losses = zip(
                *((n, float(loss)) for (n, (loss, _)) in results.items())
            )

            ms = 6
            aic_ms = 8

            split_ax.plot(
                n_params,
                losses,
                marker=marker,
                linestyle="",
                label=key,
                ms=ms,
                c=colors[0],
                zorder=2,
            )

            min_aic_i = np.argmin(aics)
            min_arcsinh_aic_i = np.argmin(arcsinh_aics)

            print("AIC indices:", min_aic_i, min_arcsinh_aic_i)

            min_params_dict["SSE AIC"] = str(n_params[min_aic_i])
            min_params_dict["arcsinh-SSE AIC"] = str(n_params[min_arcsinh_aic_i])

            print(
                f"{key} SSE AIC",
                "params:",
                n_params[min_aic_i],
                "loss:",
                losses[min_aic_i],
            )

            print(
                f"{key} arcsinh-SSE AIC",
                "params:",
                n_params[min_arcsinh_aic_i],
                "loss:",
                losses[min_arcsinh_aic_i],
            )

            # TODO put these values into a table, along with the other mins positions

            # CV plotting.
            cv_n_params, cv_test_losses = zip(*cv_results.items())
            split_ax.plot(
                cv_n_params,
                cv_test_losses,
                marker=marker,
                linestyle="",
                label=f"{key} CV",
                ms=ms,
                c=colors[1],
                zorder=2,
            )

            print(scheme_name_map[method_name], key)
            for (n, (loss, _)) in results.items():
                if n in cv_results:
                    print(n, cv_results[n])

            min_cv_loss = np.min(cv_test_losses)
            min_cv_params = np.array(cv_n_params)[
                np.where((np.array(cv_test_losses) - min_cv_loss) < 1e-8)
            ]
            if min_cv_params.size > 1:
                assert np.all(np.diff(min_cv_params) == 1)

            min_params_dict["CV"] = (
                str(min_cv_params[0])
                if min_cv_params.size == 1
                else f"{min_cv_params[0]}--{min_cv_params[-1]}"
            )

            df_data_opts[f"\lc{{}} {key}"] = min_params_dict

        df_data_methods[scheme_name_map[method_name]] = df_data_opts

        split_ax.hlines(
            full_opt_loss,
            0,
            max(n_params),
            colors="C6",
            label="1000,15",
            linestyles="--",
            zorder=1,
        )
        split_ax.hlines(
            mcmc_min_result,
            0,
            max(n_params),
            colors="C9",
            label="DREAM MCMC",
            linestyles="-.",
            zorder=1,
        )

        if method_index == 0:
            split_ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                bbox_transform=fig.transFigure,
                ncol=6,
            )

        if method_index in (0, 2):
            split_ax.set_ylabel("Performance (loss)")
        if method_index in (2, 3):
            split_ax.set_xlabel("Nr. of Optimised Parameters")

    fig.savefig(save_dir / "iter_model_perf.pdf")
    plt.close(fig)

    df = pd.concat({k: pd.DataFrame(v).T for k, v in df_data_methods.items()}, axis=0)
    df.index = df.index.set_names(["Scheme", "Optimisation"])

    for col in df.columns:
        for i in range(0, df[col].size, 2):
            df[col].values[i] = "\lc{} " + df[col].values[i]

    print(
        df.style.to_latex(
            hrules=True, position_float="centering", siunitx=True, caption=("", "")
        )
    )
