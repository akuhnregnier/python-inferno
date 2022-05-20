#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.widgets import Slider
from numba import njit, prange
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from tqdm import tqdm

from python_inferno.basinhopping import BasinHoppingSpace
from python_inferno.configuration import N_pft_groups
from python_inferno.data import (
    handle_param,
    key_cached_calculate_grouped_vpd,
    key_cached_precip_moving_sum,
    load_single_year_cubes,
    timestep,
)
from python_inferno.space import generate_space


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def fewer_calculate_grouped_dry_bal(
    *,
    grouped_vpd,
    cum_rain,
    rain_f,
    vpd_f,
    # NOTE This is where the output is placed and should be an (Nt, N_pft_groups,
    # land_pts) np.float64 array.
    out,
):
    Nt = grouped_vpd.shape[0]

    assert rain_f.shape[0] == N_pft_groups
    assert vpd_f.shape[0] == N_pft_groups
    assert len(out.shape) == 3
    assert out.shape[:2] == (Nt, N_pft_groups)

    for l in prange(out.shape[2]):
        for ti in range(Nt):
            for i in range(N_pft_groups):
                if ti == 0:
                    prev_dry_bal = 0
                else:
                    prev_dry_bal = out[ti - 1, i, l]

                vpd_val = grouped_vpd[ti, i, l]

                new_dry_bal = (
                    prev_dry_bal
                    + rain_f[i] * cum_rain[ti, l]
                    - (1 - np.exp(-vpd_f[i] * vpd_val))
                )

                if new_dry_bal < -1.0:
                    out[ti, i, l] = -1.0
                elif new_dry_bal > 1.0:
                    out[ti, i, l] = 1.0
                else:
                    out[ti, i, l] = new_dry_bal

    return out


def get_fewer_climatological_grouped_dry_bal(
    *,
    filenames=tuple(
        str(Path(s).expanduser())
        for s in (
            "~/tmp/new-with-antec5/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUPD0.Instant.2010.nc",
            "~/tmp/new-with-antec5/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUPD0.Instant.2011.nc",
        )
    ),
    rain_f,
    vpd_f,
    verbose=False,
    points=5,
):
    """Load instantaneous values from the files below, then calculate dry_bal, then
    perform climatological averaging."""

    rain_f = handle_param(rain_f)
    vpd_f = handle_param(vpd_f)

    clim_dry_bal = None
    n_avg = 0

    # Array used to store and calculate dry_bal.
    grouped_dry_bal = None

    for f in tqdm(
        list(map(str, filenames)), desc="Processing dry_bal", disable=not verbose
    ):
        data_dict = load_single_year_cubes(
            filename=f,
            variable_name_slices={
                "t1p5m": (slice(None), slice(None), 0),
                "q1p5m": (slice(None), slice(None), 0),
                "pstar": (slice(None), 0),
                "ls_rain": (slice(None), 0),
                "con_rain": (slice(None), 0),
            },
        )

        grouped_vpd = key_cached_calculate_grouped_vpd(
            t1p5m_tile=data_dict["t1p5m"],
            q1p5m_tile=data_dict["q1p5m"],
            pstar=data_dict["pstar"],
            # NOTE Special key used to store and retrieve memoized results.
            cache_key=f,
        )
        cum_rain = key_cached_precip_moving_sum(
            ls_rain=data_dict["ls_rain"],
            con_rain=data_dict["con_rain"],
            timestep=timestep,
            # NOTE Special key used to store and retrieve memoized results.
            cache_key=f,
        )

        if grouped_dry_bal is None:
            # This array only needs to be allocated once for the first file.
            grouped_dry_bal = np.zeros(
                (data_dict["pstar"].shape[0], N_pft_groups, points), dtype=np.float64
            )

        # Calculate grouped dry_bal.
        grouped_dry_bal = fewer_calculate_grouped_dry_bal(
            grouped_vpd=grouped_vpd,
            cum_rain=cum_rain,
            rain_f=rain_f,
            vpd_f=vpd_f,
            out=grouped_dry_bal,
        )
        if clim_dry_bal is None:
            clim_dry_bal = grouped_dry_bal
            assert n_avg == 0
        else:
            clim_dry_bal += grouped_dry_bal
        n_avg += 1

    return clim_dry_bal / n_avg


space_template = dict(
    rain_f=(1, [(0.5, 20.0)], float),
    vpd_f=(1, [(1, 5000)], float),
)

space = BasinHoppingSpace(generate_space(space_template))


# Histogram bins for `loss1`.
bins = np.linspace(-1, 1, 20)
# Need equally spaced bins.
bin_diff = np.diff(bins)[0]
assert_allclose(np.diff(bins), bin_diff)


# Histogram bins for `loss1`.
diff_bins = np.linspace(-2, 2, 10)
# Need equally spaced bins.
diff_bin_diff = np.diff(diff_bins)[0]
assert_allclose(np.diff(diff_bins), diff_bin_diff)


@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def calc_loss1(*, dry_bal, bins, hists):
    for i in prange(dry_bal.shape[1]):
        hists[i] = np.histogram(dry_bal[:, i], bins=bins)[0]
    hists /= dry_bal.shape[0] * (bins[1] - bins[0])
    # Minimise the amount of variation between bins - all values should be represented
    # as equally as possible.
    # Normalise this metric by the number of samples.
    return np.linalg.norm(hists - 0.5) / np.sqrt(hists.size)


def get_to_optimise(*, loss1_c, loss2_c, loss3_c):
    def to_optimise(x):
        dry_bal = get_fewer_climatological_grouped_dry_bal(
            **space.inv_map_float_to_0_1(dict(zip(space.continuous_param_names, x))),
            verbose=False,
            points=5,
        )
        # Select a single PFT since we are only using single parameters.
        dry_bal = dry_bal[:, 0]
        assert len(dry_bal.shape) == 2

        loss1 = calc_loss1(
            dry_bal=dry_bal,
            bins=bins,
            hists=np.empty((dry_bal.shape[1], bins.size - 1)),
        )
        loss3 = calc_loss1(
            dry_bal=np.diff(dry_bal, axis=0),
            bins=diff_bins,
            hists=np.empty((dry_bal.shape[1] - 1, diff_bins.size - 1)),
        )

        # At the same time, the `dry_bal` variable should fluctuate between high and low
        # values (instead of e.g. monotonically increasing).
        # Add a factor to enhance the weight of this metric.
        loss2 = abs(
            (
                (
                    np.sum(np.diff(dry_bal, axis=0) < 0)
                    / ((dry_bal.shape[0] - 1) * dry_bal.shape[1])
                )
                - 0.5
            )
        )
        c_arr = np.array([loss1_c, loss2_c, loss3_c])
        c_arr /= np.sum(c_arr)

        loss_arr = np.array([loss1, loss2, loss3])

        # logger.info(f"cs: {','.join(map(str, c_arr))}")

        return np.sum(c_arr * loss_arr)

    return to_optimise


if __name__ == "__main__":

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()

    def plot_line(x):
        dry_bal = get_fewer_climatological_grouped_dry_bal(
            **space.inv_map_float_to_0_1(dict(zip(space.continuous_param_names, x))),
            verbose=False,
            points=5,
        )
        # Select a single PFT since we are only using single parameters.
        dry_bal = dry_bal[:, 0]
        ys = dry_bal[:, 0]
        if not hasattr(plot_line, "line"):
            (line,) = ax.plot(ys)
            plot_line.line = line
        else:
            plot_line.line.set_ydata(ys)

    plot_line(space.float_x0_mid)
    ax.set_xlabel("Time [s]")

    axcolor = "0.2"
    ax.margins(x=0)

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(bottom=0.4)

    def get_slider(*, ypos, label, valmin=0.0, valmax=1.0, valinit=0.5):
        slider_ax = plt.axes([0.25, ypos, 0.65, 0.03], facecolor=axcolor)
        return Slider(
            ax=slider_ax,
            label=label,
            valmin=valmin,
            valmax=valmax,
            valinit=valinit,
        )

    loss1_slider = get_slider(ypos=0.1, label="loss1")
    loss2_slider = get_slider(ypos=0.15, label="loss2")
    loss3_slider = get_slider(ypos=0.2, label="loss3")

    # The function to be called anytime a slider's value changes
    def update(val):
        logger.info("Starting minimisation")
        result = minimize(
            get_to_optimise(
                loss1_c=loss1_slider.val,
                loss2_c=loss2_slider.val,
                loss3_c=loss3_slider.val,
            ),
            x0=space.float_x0_mid,
            method="L-BFGS-B",
            jac=None,
            bounds=[(0, 1)] * len(space.continuous_param_names),
            options=dict(maxfun=1000, ftol=1e-6, eps=1e-4, disp=True),
        )
        opt_x = result.x
        logger.info(f"Completed minimisation, success: {result.success}.")

        plot_line(opt_x)
        fig.canvas.draw_idle()

    # register the update function with each slider
    for slider in (loss1_slider, loss2_slider, loss3_slider):
        slider.on_changed(update)

    plt.show()
