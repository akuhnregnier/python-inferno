#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from sklearn.metrics import r2_score
from wildfires.analysis import cube_plotting

from python_inferno.cache import cache
from python_inferno.data import load_data, load_jules_lats_lons
from python_inferno.plotting import use_style
from python_inferno.utils import frac_weighted_mean, monthly_average_data


@cache
def get_plot_data(*, fapar_diag_pft, frac, obs_fapar_1d, npp_gb):
    jules_fapar_1d = frac_weighted_mean(data=fapar_diag_pft, frac=frac)
    return (jules_fapar_1d, obs_fapar_1d, npp_gb)


def calculate_spatial_r2s(*, y_true, y_pred):
    assert len(y_true.shape) == 2
    assert y_true.shape == y_pred.shape

    r2_scores_1d = np.ma.MaskedArray(np.zeros(y_true.shape[1]), mask=True)

    for i in range(y_true.shape[1]):
        try:
            r2_scores_1d[i] = r2_score(y_true=y_true[:, i], y_pred=y_pred[:, i])
        except ValueError:
            pass

    return r2_scores_1d


@cache
def _time_coord_passthrough(time_coord):
    return time_coord


if __name__ == "__main__":
    use_style()

    jules_lats, jules_lons = load_jules_lats_lons()

    def convert_to_2d(data_1d):
        assert len(data_1d.shape) == 1
        return cube_1d_to_2d(
            get_1d_data_cube(data_1d, lats=jules_lats, lons=jules_lons)
        )

    (
        t1p5m_tile,
        q1p5m_tile,
        pstar,
        sthu_soilt_single,
        frac,
        c_soil_dpm_gb,
        c_soil_rpm_gb,
        canht,
        ls_rain,
        con_rain,
        fuel_build_up,
        fapar_diag_pft,
        _,
        _,
        gfed_ba_1d,
        obs_fapar_1d,
        jules_ba_gb,
        obs_pftcrop_1d,
        _jules_time_coord,
        npp_pft,
        npp_gb,
        is_climatology,
    ) = load_data(N=None)

    jules_time_coord = _time_coord_passthrough(_jules_time_coord)

    (jules_fapar_1d, obs_fapar_1d, npp_gb) = get_plot_data(
        fapar_diag_pft=fapar_diag_pft,
        frac=frac,
        obs_fapar_1d=obs_fapar_1d,
        npp_gb=npp_gb,
    )

    mon_obs_fapar_1d = monthly_average_data(obs_fapar_1d, time_coord=jules_time_coord)
    mon_jules_fapar_1d = monthly_average_data(
        jules_fapar_1d, time_coord=jules_time_coord
    )
    mon_jules_npp_1d = monthly_average_data(npp_gb, time_coord=jules_time_coord)

    mean_jules_fapar_2d = convert_to_2d(np.mean(mon_jules_fapar_1d, axis=0))
    mean_jules_npp_2d = convert_to_2d(np.mean(mon_jules_npp_1d, axis=0))
    mean_obs_fapar_2d = convert_to_2d(np.mean(mon_obs_fapar_1d, axis=0))
    # NaNs.
    mean_jules_fapar_2d.data.mask |= np.isnan(mean_jules_fapar_2d.data.data)
    mean_jules_npp_2d.data.mask |= np.isnan(mean_jules_npp_2d.data.data)

    fapar_r2_scores_1d = calculate_spatial_r2s(
        y_true=mon_obs_fapar_1d, y_pred=mon_jules_fapar_1d
    )
    npp_r2_scores_1d = calculate_spatial_r2s(
        y_true=mon_obs_fapar_1d, y_pred=mon_jules_npp_1d
    )

    fapar_r2_2d = convert_to_2d(fapar_r2_scores_1d)
    npp_r2_2d = convert_to_2d(npp_r2_scores_1d)

    # Plotting.

    plt.ioff()

    ###

    # cube_plotting(mean_jules_fapar_2d, title="JULES")

    ###

    # cube_plotting(mean_obs_fapar_2d, title="Obs")

    ###

    fig, axes = plt.subplots(
        2,
        1,
        subplot_kw={"projection": ccrs.Robinson()},
        squeeze=True,
        figsize=(5, 4.5),
    )

    def get_common_kwargs(cax):
        return dict(
            title="",
            boundaries=[-10, -1, 0, 0.2, 0.4],
            fig=fig,
            colorbar_kwargs=dict(label=r"$\mathrm{R}^2$", cax=cax),
        )

    axes[0].set_title(r"Obs FAPAR $\leftrightarrow$ JULES FAPAR")
    axes[1].set_title(r"Obs FAPAR $\leftrightarrow$ JULES NPP")

    cbar_width = 0.01
    cbar_height = 0.25
    cbar_x = 0.85

    cube_plotting(
        fapar_r2_2d,
        ax=axes[0],
        **get_common_kwargs(fig.add_axes([cbar_x, 0.58, cbar_width, cbar_height]))
    )
    cube_plotting(
        npp_r2_2d,
        ax=axes[1],
        **get_common_kwargs(fig.add_axes([cbar_x, 0.16, cbar_width, cbar_height]))
    )

    axes[0].text(0.01, 1.05, "(a)", transform=axes[0].transAxes)
    axes[1].text(0.01, 1.05, "(b)", transform=axes[1].transAxes)

    fig.savefig(Path("~/tmp/r2_fapar_npp.png").expanduser())

    ###

    # plt.figure()
    # plt.hexbin(jules_fapar_1d.ravel(), obs_fapar_1d.ravel())
    # plt.xlabel("jules fapar")
    # plt.ylabel("obs fapar")
