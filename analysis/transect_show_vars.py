#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from textwrap import wrap

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy
import numpy as np
from cf_units import Unit
from loguru import logger
from sklearn.metrics.pairwise import haversine_distances
from wildfires.data import Datasets, Ext_MOD15A2H_fPAR, MOD15A2H_LAI_fPAR

from python_inferno.configuration import land_pts, npft
from python_inferno.data import (
    load_jules_lats_lons,
    load_obs_data,
    load_single_year_cubes,
    timestep,
)
from python_inferno.plotting import use_style

# TIME, PFT, LAND

output_dir = Path("~/tmp/jules_diagnostic_graphs_transects").expanduser()
output_dir.mkdir(exist_ok=True, parents=False)


def frac_weighted_avg(data, frac):
    """Calculate the frac-weighted mean."""
    frac = frac[:, :13]  # Select vegetation PFT fractions only.

    assert len(data.shape) == len(frac.shape) == 3
    assert data.shape == frac.shape

    frac_sum = np.sum(frac, axis=1)
    sel = ~np.isclose(frac_sum, 0)

    agg_data_flat = np.sum(data * frac, axis=1)[sel] / frac_sum[sel]

    agg_data = np.ma.MaskedArray(np.zeros((frac.shape[0], frac.shape[2])), mask=True)
    agg_data[sel] = agg_data_flat

    return agg_data


def get_land_index(lat, lon, lats_arr, lons_arr):
    """Retrieve the index into `lats_arr` and `lons_arr`."""

    # Compute Haversine distances.
    distances = haversine_distances(
        np.radians(np.hstack((lats_arr.reshape(-1, 1), lons_arr.reshape(-1, 1)))),
        np.radians(np.array([[lat, lon]])),
    )
    assert distances.shape == (lats_arr.shape[0], 1)
    return np.argmin(distances[:, 0])


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    use_style()

    jules_lats, jules_lons = load_jules_lats_lons()
    jules_lats = jules_lats.points.ravel()
    jules_lons = jules_lons.points.ravel()
    assert jules_lats.shape == jules_lons.shape == (land_pts,)

    year = 2000
    temporal_subsampling = 4

    fname = (
        f"~/tmp/new6/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUPZ.Instant.{year}.nc"
    )
    data_dict = load_single_year_cubes(
        filename=fname,
        variable_name_slices={
            "leafC": (slice(None), slice(None), 0),
            "leaf_litC": (slice(None), slice(None), 0),
            "wood_litC": (slice(None), slice(None), 0),
            "root_litC": (slice(None), slice(None), 0),
            "t1p5m": (slice(None), slice(npft), 0),
            "sthu": (slice(None), 0, 0),
            "landCoverFrac": (slice(None), slice(None), 0),
            "npp_pft": (slice(None), slice(None), 0),
            "lai": (slice(None), slice(None), 0),
            "lai_bal": (slice(None), slice(None), 0),
            # "lai_phen": (slice(None), slice(None), 0),
            "g_leaf": (slice(None), slice(None), 0),
            "g_leaf_day": (slice(None), slice(None), 0),
            "g_leaf_dr_out": (slice(None), slice(None), 0),
            "g_leaf_phen": (slice(None), slice(None), 0),
            "fapar": (slice(None), slice(None), 0),
        },
        temporal_subsampling=temporal_subsampling,
    ).copy()

    # data_dict["Litter Pool"] = calc_litter_pool(
    #     filename=fname,
    #     litter_tc=1e-9,
    #     leaf_f=1e-3,
    #     verbose=False,
    #     Nt=None,
    # )[::temporal_subsampling]

    Nt = next(iter(data_dict.values())).shape[0]

    data_dict["Obs. LAI"] = load_obs_data(
        Datasets(MOD15A2H_LAI_fPAR()).select_variables("LAI").dataset,
        obs_dates=(datetime(year, 1, 1), datetime(year, 12, 31)),
        climatology=False,
        Nt=Nt,
    )
    data_dict["Obs. FAPAR"] = load_obs_data(
        Ext_MOD15A2H_fPAR(),
        obs_dates=(datetime(year, 1, 1), datetime(year, 12, 31)),
        climatology=False,
        Nt=Nt,
    )

    name_dict = {
        "leafC": "Leaf Carbon",
        "leaf_litC": "Leaf Litter",
        "wood_litC": "Wood Litter",
        "root_litC": "Root Litter",
        "t1p5m": "Temperature",
        "sthu": "Soil Moisture",
        "npp_pft": "NPP",
        "lai": "LAI",
        "lai_bal": "LAI (seasonal max)",
        # "lai_phen": "LAI after phen",
        # "g_leaf": "Leaf turnover",
        # "g_leaf_day": "Leaf turn. PHENOL",
        # "g_leaf_dr_out": "Leaf turn. TRIFFID",
        # "g_leaf_phen": "Mean leaf turn. phen",
        "fapar": "FAPAR",
    }
    name_dict = {key: "\n".join(wrap(name, 12)) for key, name in name_dict.items()}

    units_dict = {
        "leafC": r"$\mathrm{kg}\ \mathrm{m}^{-2}$",
        # "leaf_litC": r"$kg m^{-2} (360\ \mathrm{days})^{-1}$",
        # NOTE Omitted here for plotting purposes.
        # "leaf_litC": r"$\leftarrow$",
        "leaf_litC": "",
        "wood_litC": r"$\mathrm{kg}\ \mathrm{m}^{-2}\ (360\ \mathrm{days})^{-1}$",
        # NOTE Omitted here for plotting purposes.
        # "root_litC": r"$kg m^{-2} (360\ \mathrm{days})^{-1}$",
        # "root_litC": r"$\rightarrow$",
        "root_litC": "",
        "t1p5m": "K",
        "sthu": "1",
        "npp_pft": r"$\mathrm{kg}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1}$",
        "lai": "1",
        "lai_bal": "1",
        # "lai_phen": "1",
        # "Litter Pool": "1",
        "Obs. FAPAR": "1",
        "Obs. LAI": "1",
        "g_leaf": r"$(360\ \mathrm{days})^{-1}$",
        "g_leaf_day": r"$(360\ \mathrm{days})^{-1}$",
        "g_leaf_dr_out": r"$(360\ \mathrm{days})^{-1}$",
        "g_leaf_phen": r"$(360\ \mathrm{days})^{-1}$",
        "fapar": "1",
    }

    date_unit = Unit(f"seconds since {year}-01-01")
    datetimes = [
        dt._to_real_datetime()
        for dt in date_unit.num2date(
            np.arange(next(iter(data_dict.values())).shape[0])
            * timestep
            * temporal_subsampling
        )
    ]

    def set_ylabel(ax):
        ax.set_ylabel(
            f"{name_dict.get(name, name)}\n"
            + (f"({units_dict[name]})" if units_dict[name] else "")
        )

    frac = data_dict.pop("landCoverFrac")

    # Calculate frac-weighted averages if needed.
    data_dict = {
        name: frac_weighted_avg(data, frac=frac) if len(data.shape) == 3 else data
        for name, data in data_dict.items()
    }

    for lat_range, lon_range, nlat, nlon in [
        [(-45, 65), (110, 130), 5, 8],
        [(-45, 65), (10, 30), 5, 8],
        [(-45, 65), (-80, -60), 5, 8],
    ]:
        plotted_coords = []
        plotted_lat_indices = []

        lon_avg_data = defaultdict(lambda: defaultdict(dict))

        for (data_i, (name, data)) in enumerate(data_dict.items()):
            for lat_i, olat in enumerate(np.linspace(*lat_range, nlat)):
                avg_raw_data = np.ma.MaskedArray(np.zeros((nlon, Nt)), mask=True)

                for lon_i, olon in enumerate(np.linspace(*lon_range, nlon)):
                    # Get closest land point.
                    land_index = get_land_index(olat, olon, jules_lats, jules_lons)

                    avg_raw_data[lon_i] = data[:, land_index]

                    if data_i == 0:
                        # Only do this for the first variable.

                        # Get the corresponding actual lat, lon.
                        lat = jules_lats[land_index]
                        lon = jules_lons[land_index]

                        plotted_coords.append((lat, lon))
                        plotted_lat_indices.append(lat_i)

                lon_avg_data[lat_i][name] = np.ma.average(avg_raw_data, axis=0)

        def plot_map(map_ax):
            map_ax.set_global()
            map_ax.coastlines(linewidth=0.2)
            map_ax.gridlines(zorder=0, alpha=0.4, linestyle="--", linewidth=0.2)

            for (plot_i, (p_lat, p_lon)) in zip(plotted_lat_indices, plotted_coords):
                map_ax.plot(
                    p_lon,
                    p_lat,
                    marker="x",
                    color=f"C{plot_i}",
                    transform=ccrs.PlateCarree(),
                    alpha=0.6,
                )

        # Save setup.
        loc_str = (
            f"{lat_range[0]:0.2f}_{lat_range[1]:0.2f}__"
            f"{lon_range[0]:0.2f}_{lon_range[1]:0.2f}"
        )

        loc_dir = output_dir / loc_str
        loc_dir.mkdir(exist_ok=True, parents=False)

        # Title with location information.
        title_str = f"{lon_range[0]:.2f}°E - {lon_range[1]:.2f}°E\n" + ", ".join(
            [f"{lat:0.2f}°N" for lat, _ in plotted_coords[::nlon]]
        )

        ncols = 2
        nrows = math.ceil(len(data_dict) / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=True,
            figsize=(4.5 * ncols, (7.0 / 6.0) * (nrows + 1)),
        )
        for ax, name in zip(axes.T.ravel(), data_dict):
            for lat_i in range(nlat):
                ax.plot(datetimes, lon_avg_data[lat_i][name], c=f"C{lat_i}", alpha=0.6)
            set_ylabel(ax)

        # (left, bottom, right, top)
        fig.tight_layout(rect=(0, 0.05, 1.0, 0.85))

        plt.text(
            0.33, 0.95, title_str, ha="center", va="top", transform=fig.transFigure
        )
        fig.align_ylabels()
        # plt.setp(ax.get_xticklabels(), ha="right", rotation=45)

        # [left, bottom, width, height]
        plot_map(plt.axes([0.6, 0.85, 0.35, 0.14], projection=ccrs.Robinson()))

        # Save.
        fig.savefig(loc_dir / "all_variables.png")
        plt.close(fig)

        # Single variables per plot.

        for name in data_dict:
            fig, ax = plt.subplots(1, 1, figsize=(4.5, 14.0 / 6.0))

            for lat_i in range(nlat):
                ax.plot(datetimes, lon_avg_data[lat_i][name], c=f"C{lat_i}", alpha=0.6)
            set_ylabel(ax)

            # (left, bottom, right, top)
            fig.tight_layout(rect=(0, 0.05, 1.0, 0.74))

            plt.text(
                0.33, 0.95, title_str, ha="center", va="top", transform=fig.transFigure
            )
            fig.align_ylabels()

            # [left, bottom, width, height]
            plot_map(plt.axes([0.6, 0.75, 0.4, 0.24], projection=ccrs.Robinson()))

            fig.savefig(loc_dir / f"{name}.png")
            plt.close(fig)
