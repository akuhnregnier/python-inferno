#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import numpy as np
from cf_units import Unit
from jules_output_analysis.utils import PFTs, pft_acronyms, pft_names
from loguru import logger
from tqdm import tqdm
from wildfires.data import Ext_MOD15A2H_fPAR

from python_inferno.configuration import land_pts, npft
from python_inferno.data import (
    calc_litter_pool,
    load_jules_lats_lons,
    load_obs_data,
    load_single_year_cubes,
    timestep,
)

output_dir = Path("~/tmp/jules_diagnostic_graphs").expanduser()
filtered_output_dir = Path("~/tmp/jules_diagnostic_graphs_filtered").expanduser()
output_dir.mkdir(exist_ok=True, parents=False)
filtered_output_dir.mkdir(exist_ok=True, parents=False)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    mpl.rc_file(Path(__file__).absolute().parent / "matplotlibrc")

    jules_lats, jules_lons = load_jules_lats_lons()
    jules_lats = jules_lats.points.ravel()
    jules_lons = jules_lons.points.ravel()
    assert jules_lats.shape == jules_lons.shape == (land_pts,)
    year = 2000
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
            "g_leaf": (slice(None), slice(None), 0),
            "g_leaf_day": (slice(None), slice(None), 0),
            "g_leaf_dr_out": (slice(None), slice(None), 0),
            "g_leaf_phen": (slice(None), slice(None), 0),
        },
    )

    data_dict["Litter Pool"] = calc_litter_pool(
        filename=fname,
        litter_tc=1e-9,
        leaf_f=1e-3,
        verbose=False,
        Nt=None,
    )
    data_dict["Obs. FAPAR"] = load_obs_data(
        Ext_MOD15A2H_fPAR(),
        obs_dates=(datetime(year, 1, 1), datetime(year, 12, 31)),
        climatology=False,
        Nt=next(iter(data_dict.values())).shape[0],
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
        "g_leaf": "Leaf turnover",
        "g_leaf_day": "Leaf turn. PHENOL",
        "g_leaf_dr_out": "Leaf turn. TRIFFID",
        "g_leaf_phen": "Mean leaf turn. phen",
    }

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
        "Litter Pool": "1",
        "Obs. FAPAR": "1",
        "g_leaf": r"$(360\ \mathrm{days})^{-1}$",
        "g_leaf_day": r"$(360\ \mathrm{days})^{-1}$",
        "g_leaf_dr_out": r"$(360\ \mathrm{days})^{-1}$",
        "g_leaf_phen": r"$(360\ \mathrm{days})^{-1}$",
    }

    date_unit = Unit(f"seconds since {year}-01-01")
    datetimes = [
        dt._to_real_datetime()
        for dt in date_unit.num2date(
            np.arange(next(iter(data_dict.values())).shape[0]) * timestep
        )
    ]

    def param_iter():
        for land_index in np.random.default_rng(0).choice(land_pts, size=50):
            for pft_index in range(npft):
                yield land_index, pft_index

    plot_data = {
        name: data for name, data in data_dict.items() if name not in ("landCoverFrac",)
    }
    frac = data_dict["landCoverFrac"]

    for land_index, pft_index in tqdm(list(param_iter()), desc="Plotting"):
        lat = jules_lats[land_index]
        lon = jules_lons[land_index]
        pft_name = pft_names[PFTs.VEG13][pft_index]
        pft_acronym = pft_acronyms[PFTs.VEG13][pft_index]

        fig, axes = plt.subplots(
            nrows=len(plot_data),
            sharex=True,
            figsize=(4.5, (7.0 / 6.0) * len(plot_data)),
        )
        for (ax, (name, data)) in zip(axes.ravel(), plot_data.items()):
            sdata = (
                data[:, pft_index, land_index]
                if len(data.shape) == 3
                else data[:, land_index]
            )
            ax.plot(datetimes, sdata)
            ax.set_ylabel(
                f"{name_dict.get(name, name)}\n"
                + (f"({units_dict[name]})" if units_dict[name] else "")
            )

        # (left, bottom, right, top)
        fig.tight_layout(rect=(0, 0.05, 1.0, 0.85))

        print_lon = abs(((lon + 180) % 360) - 180)
        print_lon_symb = "E" if np.isclose(print_lon, lon) else "W"

        print_lat = abs(lat)
        print_lat_symb = "N" if lat >= 0 else "S"

        mean_frac = np.mean(frac[:, pft_index, land_index], axis=0)
        std_frac = np.std(frac[:, pft_index, land_index], axis=0)

        plt.text(
            0.33,
            0.95,
            f"{print_lat:.2f}°{print_lat_symb}, {print_lon:.2f}°{print_lon_symb}"
            f"\n{pft_name}"
            f"\nfrac: {mean_frac:0.3f}±{std_frac:0.3f}",
            ha="center",
            va="top",
            transform=fig.transFigure,
        )
        fig.align_ylabels()
        plt.setp(ax.get_xticklabels(), ha="right", rotation=45)

        # [left, bottom, width, height]
        map_ax = plt.axes([0.6, 0.85, 0.35, 0.14], projection=ccrs.Robinson())
        map_ax.set_global()
        map_ax.coastlines(linewidth=0.2)
        map_ax.gridlines(zorder=0, alpha=0.4, linestyle="--", linewidth=0.2)
        map_ax.plot(
            lon,
            lat,
            marker="x",
            color="C1",
            transform=ccrs.PlateCarree(),
        )

        sub_dir = output_dir / f"lat_{lat:.2f}_lon_{lon:.2f}"
        sub_dir.mkdir(exist_ok=True, parents=False)

        fig.savefig(sub_dir / f"lat_{lat:.2f}_lon_{lon:.2f}_pft_{pft_acronym}.png")

        if mean_frac >= 1e-3:
            filtered_sub_dir = filtered_output_dir / f"lat_{lat:.2f}_lon_{lon:.2f}"
            filtered_sub_dir.mkdir(exist_ok=True, parents=False)

            fig.savefig(
                filtered_sub_dir / f"lat_{lat:.2f}_lon_{lon:.2f}_pft_{pft_acronym}.png"
            )
        plt.close(fig)
