# -*- coding: utf-8 -*-
from pathlib import Path

import cartopy.crs as ccrs
import dask.array as darray
import iris
import matplotlib.pyplot as plt
import numpy as np
from jules_output_analysis.data import (
    cube_1d_to_2d,
    get_1d_to_2d_indices,
    n96e_lats,
    n96e_lons,
)
from jules_output_analysis.utils import convert_longitudes
from tqdm.auto import tqdm
from wildfires.analysis import cube_plotting
from wildfires.data import Ext_MOD15A2H_fPAR, GFEDv4

from python_inferno import inferno_io
from python_inferno.configuration import land_pts
from python_inferno.utils import (
    repeated_exponential_average,
    temporal_nearest_neighbour_interp,
)


def plot_comparison(jules_ba_gb, python_ba_gb, label="BA", title="", mask_cube=None):
    # Compare to the values calculated within JULES.
    jules_ba_2d = cube_1d_to_2d(jules_ba_gb)

    def add_mask(data):
        if mask_cube is not None:
            mask_2d = cube_1d_to_2d(mask_cube).data
            if isinstance(data, np.ndarray):
                data = np.ma.MaskedArray(data, mask=mask_2d)
            elif isinstance(data, iris.cube.Cube):
                data.data = np.ma.MaskedArray(data.data, mask=mask_2d)
            else:
                raise ValueError(f"Unknown type {type(data)}.")
        return data

    # Set up the plots.
    fig, axes = plt.subplots(
        2, 2, subplot_kw=dict(projection=ccrs.Robinson()), figsize=(13.4, 6.9)
    )

    common_kwargs = dict(colorbar_kwargs=dict(label=label), title=title)

    ax = axes[0, 0]
    cube_plotting(
        add_mask(jules_ba_2d),
        ax=ax,
        **common_kwargs,
    )
    ax.set_title("JULES BA")

    ax = axes[0, 1]
    cube_plotting(
        add_mask(cube_1d_to_2d(python_ba_gb["normal"])),
        ax=ax,
        **common_kwargs,
    )
    ax.set_title("Python BA")

    ax = axes[1, 0]
    try:
        cube_plotting(
            add_mask(cube_1d_to_2d(python_ba_gb["new"])),
            ax=ax,
            **common_kwargs,
        )
    except AssertionError:
        print("Could not plot 'new'.")
    ax.set_title("Python BA with new Flamm.")

    ax = axes[1, 1]
    cube_plotting(
        add_mask(cube_1d_to_2d(python_ba_gb["new_obs_fapar"])),
        ax=ax,
        **common_kwargs,
    )
    ax.set_title("Python BA with new Flamm. & Obs FAPAR")


def get_1d_data_cube(data, lats, lons):
    # 1D JULES land points only.
    assert data.shape == (7771,)
    cube = iris.cube.Cube(data[np.newaxis])
    cube.add_aux_coord(lats, data_dims=(0, 1))
    cube.add_aux_coord(lons, data_dims=(0, 1))
    return cube[0]


def run_inferno(
    t1p5m_tile,
    q1p5m_tile,
    pstar,
    sthu_soilt,
    frac,
    c_soil_dpm_gb,
    c_soil_rpm_gb,
    canht,
    ls_rain,
    con_rain,
    pop_den,
    flash_rate,
    ignition_method,
    fuel_build_up,
    fapar_diag_pft,
    fapar_factor,
    fapar_centre,
    fuel_build_up_factor,
    fuel_build_up_centre,
    temperature_factor,
    temperature_centre,
    jules_lats,
    jules_lons,
    obs_fapar_1d,
    obs_fuel_build_up_1d,
):
    inferno_kwargs = dict(
        t1p5m_tile=t1p5m_tile,
        q1p5m_tile=q1p5m_tile,
        pstar=pstar,
        sthu_soilt=sthu_soilt,
        frac=frac,
        c_soil_dpm_gb=c_soil_dpm_gb,
        c_soil_rpm_gb=c_soil_rpm_gb,
        canht=canht,
        ls_rain=ls_rain,
        con_rain=con_rain,
        # Not used for ignition mode 1.
        pop_den=pop_den,
        flash_rate=flash_rate,
        ignition_method=ignition_method,
        fuel_build_up=fuel_build_up,
        fapar_diag_pft=fapar_diag_pft,
        fapar_factor=fapar_factor,
        fapar_centre=fapar_centre,
        fuel_build_up_factor=fuel_build_up_factor,
        fuel_build_up_centre=fuel_build_up_centre,
        temperature_factor=temperature_factor,
        temperature_centre=temperature_centre,
    )

    python_ba_gb = {
        "normal": inferno_io(
            **inferno_kwargs,
            # 1 - old, 2 - new flammability calculation
            flammability_method=1,
        )[0],
        "new": inferno_io(
            **inferno_kwargs,
            # 1 - old, 2 - new flammability calculation
            flammability_method=2,
        )[0],
    }

    inferno_kwargs["fuel_build_up"] = np.repeat(
        obs_fuel_build_up_1d[ti][np.newaxis], repeats=13, axis=0
    )
    inferno_kwargs["fapar_diag_pft"] = np.repeat(
        obs_fapar_1d[ti][np.newaxis], repeats=13, axis=0
    )

    python_ba_gb["new_obs_fapar"] = inferno_io(
        **inferno_kwargs,
        # 1 - old, 2 - new flammability calculation
        flammability_method=2,
    )[0]

    # Transform the 1D data to cubes.
    python_ba_gb = {
        key: get_1d_data_cube(data, lats=jules_lats, lons=jules_lons)
        for key, data in python_ba_gb.items()
    }

    return python_ba_gb


if __name__ == "__main__":
    # Load data.
    cubes = iris.load(
        str(
            Path(
                # "~/tmp/new-with-antec3/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUP0.Monthly.2000.nc"
                # "~/tmp/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUP0.Monthly.2009.nc"
                "~/tmp/new-with-antec4/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUP0.Instant.2001.nc"
            ).expanduser()
        )
    )

    # Extract the necessary variables.
    t1p5m_tile = cubes.extract_cube("t1p5m")
    q1p5m_tile = cubes.extract_cube("q1p5m")
    pstar = cubes.extract_cube("pstar")
    sthu_soilt = cubes.extract_cube("sthu")
    frac = cubes.extract_cube("landCoverFrac")
    c_soil_dpm_gb = cubes.extract_cube("c_dpm_gb")
    c_soil_rpm_gb = cubes.extract_cube("c_rpm_gb")
    canht = cubes.extract_cube("canht")
    ls_rain = cubes.extract_cube("ls_rain")
    con_rain = cubes.extract_cube("con_rain")
    fuel_build_up = cubes.extract_cube("fuel_build")
    fapar_diag_pft = cubes.extract_cube("fapar")

    jules_fapar = (
        darray.sum(fapar_diag_pft.core_data() * frac.core_data()[:, :13], axis=1)
        / darray.sum(frac.core_data()[:, :13], axis=1)
    )[:, 0]

    jules_lat_coord = pstar.coord("latitude")
    jules_lats = pstar.coord("latitude")
    jules_lon_coord = pstar.coord("longitude")
    jules_lons = pstar.coord("longitude")

    data_time_coord = fapar_diag_pft.coord("time")

    # Load observed monthly BA.
    gfed = GFEDv4()
    gfed.limit_months(data_time_coord.cell(0).point, data_time_coord.cell(-1).point)
    gfed.regrid(new_latitudes=n96e_lats, new_longitudes=n96e_lons, area_weighted=True)
    gfed_ba = gfed.cube

    # Load population density (`pop_den`).
    # hyde_popd = Datasets(Ext_HYDE()).select_variables("popd").dataset
    # clim_hyde_popd = hyde_popd.get_climatology_dataset(
    #     hyde_popd.min_time, hyde_popd.max_time
    # )
    # pop_den = clim_hyde_popd.cube

    # Load lightning data (`flash_rate`).
    # wwlln = WWLLN()
    # clim_wwlln = wwlln.get_climatology_dataset(wwlln.min_time, wwlln.max_time)
    # flash_rate = clim_wwlln.cube

    # Load observed monthly FAPAR.
    fapar = Ext_MOD15A2H_fPAR()
    fapar.limit_months(data_time_coord.cell(0).point, data_time_coord.cell(-1).point)
    fapar.regrid(new_latitudes=n96e_lats, new_longitudes=n96e_lons, area_weighted=True)
    mon_obs_fapar = fapar.cube

    indices_1d_to_2d = get_1d_to_2d_indices(
        pstar.coord("latitude").points[0],
        convert_longitudes(pstar.coord("longitude").points[0]),
        mon_obs_fapar.coord("latitude").points,
        mon_obs_fapar.coord("longitude").points,
    )

    mon_obs_fapar_1d = np.ma.vstack(
        [data[indices_1d_to_2d][np.newaxis] for data in mon_obs_fapar.data]
    )
    # Convert from monthly to timestep-aligned values.
    # NOTE that this discards the mask!
    obs_fapar_1d = temporal_nearest_neighbour_interp(
        mon_obs_fapar_1d.data,
        int(np.ceil(jules_fapar.shape[0] / mon_obs_fapar_1d.shape[0])),
    )[: jules_fapar.shape[0]]

    mask_obs_fapar_1d = temporal_nearest_neighbour_interp(
        mon_obs_fapar_1d.mask,
        int(np.ceil(jules_fapar.shape[0] / mon_obs_fapar_1d.shape[0])),
    )[: jules_fapar.shape[0]]

    # Calculate the antecedent fuel build-up metric.
    # This uses the fact that we are using data that is exported by the model every 4
    # timesteps.
    # Repeat the averaging procedure in order to reach convergence for a more
    # realistic depiction of the averaged parameter.
    obs_fuel_build_up_1d = repeated_exponential_average(
        temporal_nearest_neighbour_interp(obs_fapar_1d, 4),
        4.6283007e-04,
        repetitions=10,
    )[::4]

    # Define the ignition method (`ignition_method`).
    ignition_method = 1

    ti = 0

    if 0:
        # Optimisation.
        fapar_factors = np.linspace(-1, -2, 10)
        fuel_build_up_factors = np.linspace(1, 2, 10)
        results = np.zeros(
            (fapar_factors.shape[0], fuel_build_up_factors.shape[0]), dtype=np.float64
        )
        for i, fapar_factor in enumerate(tqdm(fapar_factors)):
            for j, fuel_build_up_factor in enumerate(fuel_build_up_factors):
                python_ba_gb = run_inferno(
                    t1p5m_tile=t1p5m_tile[ti, :, 0].data,
                    q1p5m_tile=q1p5m_tile[ti, :, 0].data,
                    pstar=pstar[ti, 0].data,
                    sthu_soilt=sthu_soilt[ti].data,
                    frac=frac[ti, :, 0].data,
                    c_soil_dpm_gb=c_soil_dpm_gb[ti, 0].data,
                    c_soil_rpm_gb=c_soil_rpm_gb[ti, 0].data,
                    canht=canht[ti, :, 0].data,
                    ls_rain=ls_rain[ti, 0].data,
                    con_rain=con_rain[ti, 0].data,
                    # Not used for ignition mode 1.
                    pop_den=np.zeros((land_pts,)) - 1,
                    flash_rate=np.zeros((land_pts,)) - 1,
                    ignition_method=ignition_method,
                    fuel_build_up=fuel_build_up[ti, :, 0].data,
                    fapar_diag_pft=fapar_diag_pft[ti, :, 0].data,
                    fapar_factor=fapar_factor,
                    fapar_centre=0.4,
                    fuel_build_up_factor=fuel_build_up_factor,
                    fuel_build_up_centre=0.4,
                    temperature_factor=0.03,
                    temperature_centre=300,
                    jules_lats=jules_lats,
                    jules_lons=jules_lons,
                    obs_fapar_1d=obs_fapar_1d,
                    obs_fuel_build_up_1d=obs_fuel_build_up_1d,
                )

                if len(np.unique(python_ba_gb["new"].data)) == 1:
                    results[i, j] += 0.25
                if len(np.unique(python_ba_gb["new_obs_fapar"].data)) == 1:
                    # Give these results more weighting if they are absent (i.e. all 0s).
                    results[i, j] += 0.75

        plt.figure()
        plt.pcolormesh(fapar_factors, fuel_build_up_factors, results.T)
        plt.xlabel("FAPAR")
        plt.ylabel("fuel build up")
        plt.colorbar()

    if 1:
        # Comparison plotting.
        python_ba_gb = run_inferno(
            t1p5m_tile=t1p5m_tile[ti, :, 0].data,
            q1p5m_tile=q1p5m_tile[ti, :, 0].data,
            pstar=pstar[ti, 0].data,
            sthu_soilt=sthu_soilt[ti].data,
            frac=frac[ti, :, 0].data,
            c_soil_dpm_gb=c_soil_dpm_gb[ti, 0].data,
            c_soil_rpm_gb=c_soil_rpm_gb[ti, 0].data,
            canht=canht[ti, :, 0].data,
            ls_rain=ls_rain[ti, 0].data,
            con_rain=con_rain[ti, 0].data,
            # Not used for ignition mode 1.
            pop_den=np.zeros((land_pts,)) - 1,
            flash_rate=np.zeros((land_pts,)) - 1,
            ignition_method=ignition_method,
            fuel_build_up=fuel_build_up[ti, :, 0].data,
            fapar_diag_pft=fapar_diag_pft[ti, :, 0].data,
            fapar_factor=-2,
            # fapar_factor=-1.0,
            fapar_centre=0.4,
            fuel_build_up_factor=1.2,
            # fuel_build_up_factor=1.0,
            fuel_build_up_centre=0.4,
            temperature_factor=0.03,
            temperature_centre=300,
            jules_lats=jules_lats,
            jules_lons=jules_lons,
            obs_fapar_1d=obs_fapar_1d,
            obs_fuel_build_up_1d=obs_fuel_build_up_1d,
        )
        plot_comparison(
            jules_ba_gb=cubes.extract_cube("burnt_area_gb")[ti, 0],
            python_ba_gb=python_ba_gb,
            mask_cube=get_1d_data_cube(
                mask_obs_fapar_1d[ti], lats=jules_lats, lons=jules_lons
            ),
        )
        plt.show()
