# -*- coding: utf-8 -*-
from pathlib import Path

import iris
import numpy as np

from python_inferno import inferno_io
from python_inferno.configuration import land_pts

if __name__ == "__main__":
    # Load data.
    cubes = iris.load(
        str(
            Path(
                "~/tmp/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUP0.Monthly.2009.nc"
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

    # Define the ignition method (`ignition_method`).
    ignition_method = 1

    python_ba_gb, python_ba = inferno_io(
        t1p5m_tile=t1p5m_tile.data[0, :, 0],
        q1p5m_tile=q1p5m_tile.data[0, :, 0],
        pstar=pstar.data[0, 0],
        sthu_soilt=sthu_soilt.data[0],
        frac=frac.data[0, :, 0],
        c_soil_dpm_gb=c_soil_dpm_gb.data[0, 0],
        c_soil_rpm_gb=c_soil_rpm_gb.data[0, 0],
        canht=canht.data[0, :, 0],
        ls_rain=ls_rain.data[0, 0],
        con_rain=con_rain.data[0, 0],
        # Not used for ignition mode 1.
        pop_den=np.zeros((land_pts,)) - 1,
        flash_rate=np.zeros((land_pts,)) - 1,
        ignition_method=ignition_method,
        fuel_build_up=fuel_build_up.data[0, :, 0],
        fapar_diag_pft=fapar_diag_pft.data[0, :, 0],
    )

    # Compare to the values calculated within JULES.
    jules_ba_gb = cubes.extract_cube("burnt_area_gb").data
