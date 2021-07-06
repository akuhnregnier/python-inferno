# -*- coding: utf-8 -*-
from pathlib import Path

import dask.array as darray
import iris
import numpy as np
from iris.time import PartialDateTime
from jules_output_analysis.data import get_1d_to_2d_indices, n96e_lats, n96e_lons
from jules_output_analysis.utils import convert_longitudes
from wildfires.data import Ext_MOD15A2H_fPAR, GFEDv4

from .cache import cache, mark_dependency
from .utils import (
    exponential_average,
    make_contiguous,
    temporal_nearest_neighbour_interp,
)


@cache(dependencies=[make_contiguous])
@mark_dependency
def load_data(
    filename=str(
        Path(
            # "~/tmp/new-with-antec3/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUP0.Monthly.2000.nc"
            # "~/tmp/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUP0.Monthly.2009.nc"
            # "~/tmp/new-with-antec4/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUP0.Instant.2001.nc"
            "~/tmp/climatology5_c.nc"
        ).expanduser()
    ),
    N=None,
    obs_dates=(PartialDateTime(2000, 1), PartialDateTime(2016, 12)),
):
    cubes = iris.load(filename)

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

    # JULES BA.
    jules_ba_gb = cubes.extract_cube("burnt_area_gb")

    jules_fapar = (
        darray.sum(fapar_diag_pft.core_data() * frac.core_data()[:, :13], axis=1)
        / darray.sum(frac.core_data()[:, :13], axis=1)
    )[:, 0]

    jules_lats = pstar.coord("latitude")
    jules_lons = pstar.coord("longitude")

    data_time_coord = fapar_diag_pft.coord("time")

    indices_1d_to_2d = get_1d_to_2d_indices(
        jules_lats.points[0],
        convert_longitudes(jules_lons.points[0]),
        n96e_lats,
        n96e_lons,
    )

    if obs_dates is None:
        obs_dates = (data_time_coord.cell(0).point, data_time_coord.cell(-1).point)

    # Load observed monthly BA.
    gfed = GFEDv4()
    gfed.limit_months(*obs_dates)
    gfed.regrid(new_latitudes=n96e_lats, new_longitudes=n96e_lons, area_weighted=True)
    gfed_ba = gfed.cube

    mon_gfed_ba_1d = np.ma.vstack(
        [data[indices_1d_to_2d][np.newaxis] for data in gfed_ba.data]
    )
    # Convert from monthly to timestep-aligned values.
    gfed_ba_1d = temporal_nearest_neighbour_interp(
        mon_gfed_ba_1d,
        int(np.ceil(jules_fapar.shape[0] / mon_gfed_ba_1d.shape[0])),
    )[: jules_fapar.shape[0]]

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
    fapar.limit_months(*obs_dates)
    fapar.regrid(new_latitudes=n96e_lats, new_longitudes=n96e_lons, area_weighted=True)
    mon_obs_fapar = fapar.cube

    mon_obs_fapar_1d = np.ma.vstack(
        [data[indices_1d_to_2d][np.newaxis] for data in mon_obs_fapar.data]
    )
    # Convert from monthly to timestep-aligned values.
    obs_fapar_1d = temporal_nearest_neighbour_interp(
        mon_obs_fapar_1d,
        int(np.ceil(jules_fapar.shape[0] / mon_obs_fapar_1d.shape[0])),
    )[: jules_fapar.shape[0]]

    # Calculate the antecedent fuel build-up metric.
    # This uses the fact that we are using data that is exported by the model every 4
    # timesteps.
    # Repeat the averaging procedure in order to reach convergence for a more
    # realistic depiction of the averaged parameter.
    obs_fuel_build_up_1d = exponential_average(
        temporal_nearest_neighbour_interp(obs_fapar_1d, 4),
        4.6283007e-04,
        repetitions=10,
    )[::4]

    # Pre-load data.
    if N is None:
        N = frac.shape[0]
    else:
        assert N <= frac.shape[0]

    iris.cube.CubeList(
        [
            t1p5m_tile[:N, :, 0],
            q1p5m_tile[:N, :, 0],
            pstar[:N, 0],
            sthu_soilt[:N],
            frac[:N, :, 0],
            c_soil_dpm_gb[:N, 0],
            c_soil_rpm_gb[:N, 0],
            canht[:N, :, 0],
            ls_rain[:N, 0],
            con_rain[:N, 0],
            fuel_build_up[:N, :, 0],
            fapar_diag_pft[:N, :, 0],
            jules_ba_gb[:N, 0],
        ]
    ).realise_data()

    jules_time_coord = ls_rain.coord("time")[:N]

    return (
        # For most variables, return the arrays without any masks (which should just
        # be `False` for JULES outputs).
        make_contiguous(t1p5m_tile[:N, :, 0].data.data),
        make_contiguous(q1p5m_tile[:N, :, 0].data.data),
        make_contiguous(pstar[:N, 0].data.data),
        make_contiguous(sthu_soilt[:N].data.data),
        make_contiguous(frac[:N, :, 0].data.data),
        make_contiguous(c_soil_dpm_gb[:N, 0].data.data),
        make_contiguous(c_soil_rpm_gb[:N, 0].data.data),
        make_contiguous(canht[:N, :, 0].data.data),
        make_contiguous(ls_rain[:N, 0].data.data),
        make_contiguous(con_rain[:N, 0].data.data),
        make_contiguous(fuel_build_up[:N, :, 0].data.data),
        make_contiguous(fapar_diag_pft[:N, :, 0].data.data),
        jules_lats,
        jules_lons,
        make_contiguous(gfed_ba_1d[:N]),
        make_contiguous(obs_fapar_1d[:N]),
        make_contiguous(obs_fuel_build_up_1d[:N]),
        make_contiguous(jules_ba_gb[:N, 0]),
        jules_time_coord,
    )
