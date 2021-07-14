# -*- coding: utf-8 -*-
from pathlib import Path

import iris
import numpy as np
from jules_output_analysis.data import get_1d_to_2d_indices, n96e_lats, n96e_lons
from jules_output_analysis.utils import convert_longitudes
from tqdm import tqdm
from wildfires.data import Ext_MOD15A2H_fPAR, GFEDv4, homogenise_time_coordinate

from .cache import cache, mark_dependency
from .utils import make_contiguous, temporal_nearest_neighbour_interp


@cache(dependencies=[make_contiguous, temporal_nearest_neighbour_interp])
@mark_dependency
def load_data(
    filenames=(
        [
            str(Path(s).expanduser())
            for s in (
                "~/tmp/new-with-antec5/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUPD0.Instant.2010.nc",
                "~/tmp/new-with-antec5/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUPD0.Instant.2011.nc",
            )
        ]
    ),
    N=None,
    output_timesteps=4,
):
    assert output_timesteps == 4

    raw_cubes = iris.cube.CubeList([])
    for f in tqdm(list(map(str, filenames)), desc="Initial file loading"):
        raw_cubes.extend(iris.load_raw(f))

    # Ensure cubes can be concatenated.
    cubes = homogenise_time_coordinate(raw_cubes).concatenate()
    # Ensure all cubes have the same number of temporal samples after concatenation.
    assert len(set(cube.shape[0] for cube in cubes)) == 1

    # Load variables.
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
    jules_ba_gb = cubes.extract_cube("burnt_area_gb")
    npp_pft = cubes.extract_cube("npp_pft")
    npp_gb = cubes.extract_cube("npp")

    jules_lats = pstar.coord("latitude")
    jules_lons = pstar.coord("longitude")

    jules_time_coord = fapar_diag_pft.coord("time")
    if (
        jules_time_coord.cell(-1).point.year != jules_time_coord.cell(-2).point.year
        and jules_time_coord.cell(-1).point.month == 1
        and (N is None or N == jules_time_coord.shape[0])
    ):
        # Ensure the last sample is not taken into account if it is the first day of a
        # new year.
        N = jules_time_coord.shape[0] - 1
        obs_dates = (jules_time_coord.cell(0).point, jules_time_coord.cell(-2).point)
    else:
        obs_dates = (jules_time_coord.cell(0).point, jules_time_coord.cell(-1).point)

    indices_1d_to_2d = get_1d_to_2d_indices(
        jules_lats.points[0],
        convert_longitudes(jules_lons.points[0]),
        n96e_lats,
        n96e_lons,
    )

    def load_obs_data(dataset):
        # Load observed monthly data.
        dataset.limit_months(*obs_dates)
        dataset.regrid(
            new_latitudes=n96e_lats, new_longitudes=n96e_lons, area_weighted=True
        )
        mon_data_1d = np.ma.vstack(
            [data[indices_1d_to_2d][np.newaxis] for data in dataset.cube.data]
        )
        # Convert from monthly to timestep-aligned values.
        data_1d = temporal_nearest_neighbour_interp(
            mon_data_1d,
            int(np.ceil(jules_time_coord.shape[0] / mon_data_1d.shape[0])),
            "start",
        )[: jules_time_coord.shape[0]]
        return data_1d

    # Load observed data.
    gfed_ba_1d = load_obs_data(GFEDv4())
    obs_fapar_1d = load_obs_data(Ext_MOD15A2H_fPAR())

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
        make_contiguous(jules_ba_gb[:N, 0]),
        jules_time_coord[:N],
        make_contiguous(npp_pft[:N, :, 0].data.data),
        make_contiguous(npp_gb[:N, 0].data.data),
    )
