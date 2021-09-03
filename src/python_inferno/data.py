# -*- coding: utf-8 -*-
from pathlib import Path

import iris
import numpy as np
from jules_output_analysis.data import get_1d_to_2d_indices, n96e_lats, n96e_lons
from jules_output_analysis.utils import convert_longitudes
from tqdm import tqdm
from wildfires.data import Ext_MOD15A2H_fPAR, GFEDv4, homogenise_time_coordinate

from .cache import cache, mark_dependency
from .precip_dry_day import calculate_inferno_dry_days
from .utils import (
    PartialDateTime,
    make_contiguous,
    monthly_average_data,
    temporal_nearest_neighbour_interp,
    temporal_processing,
    unpack_wrapped,
)

timestep = 4 * 60 * 60


@cache(dependencies=[make_contiguous, temporal_nearest_neighbour_interp])
@mark_dependency
def load_data(
    filenames=(
        tuple(
            [
                str(Path(s).expanduser())
                for s in [
                    "~/tmp/climatology5_c.nc",
                ]
            ]
        )
    ),
    N=None,
    output_timesteps=4,
    climatology_dates=(PartialDateTime(2000, 1), PartialDateTime(2016, 12)),
):
    assert output_timesteps == 4

    raw_cubes = iris.cube.CubeList([])
    for f in tqdm(list(map(str, filenames)), desc="Initial file loading"):
        raw_cubes.extend(iris.load_raw(f))

    # Ensure cubes can be concatenated.
    cubes = homogenise_time_coordinate(
        iris.cube.CubeList(
            [cube for cube in raw_cubes if cube.name() not in ("latitude", "longitude")]
        )
    ).concatenate()
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

    if climatology_dates is not None:
        # Verify we are dealing with 1 year of data.
        assert obs_dates[0].year == obs_dates[1].year
        assert obs_dates[0].month == 1
        assert obs_dates[1].month == 12

        obs_dates = climatology_dates

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
        # If the JULES data is climatological, compute the climatology here too
        if climatology_dates is not None:
            dataset = dataset.get_climatology_dataset(
                dataset.min_time, dataset.max_time
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
        # Make magnitudes more similar to e.g. FAPAR, i.e. ~[0, 1].
        make_contiguous(npp_pft[:N, :, 0].data.data) / 1e-7,
        make_contiguous(npp_gb[:N, 0].data.data) / 1e-7,
        # Whether or not the data is climatological.
        climatology_dates is not None,
    )


@cache(dependencies=[load_data, temporal_processing, monthly_average_data])
def get_processed_climatological_data(n_samples_pft, average_samples):
    (
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
        fuel_build_up,
        fapar_diag_pft,
        jules_lats,
        jules_lons,
        gfed_ba_1d,
        obs_fapar_1d,
        jules_ba_gb,
        jules_time_coord,
        npp_pft,
        npp_gb,
        climatology_output,
    ) = load_data(
        filenames=(
            tuple(
                [
                    str(Path(s).expanduser())
                    for s in [
                        "~/tmp/climatology5_c.nc",
                    ]
                ]
            )
        ),
        N=None,
        output_timesteps=4,
        climatology_dates=(PartialDateTime(2000, 1), PartialDateTime(2016, 12)),
    )

    data_dict = dict(
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
        # NOTE NPP is used here now, NOT FAPAR!
        fuel_build_up=npp_pft,
        fapar_diag_pft=npp_pft,
        # TODO: How is dry-day calculation affected by climatological input data?
        dry_days=unpack_wrapped(calculate_inferno_dry_days)(
            ls_rain, con_rain, threshold=1.0, timestep=timestep
        ),
        # NOTE The target BA is only included here to ease processing. It will be
        # removed prior to the modelling function.
        gfed_ba_1d=gfed_ba_1d,
    )

    data_dict, jules_time_coord = temporal_processing(
        data_dict=data_dict,
        antecedent_shifts_dict={"fuel_build_up": n_samples_pft},
        average_samples=average_samples,
        aggregator={
            name: {"dry_days": "MAX", "t1p5m_tile": "MAX"}.get(name, "MEAN")
            for name in data_dict
        },
        time_coord=jules_time_coord,
        climatology_input=climatology_output,
    )

    assert jules_time_coord.cell(-1).point.month == 12
    last_year = jules_time_coord.cell(-1).point.year
    for start_i in range(jules_time_coord.shape[0]):
        if jules_time_coord.cell(start_i).point.year == last_year:
            break
    else:
        raise ValueError("Target year not encountered.")

    # Trim the data and temporal coord such that the data spans a single year.
    jules_time_coord = jules_time_coord[start_i:]
    for data_name in data_dict:
        data_dict[data_name] = data_dict[data_name][start_i:]

    # Remove the target BA.
    gfed_ba_1d = data_dict.pop("gfed_ba_1d")

    # NOTE The mask array on `gfed_ba_1d` determines which samples are selected for
    # comparison later on.

    # Calculate monthly averages.
    mon_avg_gfed_ba_1d = monthly_average_data(gfed_ba_1d, time_coord=jules_time_coord)

    # Ensure the data spans a single year.
    assert mon_avg_gfed_ba_1d.shape[0] == 12
    assert (
        jules_time_coord.cell(0).point.year == jules_time_coord.cell(-1).point.year
        and jules_time_coord.cell(0).point.month == 1
        and jules_time_coord.cell(-1).point.month == 12
        and jules_time_coord.shape[0] >= 12
    )
    return data_dict, mon_avg_gfed_ba_1d, jules_time_coord
