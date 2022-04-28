# -*- coding: utf-8 -*-
from datetime import datetime
from functools import partial
from pathlib import Path

import iris
import numpy as np
from dateutil.relativedelta import relativedelta
from jules_output_analysis.data import (
    cube_1d_to_2d,
    get_1d_data_cube,
    get_1d_to_2d_indices,
    n96e_lats,
    n96e_lons,
)
from jules_output_analysis.utils import convert_longitudes
from loguru import logger
from tqdm import tqdm
from wildfires.data import (
    Datasets,
    Ext_ESA_CCI_Landcover_PFT,
    Ext_MOD15A2H_fPAR,
    GFEDv4,
    homogenise_time_coordinate,
    regions_GFED,
    regrid,
)

from python_inferno.pnv import get_pnv_mega_regions

from .cache import cache, mark_dependency
from .configuration import N_pft_groups, land_pts, npft
from .dry_bal import calculate_grouped_dry_bal
from .litter_pool import calculate_litter
from .precip_dry_day import calculate_inferno_dry_days, precip_moving_sum
from .utils import (
    PartialDateTime,
    key_cache,
    make_contiguous,
    memoize,
    monthly_average_data,
    temporal_nearest_neighbour_interp,
    temporal_processing,
    unpack_wrapped,
)
from .vpd import calculate_grouped_vpd

# Time step in seconds (for the 'Instant' output profile, which outputs every 4 hours).
timestep = 4 * 60 * 60


@memoize
@cache
@mark_dependency
def load_single_year_cubes(*, filename, variable_name_slices, temporal_subsampling=1):
    logger.info(f"Loading '{', '.join(variable_name_slices)}' from {filename}.")

    cubes = iris.load_raw(filename)

    # Load variables.
    data_dict = {name: cubes.extract_cube(name) for name in variable_name_slices}

    # Ensure cubes have the same temporal dimensions.
    assert len(set(cube.shape[0] for cube in data_dict.values())) == 1

    jules_time_coord = next(iter(data_dict.values())).coord("time")
    if (
        jules_time_coord.cell(-1).point.year != jules_time_coord.cell(-2).point.year
        and jules_time_coord.cell(-1).point.month == 1
    ):
        # Ensure the last sample is not taken into account if it is the first day of a
        # new year.
        N = jules_time_coord.shape[0] - 1
    else:
        N = jules_time_coord.shape[0]

    jules_time_coord = jules_time_coord[:N]

    assert (
        jules_time_coord.cell(-1).point.year == jules_time_coord.cell(0).point.year
    ), "File should span one year only."

    assert jules_time_coord.cell(0).point.month == 1, "File should start in Jan."

    assert jules_time_coord.cell(-1).point.month == 12, "Data should end in Dec."

    # Extract the actual data.
    def modify_slices(s):
        assert isinstance(s, tuple)
        assert s[0] == slice(None)
        s = list(s)
        s[0] = slice(0, N, temporal_subsampling)
        return tuple(s)

    return {
        name: np.asarray(
            make_contiguous(cube[modify_slices(variable_name_slices[name])].data.data),
            dtype=np.float32,
        )
        for name, cube in data_dict.items()
    }


def load_obs_data(dataset, obs_dates=None, climatology=False, Nt=None):
    jules_lats, jules_lons = load_jules_lats_lons()

    # Load observed monthly data.

    if obs_dates is not None:
        dataset.limit_months(*obs_dates)

    dataset.regrid(
        new_latitudes=n96e_lats, new_longitudes=n96e_lons, area_weighted=True
    )
    # If the JULES data is climatological, compute the climatology here too
    if climatology:
        dataset = dataset.get_climatology_dataset(dataset.min_time, dataset.max_time)

    indices_1d_to_2d = get_1d_to_2d_indices(
        np.ma.getdata(jules_lats.points[0]),
        convert_longitudes(np.ma.getdata(jules_lons.points[0])),
        n96e_lats,
        n96e_lons,
    )

    mon_data_1d = np.ma.vstack(
        [data[indices_1d_to_2d][np.newaxis] for data in dataset.cube.data]
    )
    if Nt is not None:
        initial_pad_samples = 0
        final_pad_samples = 0

        # Convert from monthly to timestep-aligned values.
        if obs_dates is None or climatology:
            N_interp = Nt
        elif obs_dates is not None:
            # Compensate for missing data if needed.
            # Nt corresponds to obs_dates.
            total_seconds = (obs_dates[1] - obs_dates[0]).total_seconds()

            min_time = (
                dataset.min_time
                if dataset.frequency != "monthly"
                # Lowest datetime in the 'max_time' month.
                else datetime(dataset.min_time.year, dataset.min_time.month, 1)
            )

            max_time = (
                dataset.max_time
                if dataset.frequency != "monthly"
                # Highest datetime in the 'max_time' month.
                else datetime(dataset.max_time.year, dataset.max_time.month, 1)
                + relativedelta(months=+1, microseconds=-1)
            )

            initial_pad_seconds = max((min_time - obs_dates[0]).total_seconds(), 0)
            final_pad_seconds = max((obs_dates[1] - max_time).total_seconds(), 0)

            initial_pad_samples = int(
                np.floor((initial_pad_seconds / total_seconds) * Nt)
            )
            final_pad_samples = int(np.floor((final_pad_seconds / total_seconds) * Nt))

            total_pad_samples = initial_pad_samples + final_pad_samples

            N_interp = Nt - total_pad_samples
            if total_pad_samples > 0:
                logger.warning(f"Padding for {dataset}.")
                logger.warning(f"{min_time}, {max_time}.")
                logger.warning(
                    f"Initial samples: {initial_pad_samples} ({initial_pad_seconds} s)."
                )
                logger.warning(
                    f"Final samples: {final_pad_samples} ({final_pad_seconds} s)."
                )
        else:
            raise RuntimeError()

        data_1d = temporal_nearest_neighbour_interp(
            mon_data_1d, int(np.ceil(N_interp / mon_data_1d.shape[0])), "start"
        )[:N_interp]

        if initial_pad_samples > 0:
            data_1d = np.ma.vstack(
                (
                    np.ma.MaskedArray(
                        np.zeros((initial_pad_samples, *data_1d.shape[1:])), mask=True
                    ),
                    data_1d,
                )
            )
        if final_pad_samples > 0:
            data_1d = np.ma.vstack(
                (
                    data_1d,
                    np.ma.MaskedArray(
                        np.zeros((final_pad_samples, *data_1d.shape[1:])), mask=True
                    ),
                )
            )
        return data_1d
    return mon_data_1d


@cache(dependencies=[make_contiguous, temporal_nearest_neighbour_interp])
@mark_dependency
def load_data(
    filenames=(
        tuple(
            [
                str(Path(s).expanduser())
                for s in [
                    "~/tmp/climatology6.nc",
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

    # Load observed data.
    mod_load_obs_data = partial(
        load_obs_data,
        obs_dates=obs_dates,
        climatology=climatology_dates is not None,
        Nt=jules_time_coord.shape[0],
    )
    gfed_ba_1d = mod_load_obs_data(GFEDv4())
    obs_fapar_1d = mod_load_obs_data(Ext_MOD15A2H_fPAR())
    obs_pftcrop_1d = mod_load_obs_data(
        Datasets(Ext_ESA_CCI_Landcover_PFT()).select_variables("pftCrop").dataset
    )

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
        make_contiguous(obs_pftcrop_1d[:N]),
        jules_time_coord[:N],
        # Make magnitudes more similar to e.g. FAPAR, i.e. ~[0, 1].
        make_contiguous(npp_pft[:N, :, 0].data.data) / 1e-7,
        make_contiguous(npp_gb[:N, 0].data.data) / 1e-7,
        # Whether or not the data is climatological.
        climatology_dates is not None,
    )


@cache(dependencies=[load_single_year_cubes, calculate_inferno_dry_days])
@mark_dependency
def get_climatological_dry_days(
    filenames=tuple(
        str(Path(s).expanduser())
        for s in (
            "~/tmp/new-with-antec6/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUPD0.Instant.2010.nc",
            "~/tmp/new-with-antec6/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUPD0.Instant.2011.nc",
        )
    ),
    threshold=1.0,
):
    # Load instantaneous values from the files below, then calculate dry days, then
    # perform climatological averaging

    clim_dry_days = None
    n_avg = 0

    for f in tqdm(list(map(str, filenames)), desc="Processing dry-days"):
        data_dict = load_single_year_cubes(
            filename=f,
            variable_name_slices={
                "ls_rain": (slice(None), 0),
                "con_rain": (slice(None), 0),
            },
        )

        # Calculate dry days.
        dry_days = unpack_wrapped(calculate_inferno_dry_days)(
            ls_rain=data_dict["ls_rain"],
            con_rain=data_dict["con_rain"],
            threshold=threshold,
            timestep=timestep,
        )
        if clim_dry_days is None:
            clim_dry_days = dry_days
            assert n_avg == 0
        else:
            clim_dry_days += dry_days
        n_avg += 1

    return clim_dry_days / n_avg


def handle_param(param, N_params=N_pft_groups):
    param = np.asarray(param)

    if param.shape != (N_params,):
        assert param.shape in ((), (1,))
        param = np.asarray([param.ravel()[0]] * N_params)

    return param


key_cached_calculate_grouped_vpd = key_cache(calculate_grouped_vpd)
key_cached_precip_moving_sum = key_cache(precip_moving_sum)


@cache(
    dependencies=[
        load_single_year_cubes,
        calculate_grouped_vpd,
        calculate_grouped_dry_bal,
        precip_moving_sum,
    ]
)
@mark_dependency
def get_climatological_grouped_dry_bal(
    *,
    filenames=tuple(
        str(Path(s).expanduser())
        for s in (
            "~/tmp/new-with-antec6/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUPD0.Instant.2010.nc",
            "~/tmp/new-with-antec6/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUPD0.Instant.2011.nc",
        )
    ),
    rain_f,
    vpd_f,
    verbose=True,
):
    """Load instantaneous values from the files below, then calculate dry_bal, then
    perform climatological averaging."""

    rain_f = handle_param(rain_f, N_params=N_pft_groups)
    vpd_f = handle_param(vpd_f, N_params=N_pft_groups)

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
                (data_dict["pstar"].shape[0], N_pft_groups, land_pts), dtype=np.float64
            )

        # Calculate grouped dry_bal.
        grouped_dry_bal = calculate_grouped_dry_bal(
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


@memoize
@cache(
    dependencies=[
        load_data,
        temporal_processing,
        monthly_average_data,
        get_climatological_grouped_dry_bal,
        get_climatological_dry_days,
    ]
)
def get_processed_climatological_data(
    *,
    litter_tc,
    leaf_f,
    n_samples_pft,
    average_samples,
    rain_f=None,
    vpd_f=None,
):
    logger.debug("start data")
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
        obs_pftcrop_1d,
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
                        "~/tmp/climatology6.nc",
                    ]
                ]
            )
        ),
        N=None,
        output_timesteps=4,
        climatology_dates=(PartialDateTime(2000, 1), PartialDateTime(2016, 12)),
    )
    logger.debug("Got data")

    dry_bal_func = (
        get_climatological_grouped_dry_bal
        if (rain_f is None and vpd_f is None)
        else get_climatological_grouped_dry_bal._orig_func
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
        fuel_build_up=npp_pft,  # Note this is shifted below.
        fapar_diag_pft=npp_pft,
        litter_pool=calc_litter_pool(
            litter_tc=litter_tc if litter_tc is not None else tuple([1e-9] * npft),
            leaf_f=leaf_f if leaf_f is not None else tuple([1e-3] * npft),
            Nt=None,
        ),
        dry_days=get_climatological_dry_days(),
        grouped_dry_bal=dry_bal_func(
            rain_f=rain_f if rain_f is not None else tuple([0.3] * N_pft_groups),
            vpd_f=vpd_f if vpd_f is not None else tuple([40] * N_pft_groups),
            verbose=False,
        ),
        # NOTE The target BA is only included here to ease processing. It will be
        # removed prior to the modelling function.
        gfed_ba_1d=gfed_ba_1d,
        obs_pftcrop_1d=obs_pftcrop_1d,
    )

    logger.debug("Populated data_dict")

    data_dict, jules_time_coord = temporal_processing(
        data_dict=data_dict,
        # NOTE 'fuel_build_up' refers to the (initially) unshifted productivity proxy
        antecedent_shifts_dict=dict(
            (key, val)
            for (key, val) in [("fuel_build_up", n_samples_pft)]
            if val is not None
        ),
        average_samples=average_samples,
        aggregator={
            name: {"dry_days": "MAX", "t1p5m_tile": "MAX"}.get(name, "MEAN")
            for name in data_dict
        },
        time_coord=jules_time_coord,
        climatology_input=climatology_output,
    )

    logger.debug("Finished temporal processing.")

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


@memoize
@cache
def calc_litter_pool(
    *,
    filename=str(
        Path(
            "~/tmp/new-with-antec6/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUPD0.Instant.2010.nc"
        ).expanduser()
    ),
    litter_tc,
    leaf_f,
    verbose=True,
    Nt=400,
    spinup_relative_delta=1e-2,
    max_spinup_cycles=100,
):
    litter_tc = handle_param(litter_tc, N_params=npft)
    leaf_f = handle_param(leaf_f, N_params=npft)

    litter_pool = None

    data_dict = load_single_year_cubes(
        filename=filename,
        variable_name_slices={
            "leaf_litC": (slice(None), slice(None), 0),
            "t1p5m": (slice(None), slice(npft), 0),
            "sthu": (slice(None), 0, 0),
        },
    )

    if Nt is None:
        Nt = data_dict["leaf_litC"].shape[0]

    if litter_pool is None:
        litter_pool = np.zeros((Nt, npft, land_pts), dtype=np.float64)

    # Calculate the litter pool.
    spinup_cycles = 1
    prev_pool = litter_pool[0].copy()
    while spinup_cycles <= max_spinup_cycles:
        calculate_litter(
            leaf_litC=data_dict["leaf_litC"][:Nt],
            T=data_dict["t1p5m"][:Nt],
            sm=data_dict["sthu"][:Nt],
            dt=timestep,
            litter_tc=litter_tc,
            leaf_f=leaf_f,
            init=litter_pool[-1],
            out=litter_pool,
        )
        # Ignore zero values.
        sel = ~np.isclose(litter_pool[0], 0)
        if not np.any(sel):
            # No non-zero values, use dummy value.
            max_delta = 1e10
        else:
            max_delta = np.max(
                np.abs(litter_pool[0][sel] - prev_pool[sel]) / litter_pool[0][sel]
            )
        logger.debug(f"Cycle {spinup_cycles} Delta | max:{max_delta:0.1e}")

        if max_delta < spinup_relative_delta:
            # Maximum convergence delta has been reached.
            break

        prev_pool = litter_pool[0].copy()
        spinup_cycles += 1
    else:
        raise RuntimeError(
            f"Spinup did not converge within {max_spinup_cycles} cycles."
        )
    return litter_pool


@memoize
@cache
def load_jules_lats_lons(filename=str(Path("~/tmp/climatology6.nc").expanduser())):
    cubes = iris.load_raw(filename)
    pstar = cubes.extract_cube("pstar")
    jules_lats = pstar.coord("latitude")
    jules_lons = pstar.coord("longitude")
    return jules_lats, jules_lons


@memoize
@cache
def get_gfed_regions():
    gfed_regions = regrid(
        regions_GFED(),
        new_latitudes=n96e_lats,
        new_longitudes=n96e_lons,
        scheme=iris.analysis.Nearest(),
    )
    assert gfed_regions.attributes["regions"][0] == "Ocean"
    gfed_regions.data.mask |= np.ma.getdata(gfed_regions.data) == 0
    return (
        gfed_regions,
        len(gfed_regions.attributes["regions"]) - 1,
    )  # Ignore the Ocean region


def get_pnv_mega_plot_data():
    pnv_mega_regions = get_pnv_mega_regions()
    return pnv_mega_regions, len(pnv_mega_regions.attributes["regions"])


@cache
def get_2d_cubes(*, data_dict):
    jules_lats, jules_lons = load_jules_lats_lons()
    return {
        name: cube_1d_to_2d(get_1d_data_cube(data, lats=jules_lats, lons=jules_lons))
        for name, data in data_dict.items()
    }


@cache
def subset_sthu_soilt(data_dict):
    data_dict["sthu_soilt"] = data_dict["sthu_soilt"][:, 0, 0]
    return data_dict
