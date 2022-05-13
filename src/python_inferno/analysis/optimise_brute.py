# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import r2_score

from ..cache import cache
from ..configuration import land_pts
from ..data import load_data, timestep
from ..multi_timestep_inferno import multi_timestep_inferno
from ..precip_dry_day import calculate_inferno_dry_days
from ..utils import combine_ma_masks, unpack_wrapped

# Cache mask combination.
combine_ma_masks = cache(combine_ma_masks)


@cache
def calc_param_r2(opt_params, N=None):
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
        jules_lats,
        jules_lons,
        gfed_ba_1d,
        obs_fapar_1d,
        obs_fuel_build_up_1d,
        jules_ba_gb,
        jules_time_coord,
    ) = load_data(N=N)

    combined_mask = combine_ma_masks(gfed_ba_1d, obs_fapar_1d, obs_fuel_build_up_1d)

    kwargs = dict(
        t1p5m_tile=t1p5m_tile,
        q1p5m_tile=q1p5m_tile,
        pstar=pstar,
        sthu_soilt_single=sthu_soilt_single,
        frac=frac,
        c_soil_dpm_gb=c_soil_dpm_gb,
        c_soil_rpm_gb=c_soil_rpm_gb,
        canht=canht,
        ls_rain=ls_rain,
        con_rain=con_rain,
        # Not used for ignition mode 1.
        pop_den=np.zeros((land_pts,)) - 1,
        flash_rate=np.zeros((land_pts,)) - 1,
        ignition_method=1,
        fuel_build_up=np.repeat(
            np.expand_dims(obs_fuel_build_up_1d.data, 1), repeats=13, axis=1
        ),
        fapar_diag_pft=np.repeat(
            np.expand_dims(obs_fapar_1d.data, 1), repeats=13, axis=1
        ),
        dry_days=unpack_wrapped(calculate_inferno_dry_days)(
            ls_rain, con_rain, threshold=4.3e-5, timestep=timestep
        ),
        flammability_method=2,
        dryness_method=2,
        fapar_factor=-4.83e1,
        fapar_centre=4.0e-1,
        fuel_build_up_factor=1.01e1,
        fuel_build_up_centre=3.76e-1,
        temperature_factor=8.01e-2,
        temperature_centre=2.82e2,
        dry_day_factor=2.0e-2,
        dry_day_centre=1.73e2,
        rain_f=1,
        vpd_f=5e5,
        dry_bal_factor=1,
        dry_bal_centre=0,
        timestep=timestep,
    )

    for name, value in opt_params.items():
        kwargs[name] = value

    model_ba = unpack_wrapped(multi_timestep_inferno)(**kwargs)

    if np.all(np.isclose(model_ba, 0, rtol=0, atol=1e-15)):
        r2 = -1.0
    else:
        # Compute R2 score after normalising each by their mean.
        y_true = gfed_ba_1d.data[~combined_mask]
        y_pred = model_ba[~combined_mask]

        y_true /= np.mean(y_true)
        y_pred /= np.mean(y_pred)

        r2 = r2_score(y_true=y_true, y_pred=y_pred)

    return r2
