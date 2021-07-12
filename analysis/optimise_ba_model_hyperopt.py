#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from functools import partial

import hyperopt
import numpy as np
from hyperopt import fmin, hp, tpe
from hyperopt.mongoexp import MongoTrials
from sklearn.metrics import r2_score

from python_inferno.configuration import land_pts
from python_inferno.data import load_data
from python_inferno.metrics import loghist, mpd, nme, nmse
from python_inferno.multi_timestep_inferno import multi_timestep_inferno
from python_inferno.precip_dry_day import calculate_inferno_dry_days
from python_inferno.utils import (
    calculate_factor,
    expand_pft_params,
    exponential_average,
    monthly_average_data,
    temporal_nearest_neighbour_interp,
    unpack_wrapped,
)

timestep = 4 * 60 * 60


def to_optimise(opt_kwargs):
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
        # Discard this pre-calculated version here and do our own processing.
        _,  # obs_fuel_build_up_1d,
        jules_ba_gb,
        jules_time_coord,
    ) = load_data(N=None)

    expanded_opt_tmp = defaultdict(list)
    for name, val in opt_kwargs.items():
        if name[-1] not in ("2", "3"):
            expanded_opt_tmp[name].append(val)
        elif name[-1] == "2":
            assert name[:-1] in expanded_opt_tmp
            assert len(expanded_opt_tmp[name[:-1]]) == 1
            expanded_opt_tmp[name[:-1]].append(val)
        elif name[-1] == "3":
            assert name[:-1] in expanded_opt_tmp
            assert len(expanded_opt_tmp[name[:-1]]) == 2
            expanded_opt_tmp[name[:-1]].append(val)

    expanded_opt_kwargs = dict()
    for name, vals in expanded_opt_tmp.items():
        if len(vals) == 3:
            expanded_opt_kwargs[name] = expand_pft_params(vals)

    print("Opt param arrays")
    for name, vals in expanded_opt_kwargs.items():
        print(name, vals)

    assert "fuel_build_up_alpha" in expanded_opt_kwargs
    alphas = expanded_opt_kwargs.pop("fuel_build_up_alpha")
    assert "fuel_build_up_alpha" not in expanded_opt_kwargs

    obs_fuel_build_up_1d = np.ma.stack(
        list(
            exponential_average(
                temporal_nearest_neighbour_interp(obs_fapar_1d.__wrapped__, 4),
                alpha,
                repetitions=10,
            )[::4]
            for alpha in alphas
        ),
        axis=1,
    )

    combined_mask = (
        gfed_ba_1d.mask | obs_fapar_1d.mask | np.any(obs_fuel_build_up_1d.mask, axis=1)
    )

    gfed_ba_1d.mask |= combined_mask

    # Calculate monthly averages.
    mon_avg_gfed_ba_1d = monthly_average_data(gfed_ba_1d, time_coord=jules_time_coord)

    assert mon_avg_gfed_ba_1d.shape[0] <= 12

    if jules_time_coord.cell(-1).point.day == 1 and jules_time_coord.shape[0] > 1:
        # Ignore the last month if there is only a single day in it.
        mon_avg_gfed_ba_1d = mon_avg_gfed_ba_1d[:-1]

    y_true = np.ma.getdata(mon_avg_gfed_ba_1d)[~np.ma.getmaskarray(mon_avg_gfed_ba_1d)]

    # Model kwargs.
    kwargs = dict(
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
        pop_den=np.zeros((land_pts,)) - 1,
        flash_rate=np.zeros((land_pts,)) - 1,
        ignition_method=1,
        fuel_build_up=obs_fuel_build_up_1d,
        fapar_diag_pft=np.repeat(
            np.expand_dims(obs_fapar_1d.data, 1), repeats=13, axis=1
        ),
        dry_days=unpack_wrapped(calculate_inferno_dry_days)(
            ls_rain, con_rain, threshold=1.0, timestep=timestep
        ),
        timestep=timestep,
        flammability_method=2,
        dryness_method=2,
        # fapar_factor=-4.83e1,
        # fapar_centre=4.0e-1,
        # fuel_build_up_factor=1.01e1,
        # fuel_build_up_centre=3.76e-1,
        # temperature_factor=8.01e-2,
        # temperature_centre=2.82e2,
        dry_day_factor=0.0,
        dry_day_centre=0.0,
        # rain_f=0.5,
        # vpd_f=2500,
        # dry_bal_factor=1,
        # dry_bal_centre=0,
    )

    model_ba = unpack_wrapped(multi_timestep_inferno)(
        **{**kwargs, **expanded_opt_kwargs}
    )

    if np.all(np.isclose(model_ba, 0, rtol=0, atol=1e-15)):
        return {"loss": 10000.0, "status": hyperopt.STATUS_FAIL}

    # Calculate monthly averages.
    avg_ba = monthly_average_data(model_ba, time_coord=jules_time_coord)
    # Limit data if needed.
    avg_ba = avg_ba[: mon_avg_gfed_ba_1d.shape[0]]
    assert avg_ba.shape == mon_avg_gfed_ba_1d.shape

    # Get ypred.
    y_pred = np.ma.getdata(avg_ba)[~np.ma.getmaskarray(mon_avg_gfed_ba_1d)]

    # Estimate the adjustment factor by minimising the NME.
    adj_factor = calculate_factor(y_true=y_true, y_pred=y_pred)

    y_pred *= adj_factor

    assert y_pred.shape == y_true.shape

    pad_func = partial(
        np.pad,
        pad_width=((0, 12 - mon_avg_gfed_ba_1d.shape[0]), (0, 0)),
        constant_values=0.0,
    )
    obs_pad = pad_func(mon_avg_gfed_ba_1d)
    # Apply adjustment factor similarly to y_pred.
    pred_pad = adj_factor * pad_func(avg_ba)
    mpd_val, ignored = mpd(obs=obs_pad, pred=pred_pad, return_ignored=True)

    if ignored > 5600:
        # Ensure that not too many samples are ignored.
        return {"loss": 10000.0, "status": hyperopt.STATUS_FAIL}

    scores = dict(
        # 1D stats
        r2=r2_score(y_true=y_true, y_pred=y_pred),
        nme=nme(obs=y_true, pred=y_pred),
        nmse=nmse(obs=y_true, pred=y_pred),
        loghist=loghist(obs=y_true, pred=y_pred, edges=np.linspace(0, 0.4, 20)),
        # Temporal stats.
        mpd=mpd_val,
    )

    if any(np.ma.is_masked(val) for val in scores.values()):
        return {"loss": 10000.0, "status": hyperopt.STATUS_FAIL}

    # Aim to minimise the combined score.
    return {
        # "loss": scores["nme"] + scores["nmse"] + scores["mpd"] + 2 * scores["loghist"],
        "loss": scores["nme"] + scores["mpd"],
        "status": hyperopt.STATUS_OK,
    }


if __name__ == "__main__":
    space = dict(
        # flammability_method=2,
        # dryness_method=2,
        fapar_factor=hp.uniform("fapar_factor", -50, -3),
        fapar_factor2=hp.uniform("fapar_factor2", -50, -3),
        fapar_factor3=hp.uniform("fapar_factor3", -50, -3),
        fapar_centre=hp.uniform("fapar_centre", 0.4, 0.75),
        fapar_centre2=hp.uniform("fapar_centre2", 0.4, 0.75),
        fapar_centre3=hp.uniform("fapar_centre3", 0.4, 0.75),
        fuel_build_up_alpha=hp.uniform("fuel_build_up_alpha", 5e-5, 1e-3),
        fuel_build_up_alpha2=hp.uniform("fuel_build_up_alpha2", 5e-5, 1e-3),
        fuel_build_up_alpha3=hp.uniform("fuel_build_up_alpha3", 5e-5, 1e-3),
        fuel_build_up_factor=hp.uniform("fuel_build_up_factor", 19, 30),
        fuel_build_up_factor2=hp.uniform("fuel_build_up_factor2", 19, 30),
        fuel_build_up_factor3=hp.uniform("fuel_build_up_factor3", 19, 30),
        fuel_build_up_centre=hp.uniform("fuel_build_up_centre", 0.3, 0.45),
        fuel_build_up_centre2=hp.uniform("fuel_build_up_centre2", 0.3, 0.45),
        fuel_build_up_centre3=hp.uniform("fuel_build_up_centre3", 0.3, 0.45),
        temperature_factor=hp.uniform("temperature_factor", 0.11, 0.17),
        temperature_factor2=hp.uniform("temperature_factor2", 0.11, 0.17),
        temperature_factor3=hp.uniform("temperature_factor3", 0.11, 0.17),
        temperature_centre=hp.uniform("temperature_centre", 270, 290),
        temperature_centre2=hp.uniform("temperature_centre2", 270, 290),
        temperature_centre3=hp.uniform("temperature_centre3", 270, 290),
        # dry_day_factor=hp.uniform("dry_day_factor", 0.001, 0.2),
        # dry_day_centre=hp.uniform("dry_day_centre", 150, 200),
        rain_f=hp.uniform("rain_f", 0.8, 2.0),
        rain_f2=hp.uniform("rain_f2", 0.8, 2.0),
        rain_f3=hp.uniform("rain_f3", 0.8, 2.0),
        vpd_f=hp.uniform("vpd_f", 500, 2000),
        vpd_f2=hp.uniform("vpd_f2", 500, 2000),
        vpd_f3=hp.uniform("vpd_f3", 500, 2000),
        dry_bal_factor=hp.uniform("dry_bal_factor", -60, -20),
        dry_bal_factor2=hp.uniform("dry_bal_factor2", -60, -20),
        dry_bal_factor3=hp.uniform("dry_bal_factor3", -60, -20),
        dry_bal_centre=hp.uniform("dry_bal_centre", -3, 3),
        dry_bal_centre2=hp.uniform("dry_bal_centre2", -3, 3),
        dry_bal_centre3=hp.uniform("dry_bal_centre3", -3, 3),
    )

    trials = MongoTrials("mongo://localhost:1234/ba/jobs", exp_key="exp8")

    out = fmin(
        fn=to_optimise,
        space=space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=10000,
        rstate=np.random.RandomState(0),
    )
