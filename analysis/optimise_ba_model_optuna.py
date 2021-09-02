#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
from collections import defaultdict
from functools import partial

import iris
import numpy as np
import optuna
from sklearn.metrics import r2_score

from python_inferno.configuration import land_pts
from python_inferno.data import load_data
from python_inferno.metrics import loghist, mpd, nme, nmse
from python_inferno.multi_timestep_inferno import multi_timestep_inferno
from python_inferno.optuna import OptunaSpace
from python_inferno.precip_dry_day import calculate_inferno_dry_days
from python_inferno.utils import (
    calculate_factor,
    expand_pft_params,
    monthly_average_data,
    temporal_processing,
    unpack_wrapped,
)

timestep = 4 * 60 * 60


space_template = dict(
    fapar_factor=(3, [(-50, -1)], "suggest_float"),
    fapar_centre=(3, [(-0.1, 1.1)], "suggest_float"),
    fuel_build_up_n_samples=(3, [(100, 1200, 100)], "suggest_int"),
    fuel_build_up_factor=(3, [(0.5, 30)], "suggest_float"),
    fuel_build_up_centre=(3, [(0.0, 0.5)], "suggest_float"),
    temperature_factor=(3, [(0.07, 0.2)], "suggest_float"),
    temperature_centre=(3, [(260, 295)], "suggest_float"),
    rain_f=(3, [(0.8, 2.0)], "suggest_float"),
    vpd_f=(3, [(400, 2200)], "suggest_float"),
    dry_bal_factor=(3, [(-100, -1)], "suggest_float"),
    dry_bal_centre=(3, [(-3, 3)], "suggest_float"),
    # Averaged samples between ~1 week and ~1 month (4 hrs per sample).
    average_samples=(1, [(40, 160, 60)], "suggest_int"),
)
# Generate the actual `space` from the template.
spec = dict()
for name, template in space_template.items():
    bounds = template[1]
    if len(bounds) == 1:
        # Use the same bounds for all PFTs if only one are given.
        bounds *= template[0]
    for i, bound in zip(range(1, template[0] + 1), bounds):
        if i == 1:
            arg_name = name
        else:
            arg_name = name + str(i)

        spec[arg_name] = (template[2], *bound)

space = OptunaSpace(spec, remap_float_to_0_1=True, replicate_pft_groups=True)


def to_optimise(opt_kwargs):
    if not hasattr(to_optimise, "loaded_data"):
        # Cache the data for future function calls.
        to_optimise.loaded_data = load_data(N=None)

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
    ) = to_optimise.loaded_data

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

    assert "fuel_build_up_n_samples" in expanded_opt_kwargs
    n_samples_pft = expanded_opt_kwargs.pop("fuel_build_up_n_samples").astype("int64")
    assert "fuel_build_up_n_samples" not in expanded_opt_kwargs

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
        average_samples=opt_kwargs["average_samples"],
        aggregator={
            name: {"dry_days": iris.analysis.MAX, "t1p5m_tile": iris.analysis.MAX}.get(
                name, iris.analysis.MEAN
            )
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

    # Model kwargs.
    kwargs = dict(
        ignition_method=1,
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
        # These are not used for ignition mode 1, nor do they contain a temporal
        # coordinate.
        pop_den=np.zeros((land_pts,)) - 1,
        flash_rate=np.zeros((land_pts,)) - 1,
    )

    model_ba = unpack_wrapped(multi_timestep_inferno)(
        **{**kwargs, **expanded_opt_kwargs, **data_dict}
    )

    if np.all(np.isclose(model_ba, 0, rtol=0, atol=1e-15)):
        raise optuna.exceptions.TrialPruned

    # Calculate monthly averages.
    avg_ba = monthly_average_data(model_ba, time_coord=jules_time_coord)
    assert avg_ba.shape == mon_avg_gfed_ba_1d.shape

    # Get ypred.
    y_pred = np.ma.getdata(avg_ba)[~np.ma.getmaskarray(mon_avg_gfed_ba_1d)]

    y_true = np.ma.getdata(mon_avg_gfed_ba_1d)[~np.ma.getmaskarray(mon_avg_gfed_ba_1d)]

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
        raise optuna.exceptions.TrialPruned

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
        raise optuna.exceptions.TrialPruned

    # Aim to minimise the combined score.
    return (
        # scores["nme"] + scores["nmse"] + scores["mpd"] + 2 * scores["loghist"]
        scores["nme"]
        + scores["mpd"]
    )


def objective(trial):
    gc.collect()

    suggested_params = space.suggest(trial)
    loss = to_optimise(suggested_params)

    gc.collect()
    return loss


if __name__ == "__main__":
    study_name = "optuna4"
    study = optuna.load_study(
        sampler=optuna.samplers.CmaEsSampler(restart_strategy="ipop"),
        study_name=f"{study_name}",
        storage=f"mysql://alex@maritimus.webredirect.org/{study_name}",
    )
    study.optimize(objective, n_trials=5000)
