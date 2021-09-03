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
from python_inferno.data import get_processed_climatological_data, timestep
from python_inferno.hyperopt import Space
from python_inferno.metrics import loghist, mpd, nme, nmse
from python_inferno.multi_timestep_inferno import multi_timestep_inferno
from python_inferno.utils import (
    calculate_factor,
    expand_pft_params,
    monthly_average_data,
    unpack_wrapped,
)


def to_optimise(opt_kwargs):
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
    single_opt_kwargs = dict()

    for name, vals in expanded_opt_tmp.items():
        if len(vals) == 3:
            expanded_opt_kwargs[name] = expand_pft_params(vals)
        elif len(vals) == 1:
            single_opt_kwargs[name] = vals[0]
        else:
            raise ValueError(f"Unexpected number of values {len(vals)}.")

    print("Normal params")
    for name, val in single_opt_kwargs.items():
        print(" -", name, val)

    print("Opt param arrays")
    for name, vals in expanded_opt_kwargs.items():
        print(" -", name, vals)

    if "fuel_build_up_n_samples" in expanded_opt_kwargs:
        n_samples_pft = expanded_opt_kwargs.pop("fuel_build_up_n_samples").astype(
            "int64"
        )
    else:
        n_samples_pft = np.array(
            [single_opt_kwargs.pop("fuel_build_up_n_samples")] * 13
        ).astype("int64")
    assert n_samples_pft.shape == (13,)

    average_samples = int(single_opt_kwargs.pop("average_samples"))

    (
        data_dict,
        mon_avg_gfed_ba_1d,
        jules_time_coord,
    ) = get_processed_climatological_data(
        n_samples_pft=n_samples_pft, average_samples=average_samples
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
        **{**kwargs, **expanded_opt_kwargs, **single_opt_kwargs, **data_dict}
    )

    if np.all(np.isclose(model_ba, 0, rtol=0, atol=1e-15)):
        return {"loss": 10000.0, "status": hyperopt.STATUS_FAIL}

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
    space_template = dict(
        fapar_factor=(1, [(-50, -1)], hp.uniform),
        fapar_centre=(1, [(-0.1, 1.1)], hp.uniform),
        fuel_build_up_n_samples=(1, [(100, 1301, 400)], hp.quniform),
        fuel_build_up_factor=(1, [(0.5, 30)], hp.uniform),
        fuel_build_up_centre=(1, [(0.0, 0.5)], hp.uniform),
        temperature_factor=(1, [(0.07, 0.2)], hp.uniform),
        temperature_centre=(1, [(260, 295)], hp.uniform),
        rain_f=(1, [(0.8, 2.0)], hp.uniform),
        vpd_f=(1, [(400, 2200)], hp.uniform),
        dry_bal_factor=(1, [(-100, -1)], hp.uniform),
        dry_bal_centre=(1, [(-3, 3)], hp.uniform),
        # Averaged samples between ~1 week and ~1 month (4 hrs per sample).
        average_samples=(1, [(40, 161, 60)], hp.quniform),
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

    space = Space(spec)

    trials = MongoTrials(
        "mongo://maritimus.webredirect.org:1234/ba/jobs", exp_key="exp30"
    )

    part_fmin = partial(
        fmin,
        fn=to_optimise,
        algo=tpe.suggest,
        trials=trials,
        rstate=np.random.RandomState(0),
    )

    out1 = part_fmin(space=space.render(), max_evals=5000)

    # shrink_space = space.shrink(out1, factor=0.2)
    # out2 = part_fmin(space=shrink_space.render(), max_evals=4000)
