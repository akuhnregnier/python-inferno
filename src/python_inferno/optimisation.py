# -*- coding: utf-8 -*-
from collections import defaultdict
from functools import partial

import numpy as np
from loguru import logger
from sklearn.metrics import r2_score

from .configuration import N_pft_groups, land_pts
from .data import get_processed_climatological_data, timestep
from .metrics import loghist, mpd, nme, nmse
from .multi_timestep_inferno import multi_timestep_inferno
from .utils import calculate_factor, monthly_average_data, unpack_wrapped


def gen_to_optimise(
    fail_func,
    success_func,
):
    def to_optimise(opt_kwargs, defaults=dict(rain_f=0.3, vpd_f=400)):
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
                expanded_opt_kwargs[name] = np.asarray(vals)
            elif len(vals) == 1:
                single_opt_kwargs[name] = vals[0]
            else:
                raise ValueError(f"Unexpected number of values {len(vals)}.")

        logger.debug("Normal params")
        for name, val in single_opt_kwargs.items():
            logger.debug(" -", name, val)

        logger.debug("Opt param arrays")
        for name, vals in expanded_opt_kwargs.items():
            logger.debug(" -", name, vals)

        if "fuel_build_up_n_samples" in expanded_opt_kwargs:
            n_samples_pft = expanded_opt_kwargs.pop("fuel_build_up_n_samples").astype(
                "int64"
            )
            assert n_samples_pft.shape == (N_pft_groups,)
        else:
            n_samples_pft = np.array(
                [single_opt_kwargs.pop("fuel_build_up_n_samples")] * N_pft_groups
            ).astype("int64")

        average_samples = int(single_opt_kwargs.pop("average_samples"))

        proc_kwargs = dict()
        for key in ("rain_f", "vpd_f"):
            proc_kwargs[key] = single_opt_kwargs.pop(
                key, expanded_opt_kwargs.pop(key, defaults[key])
            )

        (
            data_dict,
            mon_avg_gfed_ba_1d,
            jules_time_coord,
        ) = get_processed_climatological_data(
            n_samples_pft=n_samples_pft, average_samples=average_samples, **proc_kwargs
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
            # TODO - calculation of dry_bal is carried out during data
            # loading/processing now
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
            return fail_func()

        # Calculate monthly averages.
        avg_ba = monthly_average_data(model_ba, time_coord=jules_time_coord)
        assert avg_ba.shape == mon_avg_gfed_ba_1d.shape

        # Get ypred.
        y_pred = np.ma.getdata(avg_ba)[~np.ma.getmaskarray(mon_avg_gfed_ba_1d)]

        y_true = np.ma.getdata(mon_avg_gfed_ba_1d)[
            ~np.ma.getmaskarray(mon_avg_gfed_ba_1d)
        ]

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
            return fail_func()

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
            return fail_func()

        # Aim to minimise the combined score.
        # loss = scores["nme"] + scores["nmse"] + scores["mpd"] + 2 * scores["loghist"]
        loss = scores["nme"] + scores["mpd"]
        logger.debug(f"loss: {loss:0.6f}")
        return success_func(loss)

    return to_optimise
