# -*- coding: utf-8 -*-
from collections import defaultdict
from enum import Enum
from functools import partial

import numpy as np
from loguru import logger
from sklearn.metrics import r2_score

from .configuration import N_pft_groups, land_pts, npft
from .data import get_processed_climatological_data, timestep
from .metrics import loghist, mpd, nme, nmse
from .multi_timestep_inferno import multi_timestep_inferno
from .utils import (
    calculate_factor,
    expand_pft_params,
    monthly_average_data,
    unpack_wrapped,
)

Status = Enum("Status", ["SUCCESS", "FAIL"])


class BAModelException(RuntimeError):
    """Raised when inadequate BA model parameters are used."""


def process_params(*, opt_kwargs, defaults):
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
        logger.debug(f" - {name}: {val}")

    logger.debug("Opt param arrays")
    for name, vals in expanded_opt_kwargs.items():
        logger.debug(f" - {name}: {val}")

    def extract_param(key, dtype_str, size):
        if key in expanded_opt_kwargs:
            param = expanded_opt_kwargs.pop(key).astype(dtype_str)
            if param.shape == (N_pft_groups,) and size == npft:
                return expand_pft_params(param)
            assert param.shape == (size,), f"Expected: {(size,)}, got {param.shape}"
            return param
        elif key in single_opt_kwargs:
            return np.array([single_opt_kwargs.pop(key)] * size).astype(dtype_str)
        return np.array([defaults[key]] * size).astype(dtype_str)

    n_samples_pft = extract_param("fuel_build_up_n_samples", "int64", N_pft_groups)
    litter_tc = extract_param("litter_tc", "float64", npft)
    leaf_f = extract_param("leaf_f", "float64", npft)

    average_samples = int(single_opt_kwargs.pop("average_samples"))

    rain_f = single_opt_kwargs.pop(
        "rain_f", expanded_opt_kwargs.pop("rain_f", defaults["rain_f"])
    )
    vpd_f = single_opt_kwargs.pop(
        "vpd_f", expanded_opt_kwargs.pop("vpd_f", defaults["vpd_f"])
    )

    crop_f = single_opt_kwargs.pop("crop_f", defaults["crop_f"])
    assert "crop_f" not in expanded_opt_kwargs

    return (
        dict(
            n_samples_pft=n_samples_pft,
            litter_tc=litter_tc,
            leaf_f=leaf_f,
            average_samples=average_samples,
            rain_f=rain_f,
            vpd_f=vpd_f,
            crop_f=crop_f,
        ),
        single_opt_kwargs,
        expanded_opt_kwargs,
    )


def run_model(
    *,
    dryness_method,
    fuel_build_up_method,
    include_temperature,
    single_opt_kwargs,
    expanded_opt_kwargs,
    data_dict,
    missing_param_defaults=dict(
        temperature_factor=0.0,
        temperature_centre=0.0,
        temperature_shape=1.0,
        dry_bal_factor=1,
        dry_bal_centre=0,
        dry_bal_shape=1.0,
        dry_day_factor=0.0,
        dry_day_centre=0.0,
        dry_day_shape=1.0,
        litter_pool_factor=0.0,
        litter_pool_centre=0.0,
        litter_pool_shape=1.0,
        fuel_build_up_factor=0.0,
        fuel_build_up_centre=0.0,
        fuel_build_up_shape=1.0,
    ),
):
    # Model kwargs.
    kwargs = dict(
        ignition_method=1,
        timestep=timestep,
        flammability_method=2,
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=include_temperature,
        # These are not used for ignition mode 1, nor do they contain a temporal
        # coordinate.
        pop_den=np.zeros((land_pts,)) - 1,
        flash_rate=np.zeros((land_pts,)) - 1,
    )

    # Fill in missing keys with default values.
    for name, value in missing_param_defaults.items():
        if name not in kwargs:
            kwargs[name] = value

    model_ba = unpack_wrapped(multi_timestep_inferno)(
        **{
            **kwargs,
            **single_opt_kwargs,
            **expanded_opt_kwargs,
            **data_dict,
        }
    )
    return model_ba


def calculate_scores(*, model_ba, jules_time_coord, mon_avg_gfed_ba_1d):
    fail_out = (None, Status.FAIL, None, None)

    if np.all(np.isclose(model_ba, 0, rtol=0, atol=1e-15)):
        return fail_out

    # Calculate monthly averages.
    avg_ba = monthly_average_data(
        model_ba, time_coord=jules_time_coord, conservative=True
    )
    assert avg_ba.shape == mon_avg_gfed_ba_1d.shape

    y_pred = np.ma.getdata(avg_ba)[~np.ma.getmaskarray(mon_avg_gfed_ba_1d)]
    y_true = np.ma.getdata(mon_avg_gfed_ba_1d)[~np.ma.getmaskarray(mon_avg_gfed_ba_1d)]
    assert y_pred.shape == y_true.shape

    pad_func = partial(
        np.pad,
        pad_width=((0, 12 - mon_avg_gfed_ba_1d.shape[0]), (0, 0)),
        constant_values=0.0,
    )
    mpd_val, ignored = mpd(
        obs=pad_func(mon_avg_gfed_ba_1d), pred=pad_func(avg_ba), return_ignored=True
    )

    if ignored > 5600:
        # Ensure that not too many samples are ignored.
        return fail_out

    # Estimate the adjustment factor by minimising the NME.
    adj_factor = calculate_factor(y_true=y_true, y_pred=y_pred)
    y_pred *= adj_factor

    arcsinh_factor = 1e6
    arcsinh_y_true = np.arcsinh(arcsinh_factor * y_true)
    arcsinh_y_pred = np.arcsinh(arcsinh_factor * y_pred)
    arcsinh_adj_factor = calculate_factor(y_true=arcsinh_y_true, y_pred=arcsinh_y_pred)

    scores = dict(
        # 1D stats
        r2=r2_score(y_true=y_true, y_pred=y_pred),
        nme=nme(obs=y_true, pred=y_pred),
        arcsinh_nme=nme(obs=arcsinh_y_true, pred=arcsinh_adj_factor * arcsinh_y_pred),
        nmse=nmse(obs=y_true, pred=y_pred),
        loghist=loghist(obs=y_true, pred=y_pred, edges=np.linspace(0, 0.4, 20)),
        # Temporal stats.
        mpd=mpd_val,
    )

    if any(np.ma.is_masked(val) for val in scores.values()):
        return fail_out

    calc_factors = dict(
        adj_factor=adj_factor,
        arcsinh_factor=arcsinh_factor,
        arcsinh_adj_factor=arcsinh_adj_factor,
    )

    return scores, Status.SUCCESS, avg_ba, calc_factors


def get_pred_ba(
    *,
    defaults=dict(
        rain_f=0.3,
        vpd_f=400,
        crop_f=0.5,
        fuel_build_up_n_samples=0,
        litter_tc=1e-9,
        leaf_f=1e-3,
    ),
    dryness_method=2,
    fuel_build_up_method=1,
    include_temperature=1,
    **opt_kwargs,
):
    (
        data_params,
        single_opt_kwargs,
        expanded_opt_kwargs,
    ) = process_params(opt_kwargs=opt_kwargs, defaults=defaults)

    (
        data_dict,
        mon_avg_gfed_ba_1d,
        jules_time_coord,
    ) = get_processed_climatological_data(
        litter_tc=data_params["litter_tc"],
        leaf_f=data_params["leaf_f"],
        n_samples_pft=data_params["n_samples_pft"],
        average_samples=data_params["average_samples"],
        rain_f=data_params["rain_f"],
        vpd_f=data_params["vpd_f"],
    )

    # Shallow copy to allow popping of the dictionary without affecting the
    # memoized copy.
    data_dict = data_dict.copy()
    # Extract variables not used further below.
    obs_pftcrop_1d = data_dict.pop("obs_pftcrop_1d")

    model_ba = run_model(
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=include_temperature,
        single_opt_kwargs=single_opt_kwargs,
        expanded_opt_kwargs=expanded_opt_kwargs,
        data_dict=data_dict,
    )

    # Modify the predicted BA using the crop fraction (i.e. assume a certain
    # proportion of cropland never burns, even though this may be the case in
    # given the weather conditions).
    model_ba *= 1 - data_params["crop_f"] * obs_pftcrop_1d

    scores, status, avg_ba, calc_factors = calculate_scores(
        model_ba=model_ba,
        jules_time_coord=jules_time_coord,
        mon_avg_gfed_ba_1d=mon_avg_gfed_ba_1d,
    )
    if status is Status.FAIL:
        raise BAModelException()

    assert status is Status.SUCCESS

    return avg_ba, scores, mon_avg_gfed_ba_1d, calc_factors


def gen_to_optimise(
    *,
    fail_func,
    success_func,
):
    def to_optimise(**kwargs):
        try:
            scores = get_pred_ba(**kwargs)[1]
        except BAModelException:
            return fail_func()

        # Aim to minimise the combined score.
        # loss = scores["nme"] + scores["nmse"] + scores["mpd"] + 2 * scores["loghist"]
        loss = scores["arcsinh_nme"] + scores["mpd"]
        logger.debug(f"loss: {loss:0.6f}")
        return success_func(loss)

    return to_optimise
