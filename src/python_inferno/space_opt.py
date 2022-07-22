# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.optimize import basinhopping, minimize

from .ba_model import (
    ARCSINH_FACTOR,
    BAModel,
    GPUConsAvgBAModel,
    ModAvgCropMixin,
    ModelParams,
    gen_to_optimise,
)
from .basinhopping import BoundedSteps, Recorder
from .cache import cache, mark_dependency
from .configuration import land_pts
from .data import get_processed_climatological_data
from .metrics import mpd, nme


def fail_func(*args, **kwargs):
    return 10000.0


def success_func(loss, *args, **kwargs):
    return float(loss)


# NOTE CPP dependencies (amongst others) are not taken into account here, so some code
# changes will require manual cache deletion!
@mark_dependency
@cache(
    dependencies=[
        BAModel.__init__,
        ModAvgCropMixin._mod_avg_crop,
        ModelParams.process_kwargs,
        gen_to_optimise,
        get_processed_climatological_data,
    ],
    ignore=["verbose", "_uncached_data"],
)
def space_opt(
    *,
    space,
    dryness_method,
    fuel_build_up_method,
    include_temperature,
    discrete_params,
    opt_record_dir="opt_record",
    defaults=None,
    basinhopping_options=None,
    minimizer_options=None,
    x0=None,
    return_res=False,
    verbose=True,
    _uncached_data=True,
):
    """Optimisation of the continuous (float) part of a given `space`."""
    to_optimise = gen_to_optimise(
        fail_func=fail_func,
        success_func=success_func,
        # Init (data) params.
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=include_temperature,
        _uncached_data=_uncached_data,
        **discrete_params,
    )

    defaults_dict = defaults if defaults is not None else {}

    def to_optimise_with_discrete(x):
        return to_optimise(
            **space.inv_map_float_to_0_1(dict(zip(space.continuous_param_names, x))),
            **defaults_dict,
        )

    recorder = Recorder(record_dir=Path(os.environ["EPHEMERAL"]) / opt_record_dir)

    def basinhopping_callback(x, f, accept):
        # NOTE: Parameters recorded here are authoritative, since hyperopt will not
        # properly report values modified as in e.g. `mod_quniform`.
        values = {
            **space.inv_map_float_to_0_1(dict(zip(space.continuous_param_names, x))),
            **discrete_params,
            **defaults_dict,
        }
        values["dryness_method"] = dryness_method
        values["fuel_build_up_method"] = fuel_build_up_method
        values["include_temperature"] = include_temperature

        if verbose:
            logger.info(f"Minimum found | loss: {f:0.6f}")

        for name, val in values.items():
            if verbose:
                logger.info(f" - {name}: {val}")

        if recorder is not None:
            recorder.record(values, f)

            # Update record in file.
            recorder.dump()

    minimizer_options_dict = minimizer_options if minimizer_options is not None else {}
    basinhopping_options_dict = (
        basinhopping_options if basinhopping_options is not None else {}
    )

    res = basinhopping(
        to_optimise_with_discrete,
        x0=space.continuous_x0_mid if x0 is None else x0,
        seed=0,
        callback=basinhopping_callback,
        take_step=BoundedSteps(
            stepsize=0.3, rng=np.random.default_rng(0), verbose=verbose
        ),
        **{
            "disp": verbose,
            "minimizer_kwargs": dict(
                method="L-BFGS-B",
                jac=None,
                bounds=[(0, 1)] * len(space.continuous_param_names),
                options={
                    "maxiter": 1000,
                    "ftol": 5e-9,
                    "eps": 3.6e-4,
                    **minimizer_options_dict,
                },
            ),
            "T": 0.05,
            "niter": 100,
            "niter_success": 15,
            **basinhopping_options_dict,
        },
    )

    if return_res:
        return res
    return res.fun


@mark_dependency
@cache(dependencies=[mpd, nme])
def calculate_split_loss(*, pred_ba, point_grid, sel_true_1d, sel_arcsinh_y_true):
    assert pred_ba.shape == (12, land_pts)
    sel_pred_ba = pred_ba[:, point_grid]

    assert not np.ma.isMaskedArray(sel_pred_ba)

    # Calculate MPD.
    assert sel_pred_ba.shape[0] == sel_true_1d.shape[0] == 12
    assert sel_pred_ba.size == sel_true_1d.size

    mpd_val = mpd(obs=sel_true_1d, pred=sel_pred_ba)

    # Calculate ARCSINH NME.
    y_pred = sel_pred_ba.ravel()
    arcsinh_nme_val = nme(
        obs=sel_arcsinh_y_true,
        pred=np.arcsinh(ARCSINH_FACTOR * y_pred),
    )

    # Aim to minimise the combined score.
    loss = float(arcsinh_nme_val + mpd_val)
    return loss


@mark_dependency
@cache(dependencies=[calculate_split_loss])
def split_min_space_opt(
    *,
    space,
    dryness_method,
    fuel_build_up_method,
    include_temperature,
    discrete_params,
    train_grid,
    defaults=None,
    minimizer_options=None,
    x0,
):
    """Optimisation of the continuous (float) part of a given `space`.

    NOTE - x0 should be given in [0,1] space, e.g. the result of a previous call to an
        optimisation routine.

    train_grid - array of integers specifying which land points should be used for
        optimisation.

    """
    # NOTE This routine is specialised for calculation of MPD and ARCSINH_NME.

    ba_model = GPUConsAvgBAModel(
        _uncached_data=False,
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=include_temperature,
        **discrete_params,
    )

    if np.ma.isMaskedArray(ba_model.mon_avg_gfed_ba_1d):
        assert not np.any(ba_model.mon_avg_gfed_ba_1d.mask)
        mon_avg_gfed_ba_1d = ba_model.mon_avg_gfed_ba_1d.data

    sel_gfed_ba_1d = np.ascontiguousarray(mon_avg_gfed_ba_1d[:, train_grid])
    sel_arcsinh_y_true = np.arcsinh(ARCSINH_FACTOR * sel_gfed_ba_1d.ravel())

    def to_optimise(**run_kwargs):
        loss = float(
            calculate_split_loss(
                pred_ba=ba_model.run(**run_kwargs)["model_ba"],
                point_grid=train_grid,
                sel_true_1d=sel_gfed_ba_1d,
                sel_arcsinh_y_true=sel_arcsinh_y_true,
            )
        )

        if np.isnan(loss):
            return fail_func()

        return success_func(loss)

    defaults_dict = defaults if defaults is not None else {}

    def to_optimise_with_discrete(x):
        return to_optimise(
            **space.inv_map_float_to_0_1(dict(zip(space.continuous_param_names, x))),
            **defaults_dict,
        )

    minimizer_options_dict = minimizer_options if minimizer_options is not None else {}

    res = minimize(
        to_optimise_with_discrete,
        x0=x0,
        method="L-BFGS-B",
        jac=None,
        bounds=[(0, 1)] * len(space.continuous_param_names),
        options={
            "maxiter": 1000,
            "ftol": 5e-9,
            "eps": 3.6e-4,
            **minimizer_options_dict,
        },
    )

    ba_model.release()

    return res
