# -*- coding: utf-8 -*-
from functools import wraps

import numpy as np
from sympy import (
    Abs,
    Array,
    Eq,
    Piecewise,
    acos,
    asinh,
    atan2,
    cos,
    exp,
    pi,
    sin,
    symarray,
    symbols,
)

from .ba_model import ARCSINH_FACTOR
from .cache import cache, mark_dependency
from .configuration import (
    N_pft_groups,
    m2_in_km2,
    n_total_pft,
    npft,
    pft_group_map,
    s_in_month,
)
from .utils import linspace_no_endpoint

# NOTE Ignition mode 1 only!!
nat_ign_l = 2.7 / s_in_month / m2_in_km2 / 12.0 * 0.75
man_ign_l = 1.5 / s_in_month / m2_in_km2
IGNITIONS = man_ign_l + nat_ign_l


def wrap_lambdify_kwargs(kwarg_names):
    """

    Args:
        kwarg_names (iterable of str): Keyword arg names *in order*.

    Returns:
        Decorator to be applied to lambdified function.

    """

    def deco(f):
        @wraps(f)
        def wrapped(**kwargs):
            # Arrange given `kwargs` in order of `kwarg_names`.
            args = []
            for name in kwarg_names:
                args.append(kwargs[name])
            # Call with ordered args.
            return f(*args)

        return wrapped

    return deco


@mark_dependency
def sigmoid(x, factor, centre, shape):
    """Apply generalised sigmoid with slope determine by `factor`, position by
    `centre`, and shape by `shape`, with the result being in [0, 1]."""
    return (1.0 + exp(factor * shape * (centre - x))) ** (-1.0 / shape)


@mark_dependency
def calc_flam(
    *,
    temp_l,
    temperature_weight,
    temperature_factor,
    temperature_centre,
    temperature_shape,
    dry_days,
    dryness_weight,
    dry_day_factor,
    dry_day_centre,
    dry_day_shape,
    fuel_build_up,
    fuel_weight,
    fuel_build_up_factor,
    fuel_build_up_centre,
    fuel_build_up_shape,
    fapar,
    fapar_weight,
    fapar_factor,
    fapar_centre,
    fapar_shape,
):
    temperature_sigmoid = sigmoid(
        temp_l, temperature_factor, temperature_centre, temperature_shape
    )
    weighted_temp = 1 + temperature_weight * (temperature_sigmoid - 1)

    dry_factor = sigmoid(dry_days, dry_day_factor, dry_day_centre, dry_day_shape)
    weighted_dry = 1 + dryness_weight * (dry_factor - 1)

    fuel_factor = sigmoid(
        fuel_build_up,
        fuel_build_up_factor,
        fuel_build_up_centre,
        fuel_build_up_shape,
    )
    weighted_fuel = 1 + fuel_weight * (fuel_factor - 1)

    fapar_sigmoid = sigmoid(fapar, fapar_factor, fapar_centre, fapar_shape)
    weighted_fapar = 1 + fapar_weight * (fapar_sigmoid - 1)

    # Convert fuel build-up index to flammability factor.
    return weighted_dry * weighted_temp * weighted_fuel * weighted_fapar


@mark_dependency
def calc_ba(flam, avg_ba):
    return flam * IGNITIONS * avg_ba


@mark_dependency
def np_phase_0d(arr):
    theta = 2 * np.pi * linspace_no_endpoint(0, 1, 12)
    lx = np.sum(arr * np.cos(theta))
    ly = np.sum(arr * np.sin(theta))
    return np.arctan2(lx, ly)


@mark_dependency
def phase(arr):
    lx = 0
    ly = 0
    for i in range(12):
        lx += arr[i] * cos(2 * pi * i / 12.0)
        ly += arr[i] * sin(2 * pi * i / 12.0)

    phase = atan2(lx, ly)

    return phase


@mark_dependency
def _mpd(*, obs, pred):
    obs_phase = np_phase_0d(obs)
    pred_phase = phase(pred)

    out = (1 / pi) * acos(cos(pred_phase - obs_phase))
    return out


@mark_dependency
def all_close_zero(x):
    all_close = 1
    for i in range(12):
        all_close *= Piecewise((1, x[i] < 1e-15), (0, True))

    # All elements in `x` are below the threshold if `all_close == 1` after the loop.
    return Eq(all_close, 1)


@mark_dependency
def mpd(*, obs, pred):
    return _mpd(obs=obs, pred=pred)

    # XXX - These checks make the code too complicated to compute - so perform these
    # checks before / after using NumPy somehow?
    # return Piecewise(
    #     (nan, all_close_zero(obs)),
    #     (nan, all_close_zero(pred)),
    #     (_mpd(obs=obs, pred=pred), True),
    # )


@mark_dependency
def mean_1d(arr):
    arr_sum = 0
    for i in range(arr.shape[0]):
        arr_sum += arr[i]
    return arr_sum / arr.shape[0]


# XXX NOTE That the NME denominator should be calculated wrt all grid cells.


@mark_dependency
def asinh_nme(*, obs, pred):
    asinh_obs = np.arcsinh(ARCSINH_FACTOR * obs)
    asinh_pred = pred.applyfunc(lambda x: asinh(ARCSINH_FACTOR * x))

    mean_asinh_obs = np.mean(asinh_obs)
    diff = asinh_obs - mean_asinh_obs
    denom = np.mean(np.abs(diff))

    mean_abs_diff = 0
    for i in range(12):
        mean_abs_diff += Abs(asinh_pred[i] - asinh_obs[i])
    mean_abs_diff /= 12.0

    out = mean_abs_diff / denom
    return out


@mark_dependency
def calc_grid_cell_metrics(
    *,
    # Constant parameters.
    avg_weights,
    obs_ba,
    # Parameters.
    overall_scale,
    temperature_weight,
    temperature_factor,
    temperature_centre,
    temperature_shape,
    fapar_weight,
    fapar_factor,
    fapar_centre,
    fapar_shape,
    dryness_weight,
    dry_day_factor,
    dry_day_centre,
    dry_day_shape,
    fuel_weight,
    fuel_build_up_factor,
    fuel_build_up_centre,
    fuel_build_up_shape,
    crop_f,
    avg_ba,
    # Data.
    temp_l,
    fapar,
    dry_days,
    fuel_build_up,
    crop,
    frac,
):
    # XXX - this would be a representation of the underlying flammability function.
    # calc_flam = Function("calc_flam")

    pred_ba = [0] * 12

    for source_ti in range(avg_weights.shape[0]):
        source_ba = 0
        for pft_i in range(npft):
            pft_group_i = pft_group_map[pft_i]
            source_ba += frac[source_ti, pft_i] * calc_ba(
                # calc_flam(source_ti, pft_i),
                calc_flam(
                    temp_l=temp_l[source_ti, pft_i],
                    dry_days=dry_days[source_ti],
                    fuel_build_up=fuel_build_up[source_ti, pft_i],
                    fapar=fapar[source_ti, pft_i],
                    temperature_weight=temperature_weight[pft_group_i],
                    temperature_factor=temperature_factor[pft_group_i],
                    temperature_centre=temperature_centre[pft_group_i],
                    temperature_shape=temperature_shape[pft_group_i],
                    dryness_weight=dryness_weight[pft_group_i],
                    dry_day_factor=dry_day_factor[pft_group_i],
                    dry_day_centre=dry_day_centre[pft_group_i],
                    dry_day_shape=dry_day_shape[pft_group_i],
                    fuel_weight=fuel_weight[pft_group_i],
                    fuel_build_up_factor=fuel_build_up_factor[pft_group_i],
                    fuel_build_up_centre=fuel_build_up_centre[pft_group_i],
                    fuel_build_up_shape=fuel_build_up_shape[pft_group_i],
                    fapar_weight=fapar_weight[pft_group_i],
                    fapar_factor=fapar_factor[pft_group_i],
                    fapar_centre=fapar_centre[pft_group_i],
                    fapar_shape=fapar_shape[pft_group_i],
                ),
                avg_ba[pft_i],
            )

        source_ba *= overall_scale
        source_ba *= 1 - crop_f * crop[source_ti]

        # Temporal averaging.
        for target_ti in range(12):
            pred_ba[target_ti] += source_ba * avg_weights[source_ti, target_ti]

    pred_ba = Array(pred_ba)

    mpd_val = mpd(obs=obs_ba, pred=pred_ba)
    asinh_nme_val = asinh_nme(obs=obs_ba, pred=pred_ba)

    return asinh_nme_val, mpd_val


@mark_dependency
@cache(
    dependencies=[
        _mpd,
        all_close_zero,
        asinh_nme,
        calc_ba,
        calc_flam,
        calc_grid_cell_metrics,
        mean_1d,
        mpd,
        np_phase_0d,
        phase,
        sigmoid,
    ]
)
def get_grid_cell_metrics(*, avg_weights, obs_ba):
    assert avg_weights.ndim == 2, "Expect (M, 12)."
    assert avg_weights.shape[1] == 12, "Target Nt should be 12."
    assert obs_ba.shape == (12,)

    Nin = avg_weights.shape[0]

    overall_scale = symbols("overall_scale")

    temperature_weight = symarray("temperature_weight", (N_pft_groups,))
    temperature_factor = symarray("temperature_factor", (N_pft_groups,))
    temperature_centre = symarray("temperature_centre", (N_pft_groups,))
    temperature_shape = symarray("temperature_shape", (N_pft_groups,))

    fapar_weight = symarray("fapar_weight", (N_pft_groups,))
    fapar_factor = symarray("fapar_factor", (N_pft_groups,))
    fapar_centre = symarray("fapar_centre", (N_pft_groups,))
    fapar_shape = symarray("fapar_shape", (N_pft_groups,))

    dryness_weight = symarray("dryness_weight", (N_pft_groups,))
    dry_day_factor = symarray("dry_day_factor", (N_pft_groups,))
    dry_day_centre = symarray("dry_day_centre", (N_pft_groups,))
    dry_day_shape = symarray("dry_day_shape", (N_pft_groups,))

    fuel_weight = symarray("fuel_weight", (N_pft_groups,))
    fuel_build_up_factor = symarray("fuel_build_up_factor", (N_pft_groups,))
    fuel_build_up_centre = symarray("fuel_build_up_centre", (N_pft_groups,))
    fuel_build_up_shape = symarray("fuel_build_up_shape", (N_pft_groups,))

    crop_f = symbols("crop_f")

    avg_ba = symarray("avg_ba", (npft,))

    temp_l = symarray("temp_l", (Nin, n_total_pft))
    fapar = symarray("fapar", (Nin, npft))
    dry_days = symarray("dry_days", (Nin,))
    fuel_build_up = symarray("fuel_build_up", (Nin, npft))
    frac = symarray("frac", (Nin, n_total_pft))
    crop = symarray("crop", (Nin,))

    grid_cell_nme, grid_cell_mpd = calc_grid_cell_metrics(
        avg_weights=avg_weights,
        obs_ba=obs_ba,
        overall_scale=overall_scale,
        temperature_weight=temperature_weight,
        temperature_factor=temperature_factor,
        temperature_centre=temperature_centre,
        temperature_shape=temperature_shape,
        fapar_weight=fapar_weight,
        fapar_factor=fapar_factor,
        fapar_centre=fapar_centre,
        fapar_shape=fapar_shape,
        dryness_weight=dryness_weight,
        dry_day_factor=dry_day_factor,
        dry_day_centre=dry_day_centre,
        dry_day_shape=dry_day_shape,
        fuel_weight=fuel_weight,
        fuel_build_up_factor=fuel_build_up_factor,
        fuel_build_up_centre=fuel_build_up_centre,
        fuel_build_up_shape=fuel_build_up_shape,
        crop_f=crop_f,
        avg_ba=avg_ba,
        temp_l=temp_l,
        fapar=fapar,
        dry_days=dry_days,
        fuel_build_up=fuel_build_up,
        crop=crop,
        frac=frac,
    )

    symbol_dict = dict(
        overall_scale=overall_scale,
        temperature_weight=temperature_weight,
        temperature_factor=temperature_factor,
        temperature_centre=temperature_centre,
        temperature_shape=temperature_shape,
        fapar_weight=fapar_weight,
        fapar_factor=fapar_factor,
        fapar_centre=fapar_centre,
        fapar_shape=fapar_shape,
        dryness_weight=dryness_weight,
        dry_day_factor=dry_day_factor,
        dry_day_centre=dry_day_centre,
        dry_day_shape=dry_day_shape,
        fuel_weight=fuel_weight,
        fuel_build_up_factor=fuel_build_up_factor,
        fuel_build_up_centre=fuel_build_up_centre,
        fuel_build_up_shape=fuel_build_up_shape,
        crop_f=crop_f,
        avg_ba=avg_ba,
        temp_l=temp_l,
        fapar=fapar,
        dry_days=dry_days,
        fuel_build_up=fuel_build_up,
        crop=crop,
        frac=frac,
    )

    return symbol_dict, grid_cell_nme, grid_cell_mpd
