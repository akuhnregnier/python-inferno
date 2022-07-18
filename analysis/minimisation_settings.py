#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import sys
from functools import reduce
from itertools import islice
from operator import sub
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize
from tqdm import tqdm

from python_inferno.ba_model import ARCSINH_FACTOR, GPUConsAvgBAModel
from python_inferno.cache import cache, mark_dependency
from python_inferno.configuration import land_pts
from python_inferno.hyperopt import HyperoptSpace, get_space_template
from python_inferno.iter_opt import (
    ALWAYS_OPTIMISED,
    IGNORED,
    configuration_to_hyperopt_space_spec,
    next_configurations_iter,
)
from python_inferno.metrics import mpd, nme
from python_inferno.space import generate_space_spec


@mark_dependency
def _calculate_loss(*, pred_ba, true_1d, arcsinh_y_true):
    assert pred_ba.shape == (12, land_pts)
    assert not np.ma.isMaskedArray(pred_ba)

    # Calculate MPD.
    assert pred_ba.shape[0] == true_1d.shape[0] == 12
    assert pred_ba.size == true_1d.size

    mpd_val = mpd(obs=true_1d, pred=pred_ba)

    # Calculate ARCSINH NME.
    y_pred = pred_ba.ravel()
    arcsinh_nme_val = nme(
        obs=arcsinh_y_true,
        pred=np.arcsinh(ARCSINH_FACTOR * y_pred),
    )

    # Aim to minimise the combined score.
    loss = arcsinh_nme_val + mpd_val
    return loss


@cache(dependencies=[_calculate_loss])
def _min_space_opt(
    *,
    space,
    dryness_method,
    fuel_build_up_method,
    include_temperature,
    discrete_params,
    defaults,
    x0,
    minimizer_options,
):
    """Optimisation of the continuous (float) part of a given `space`.

    NOTE - x0 should be given in [0,1] space, e.g. the result of a previous call to an
        optimisation routine.

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
        gfed_ba_1d = ba_model.mon_avg_gfed_ba_1d.data

    arcsinh_y_true = np.arcsinh(ARCSINH_FACTOR * gfed_ba_1d.ravel())

    def to_optimise(**run_kwargs):
        loss = _calculate_loss(
            pred_ba=ba_model.run(**run_kwargs)["model_ba"],
            true_1d=gfed_ba_1d,
            arcsinh_y_true=arcsinh_y_true,
        )

        if np.isnan(float(loss)):
            return 10000

        return loss

    def to_optimise_with_discrete(x):
        return to_optimise(
            **space.inv_map_float_to_0_1(dict(zip(space.continuous_param_names, x))),
            **defaults,
        )

    res = minimize(
        to_optimise_with_discrete,
        x0=x0,
        method="L-BFGS-B",
        jac=None,
        bounds=[(0, 1)] * len(space.continuous_param_names),
        options={
            "maxiter": 300,
            "ftol": 1e-5,
            "eps": 1e-3,
            "disp": -1,
            **minimizer_options,
        },
    )

    ba_model.release()

    return res


def get_example_configurations(method_index=0, seed=0):
    with (
        Path(__file__).parent.parent
        / "tests"
        / "test_data"
        / f"best_params_litter_v2.pkl"
    ).open("rb") as f:
        params_dict = pickle.load(f)
    params = next(islice(iter(params_dict.values()), method_index, None))

    dryness_method = int(params["dryness_method"])
    fuel_build_up_method = int(params["fuel_build_up_method"])
    include_temperature = int(params["include_temperature"])

    # NOTE Full space template.
    space_template = get_space_template(
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=include_temperature,
    )

    discrete_param_names = HyperoptSpace(
        generate_space_spec(space_template)
    ).discrete_param_names

    # NOTE Constant.
    defaults = dict(
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=include_temperature,
    )
    discrete_params = {}

    # Most basic config possible.
    # Keys specify which parameters are potentially subject to optimisation. All other
    # keys will be taken from the optimal configuration as set out in `params`.
    start_config = defaults.copy()

    base_spec = {}

    for key in space_template:
        if key in ALWAYS_OPTIMISED:
            base_spec.update(generate_space_spec({key: space_template[key]}))
        elif key in IGNORED:
            if key in discrete_param_names:
                # NOTE Also constant.
                for pft_key in (f"{key}{suffix}" for suffix in ("", "2", "3")):
                    if pft_key in params:
                        discrete_params[pft_key] = params[pft_key]

                assert key in discrete_params, "At least 1 key should be present"
            else:
                raise ValueError(key)
        else:
            start_config[key] = 0

    rng = np.random.default_rng(seed)

    # Generate hypothetical chain of configurations randomly.
    configurations = []

    next_config = start_config.copy()
    next_configurations = True

    while True:
        next_configurations = list(
            next_configurations_iter({**start_config, **next_config})
        )
        if not next_configurations:
            break
        next_config = rng.choice([config for config, _ in next_configurations])
        configurations.append(next_config)

    return base_spec, defaults, discrete_params, configurations


def get_opt_results(
    *,
    base_spec,
    defaults,
    discrete_params,
    configuration,
    eps_vals,
    ftol,
    verbose,
    seed=None,
):
    assert tuple(eps_vals) == tuple(sorted(eps_vals, reverse=True))

    fun_results = np.zeros(len(eps_vals))
    nfev_results = np.zeros(len(eps_vals))

    space_spec, constants = configuration_to_hyperopt_space_spec(configuration)
    space = HyperoptSpace({**base_spec, **space_spec})

    if seed is None:
        x0 = [0.5] * len(space.continuous_param_names)
    else:
        x0 = list(np.random.default_rng(seed).random(len(space.continuous_param_names)))

    for i, eps in enumerate(tqdm(eps_vals, desc="eps")):
        res = _min_space_opt(
            space=space,
            dryness_method=defaults["dryness_method"],
            fuel_build_up_method=defaults["fuel_build_up_method"],
            include_temperature=defaults["include_temperature"],
            discrete_params=discrete_params,
            defaults={**defaults, **constants},
            minimizer_options=dict(maxiter=1000, ftol=ftol, eps=eps),
            x0=x0,
        )

        if verbose:
            print(
                f"Found minimum: {res.fun} at\n{res.x} in\nftol: {ftol:0.1e}, "
                f"eps: {eps:0.1e}, needed {res.nfev} evaluations."
            )
            print()

        fun_results[i] = res.fun
        nfev_results[i] = res.nfev

    return fun_results, nfev_results


if __name__ == "__main__":
    mpl.rc_file(Path(__file__).absolute().parent / "matplotlibrc")
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    ftol = 0.5e-8
    verbose = False

    eps_vals = (
        1e-2,
        5e-3,
        2.5e-3,
        1.5e-3,
        1e-3,
        8.8e-4,
        7.5e-4,
        6.4e-4,
        5e-4,
        4.2e-4,
        3.6e-4,  #
        3.2e-4,
        2.5e-4,
        1.6e-4,
        1e-4,
        6.6e-5,
        4e-5,
        1.8e-5,
        1.7e-5,
        1.65e-5,
        1.6e-5,
        1.55e-5,
        1.5e-5,
        1.49e-5,
        1.48e-5,
        1.47e-5,
        1.46e-5,
        1.45e-5,
        1.44e-5,
        1.43e-5,
        1.42e-5,
        1.41e-5,
        1.4e-5,
        1.39e-5,
        1.38e-5,
        1.37e-5,
        1.36e-5,
        1.35e-5,
        1.34e-5,
        1.33e-5,
        1.32e-5,
        1.31e-5,
        1.3e-5,
        1.2e-5,
        1.1e-5,
        1e-5,
        9e-6,
        8e-6,
        5e-6,
        1e-6,
        # 1e-8,
    )

    assert eps_vals == tuple(sorted(eps_vals, reverse=True))

    def get_params():
        for seed in [None, 1]:
            for method_index in range(4):
                (
                    base_spec,
                    defaults,
                    discrete_params,
                    configurations,
                ) = get_example_configurations(
                    method_index=method_index,
                    seed=0,
                )
                N = len(configurations)
                for configuration_index in [10, 21, N - 10, N - 5, N - 1]:
                    yield dict(
                        method_index=method_index,
                        configuration_index=configuration_index,
                        test_configuration=configurations[configuration_index],
                        base_spec=base_spec,
                        defaults=defaults,
                        discrete_params=discrete_params,
                        seed=seed,
                    )

    fig, axes = plt.subplots(1, 2)

    diffs_list = []

    for param_data in get_params():
        method_index = param_data["method_index"]
        configuration_index = param_data["configuration_index"]
        seed = param_data["seed"]

        fun_results, nfev_results = get_opt_results(
            base_spec=param_data["base_spec"],
            defaults=param_data["defaults"],
            discrete_params=param_data["discrete_params"],
            configuration=param_data["test_configuration"],
            eps_vals=eps_vals,
            ftol=ftol,
            verbose=verbose,
            seed=seed,
        )

        sel = fun_results < 0.9

        sel_eps = np.asarray(eps_vals)[sel]
        sel_fun = fun_results[sel]
        sel_nfev = nfev_results[sel]

        label = f"{method_index},{configuration_index},{seed}"

        axes[0].plot(sel_eps, sel_fun, label=label, linestyle="", marker="x")
        axes[1].plot(sel_nfev, sel_fun, label=label, linestyle="", marker="x")

        # We care about how much worse performance is compared to the best `eps`
        # value.
        best_loss = np.min(fun_results)
        diffs = fun_results - best_loss  # diffs >= 0 as fun_results >= best_loss
        if diffs_list:
            assert diffs.shape == diffs_list[-1].shape
        diffs_list.append(diffs)

    axes[0].set_xlabel("eps")
    axes[0].set_ylabel("loss")
    axes[0].set_xscale("log")

    axes[1].set_xlabel("nfev")
    axes[1].set_ylabel("loss")

    for ax in axes:
        ax.legend()

    diffs_arr = np.array(diffs_list)

    diff_stats = pd.DataFrame(
        {
            "mean": np.mean(diffs_arr, axis=0),
            "max": np.max(diffs_arr, axis=0),
            "median": np.median(diffs_arr, axis=0),
            "iqr": reduce(sub, np.quantile(diffs_arr, [0.75, 0.25], axis=0)),
        }
    )

    fig2, axes2 = plt.subplots(2, 2, sharex=True, figsize=(12, 6.5))
    for (i, (column, data)) in enumerate(diff_stats.iteritems()):
        ax = axes2.ravel()[i]

        ax.plot(eps_vals, data, linestyle="-", marker="x")
        ax.set_xscale("log")
        ax.set_ylabel("loss")
        ax.set_xlabel("eps")
        ax.set_title(column)

    plt.tight_layout()

    plt.close(fig)

    show = 1

    if show:
        plt.show()
    else:
        plt.close("all")
