# -*- coding: utf-8 -*-

import numpy as np
import pytest
from scipy.optimize import basinhopping, minimize

from python_inferno.ba_model import gen_to_optimise
from python_inferno.basinhopping import BoundedSteps
from python_inferno.hyperopt import HyperoptSpace, get_space_template
from python_inferno.iter_opt import (
    ALWAYS_OPTIMISED,
    IGNORED,
    configuration_to_hyperopt_space_spec,
    get_next_x0,
    next_configurations_iter,
)
from python_inferno.space import generate_space_spec
from python_inferno.space_opt import fail_func, success_func

# NOTE Modified `space_opt` function.


def mod_space_opt(
    *,
    space,
    dryness_method,
    fuel_build_up_method,
    include_temperature,
    discrete_params,
    defaults=None,
    basinhopping_options=None,
    minimizer_options=None,
    mode="basinhopping",
    x0=None,
):
    """Optimisation of the continuous (float) part of a given `space`."""
    to_optimise = gen_to_optimise(
        fail_func=fail_func,
        success_func=success_func,
        # Init (data) params.
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=include_temperature,
        _uncached_data=False,
        **discrete_params,
    )

    defaults_dict = defaults if defaults is not None else {}

    def to_optimise_with_discrete(x):
        return to_optimise(
            **space.inv_map_float_to_0_1(dict(zip(space.continuous_param_names, x))),
            **defaults_dict,
        )

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

    minimizer_options_dict = minimizer_options if minimizer_options is not None else {}
    basinhopping_options_dict = (
        basinhopping_options if basinhopping_options is not None else {}
    )

    if x0 is None:
        x0 = space.continuous_x0_mid

    if mode == "basinhopping":
        res = basinhopping(
            to_optimise_with_discrete,
            x0=x0,
            seed=0,
            callback=basinhopping_callback,
            take_step=BoundedSteps(
                stepsize=0.3, rng=np.random.default_rng(0), verbose=True
            ),
            **{
                "disp": True,
                "minimizer_kwargs": dict(
                    method="L-BFGS-B",
                    jac=None,
                    bounds=[(0, 1)] * len(space.continuous_param_names),
                    options={
                        "maxiter": 60,
                        "ftol": 1e-5,
                        "eps": 1e-3,
                        **minimizer_options_dict,
                    },
                ),
                "T": 0.05,
                "niter": 100,
                "niter_success": 15,
                **basinhopping_options_dict,
            },
        )
    elif mode == "minimize":
        res = minimize(
            to_optimise_with_discrete,
            x0=x0,
            method="L-BFGS-B",
            jac=None,
            bounds=[(0, 1)] * len(space.continuous_param_names),
            options={
                "maxiter": 60,
                "ftol": 1e-5,
                "eps": 1e-3,
                **minimizer_options_dict,
            },
        )
    else:
        raise ValueError

    return res


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("opt_mode", ["minimize", "basinhopping"])
def test_perf(model_params, opt_mode, seed):
    params = next(iter(model_params.values()))

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

    # Investigate some of the later configurations.
    losses = []
    x0_dict = None
    prev_spec = None
    prev_constants = None

    for configuration in (configurations[i] for i in range(-9, 0, 1)):
        space_spec, constants = configuration_to_hyperopt_space_spec(configuration)
        space = HyperoptSpace({**base_spec, **space_spec})

        if x0_dict is not None:
            x0 = get_next_x0(
                new_space=space,
                x0_dict=x0_dict,
                prev_spec=prev_spec,
                prev_constants=prev_constants,
            )
        else:
            x0 = None

        res = mod_space_opt(
            space=space,
            dryness_method=dryness_method,
            fuel_build_up_method=fuel_build_up_method,
            include_temperature=include_temperature,
            discrete_params=discrete_params,
            defaults={**defaults, **constants},
            minimizer_options=dict(maxiter=500),
            basinhopping_options=dict(niter_success=10),
            mode=opt_mode,
            x0=x0,
        )
        x0_dict = {key: val for key, val in zip(space.continuous_param_names, res.x)}

        prev_spec = space_spec
        prev_constants = constants
        losses.append(res.fun)

    assert np.all(np.diff(losses) < 1e-6)
