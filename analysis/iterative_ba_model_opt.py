#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from multiprocessing import Process, Queue
from pathlib import Path
from pprint import pprint

import numpy as np
from loguru import logger
from tqdm import tqdm

from python_inferno.cache import IN_STORE, NotCachedError
from python_inferno.hyperopt import HyperoptSpace, get_space_template
from python_inferno.iter_opt import (
    IGNORED,
    configuration_to_hyperopt_space_spec,
    next_configurations_iter,
)
from python_inferno.model_params import get_model_params
from python_inferno.space import generate_space_spec
from python_inferno.space_opt import space_opt

ALWAYS_OPTIMISED = {
    "overall_scale",
}


def mp_space_opt(*, q, **kwargs):
    q.put(space_opt(**kwargs))


def iterative_ba_model_opt(*, params):
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

    q = Queue()

    steps_prog = tqdm(desc="Steps", position=0)

    results = {}
    init_n_params = 1  # TODO - This initial value should depends on `base_spec`.

    n_configs = None
    steps = 0

    while n_configs != 0:
        n_configs = 0

        local_best_config = None
        local_best_loss = np.inf

        for (configuration, n_new) in tqdm(
            list(next_configurations_iter(start_config)),
            desc="Step-configs",
            position=1,
        ):
            n_params = init_n_params + n_new

            space_spec, constants = configuration_to_hyperopt_space_spec(configuration)
            space = HyperoptSpace({**base_spec, **space_spec})

            opt_kwargs = dict(
                space=space,
                dryness_method=dryness_method,
                fuel_build_up_method=fuel_build_up_method,
                include_temperature=include_temperature,
                discrete_params=discrete_params,
                opt_record_dir="test",
                defaults={**defaults, **constants},
                # XXX
                minimizer_options=dict(maxiter=30),
                basinhopping_options=dict(niter_success=5),
                verbose=False,
                _uncached_data=False,
            )

            is_cached = False
            try:
                if space_opt.check_in_store(**opt_kwargs) is IN_STORE:
                    is_cached = True
            except NotCachedError:
                pass

            if is_cached:
                loss = space_opt(**opt_kwargs)
            else:
                # Avoid memory leaks by running each trial in a new process.
                p = Process(target=mp_space_opt, kwargs={"q": q, **opt_kwargs})
                p.start()
                loss = q.get()
                p.join()

            logger.info(f"loss: {loss}")
            if loss < local_best_loss:
                logger.info(f"New best loss: {loss}.")
                local_best_loss = loss
                local_best_config = configuration
                best_n_params = n_params
                results[n_params] = (local_best_loss, local_best_config)
            n_configs += 1

            steps_prog.refresh()

        start_config = {**start_config, **local_best_config}
        init_n_params = best_n_params

        steps += 1
        steps_prog.update()

    q.close()
    q.join_thread()

    return results


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    df, method_iter = get_model_params(
        record_dir=Path(os.environ["EPHEMERAL"]) / "opt_record",
        progress=False,
        verbose=False,
    )

    for method_data in method_iter():
        params = method_data[5]
        results = iterative_ba_model_opt(params=params)
        print("Results")
        pprint(results)
