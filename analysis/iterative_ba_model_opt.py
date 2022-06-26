#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from collections import defaultdict
from multiprocessing import Process, Queue
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm

from python_inferno.cache import IN_STORE, NotCachedError, cache
from python_inferno.hyperopt import HyperoptSpace, get_space_template
from python_inferno.iter_opt import (
    ALWAYS_OPTIMISED,
    IGNORED,
    any_match,
    configuration_to_hyperopt_space_spec,
    format_configurations,
    get_always_optimised,
    get_ignored,
    get_sigmoid_names,
    get_weight_sigmoid_names_map,
    match,
    next_configurations_iter,
    reorder,
)
from python_inferno.model_params import get_model_params
from python_inferno.space import generate_space_spec
from python_inferno.space_opt import space_opt


def mp_space_opt(*, q, **kwargs):
    q.put(space_opt(**kwargs))


@cache(
    dependencies=[
        any_match,
        configuration_to_hyperopt_space_spec,
        format_configurations,
        generate_space_spec,
        get_always_optimised,
        get_ignored,
        get_sigmoid_names,
        get_space_template,
        get_weight_sigmoid_names_map,
        match,
        next_configurations_iter,
        reorder,
        space_opt,
    ]
)
def iterative_ba_model_opt(
    *,
    params,
    maxiter=60,
    niter_success=15,
):
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

    steps = 0

    while True:
        local_best_config = defaultdict(lambda: None)
        local_best_loss = defaultdict(lambda: np.inf)

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
                minimizer_options=dict(maxiter=maxiter),
                basinhopping_options=dict(niter_success=niter_success),
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
            if loss < local_best_loss[n_params]:
                logger.info(f"New best loss: {loss}.")
                local_best_loss[n_params] = loss
                local_best_config[n_params] = configuration

                if n_params not in results:
                    # New `n_params`.
                    results[n_params] = (loss, configuration)
                else:
                    # Check the old loss.
                    if loss < results[n_params][0]:
                        # Only update if the new loss is lower.
                        results[n_params] = (loss, configuration)

            steps_prog.refresh()

        if not local_best_config:
            # No configurations were explored.
            break

        best_n_params = min(
            (loss, n_params) for (n_params, loss) in local_best_loss.items()
        )[1]

        start_config = {**start_config, **local_best_config[best_n_params]}
        init_n_params = best_n_params

        steps += 1
        steps_prog.update()

    q.close()
    q.join_thread()

    return results


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    df, method_iter = get_model_params(
        record_dir=Path(os.environ["EPHEMERAL"]) / "opt_record",
        progress=False,
        verbose=False,
    )

    plt.figure()

    for i, method_data in enumerate(method_iter()):
        full_opt_loss = method_data[4]
        params = method_data[5]
        method_name = method_data[6]

        results = iterative_ba_model_opt(params=params, maxiter=30, niter_success=5)

        n_params, losses = zip(
            *[(n_params, float(loss)) for (n_params, (loss, _)) in results.items()]
        )
        plt.plot(
            n_params, losses, marker="x", linestyle="", label=method_name, c=f"C{i}"
        )
        plt.hlines(full_opt_loss, 0, max(n_params), colors=f"C{i}")

    plt.legend()
    plt.show()
