# -*- coding: utf-8 -*-
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spotpy
from tqdm import tqdm

from .cache import cache, mark_dependency
from .hyperopt import get_space_template
from .mcmc import get_sse_func, iter_opt_methods

assert spotpy.__version__ == "1.5.16.1", spotpy.__version__


@mark_dependency
@cache(dependencies=[iter_opt_methods, get_sse_func, get_space_template])
def spotpy_dream(iter_opt_index=0, N=int(5e5), c=0.1, step=0.5, beta=0.05):
    opt_data = next(
        iter_opt_methods(
            indices=(
                iter_opt_index,
                iter_opt_index + 1,
            ),
            release_gpu_model=True,
        )
    )

    opt_data["score_model"]
    space = opt_data["space"]
    x0_0_1 = opt_data["x0_0_1"]

    sse_func = opt_data["sse_func"]

    class spotpy_setup(object):
        def __init__(self, x0_0_1, sse_func):
            self.params = [
                spotpy.parameter.Uniform(name, 0, 1, step, x0, 0, 1)
                for name, x0 in x0_0_1.items()
            ]
            self.sse_func = sse_func

        def parameters(self):
            return spotpy.parameter.generate(self.params)

        def simulation(self, vector):
            simulations = [self.sse_func(vector)]
            return simulations

        def evaluation(self):
            observations = [0]
            return observations

        def objectivefunction(self, simulation, evaluation):
            objectivefunction = -spotpy.objectivefunctions.rmse(evaluation, simulation)
            return objectivefunction

    spot_setup = spotpy_setup(x0_0_1, sse_func)

    sampler = spotpy.algorithms.dream(
        spot_setup,
        parallel="seq",
        random_state=0,
    )
    # 1e5 in ~20 mins
    r_hat = sampler.sample(N, c=c, beta=beta)

    results = sampler.getdata()

    results_df = pd.DataFrame(results)

    return dict(r_hat=r_hat, results_df=results_df, space=space)


def plot_combined(chain_groups, name, n_samples, save_dir):
    plt.ioff()
    fig, ax = plt.subplots(figsize=(35, 18))
    for (chain_id, (float_chain_id, data)) in enumerate(chain_groups):
        ax.plot(data[name], c=f"C{chain_id}", linestyle="", marker="o", alpha=0.4)

    name = name.lstrip("par")

    ax.set_ylabel(name)
    ax.set_xlabel("iterations")
    ax.grid(alpha=0.5, linestyle="--", color=tuple([0.8] * 3))

    fig.savefig(save_dir / name, bbox_inches="tight")
    plt.close(fig)


def plot_spotpy_results_df(*, results_df, save_dir):
    # Plot loss and parameters.
    parameter_cols = ["log_simulation_0"] + [
        col for col in results_df if col.startswith("par")
    ]
    results_df["log_simulation_0"] = np.log(results_df["simulation_0"])
    del results_df["simulation_0"]

    # Parallel plotting.
    executor = ProcessPoolExecutor(max_workers=10)
    futures = []

    for name in tqdm(parameter_cols, desc="Saving plots", disable=True):
        param_df = results_df[["chain", name]]
        chain_groups = param_df.groupby("chain", as_index=False)

        n_samples = param_df.shape[0]

        futures.append(
            executor.submit(plot_combined, chain_groups, name, n_samples, save_dir)
        )

    for f in tqdm(as_completed(futures), total=len(futures), desc="Saving plots"):
        # Get result here to be notified of any exceptions.
        f.result()

    executor.shutdown()


@mark_dependency
def get_cached_mcmc_chains(*, method_index):
    mcmc_kwargs = dict(
        iter_opt_index=method_index,
        # 1e5 - 15 mins with beta=0.05
        # 2e5 - 50 mins with beta=0.05 - due to decreasing acceptance rate over time!
        N=int(2e5),
        beta=0.05,
    )
    assert spotpy_dream.check_in_store(**mcmc_kwargs), str(mcmc_kwargs)
    dream_results = spotpy_dream(**mcmc_kwargs)
    results_df = dream_results["results_df"]
    space = dream_results["space"]

    # Analysis of results.
    names = space.continuous_param_names

    # Generate array of chain values, transform back to original ranges.
    chains = np.hstack(
        [
            space.inv_map_float_to_0_1({name: np.asarray(results_df[f"par{name}"])})[
                name
            ].reshape(-1, 1)
            for name in names
        ]
    )

    return dict(names=names, chains=chains)
