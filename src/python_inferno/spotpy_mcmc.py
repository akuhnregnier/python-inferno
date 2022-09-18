# -*- coding: utf-8 -*-
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from operator import itemgetter
from string import ascii_lowercase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spotpy
from tqdm import tqdm

from .cache import cache, mark_dependency
from .configuration import pft_group_names
from .hyperopt import get_space_template
from .mcmc import get_loss_func, iter_opt_methods
from .plotting import get_plot_name_map_total, use_style

assert spotpy.__version__ == "1.5.16.1", spotpy.__version__


@mark_dependency
@cache(
    dependencies=[
        get_space_template,
        get_loss_func,
        iter_opt_methods,
        spotpy.algorithms._RunStatistic.__call__,
        spotpy.algorithms._RunStatistic.__init__,
        spotpy.algorithms._RunStatistic.grid,
        spotpy.algorithms._RunStatistic.maximizer,
        spotpy.algorithms._RunStatistic.minimizer,
        spotpy.algorithms._algorithm.__init__,
        spotpy.algorithms._algorithm._algorithm__is_list_type,
        spotpy.algorithms._algorithm._init_database,
        spotpy.algorithms._algorithm.final_call,
        spotpy.algorithms._algorithm.get_parameters,
        spotpy.algorithms._algorithm.getdata,
        spotpy.algorithms._algorithm.getfitness,
        spotpy.algorithms._algorithm.postprocessing,
        spotpy.algorithms._algorithm.read_breakdata,
        spotpy.algorithms._algorithm.save,
        spotpy.algorithms._algorithm.set_repetition,
        spotpy.algorithms._algorithm.simulate,
        spotpy.algorithms._algorithm.update_params,
        spotpy.algorithms._algorithm.write_breakdata,
        spotpy.algorithms.dream.__init__,
        spotpy.algorithms.dream._get_gamma,
        spotpy.algorithms.dream.check,
        spotpy.algorithms.dream.check_par_validity_bound,
        spotpy.algorithms.dream.check_par_validity_reflect,
        spotpy.algorithms.dream.get_new_proposal_vector,
        spotpy.algorithms.dream.get_other_random_chains,
        spotpy.algorithms.dream.get_r_hat,
        spotpy.algorithms.dream.get_regular_startingpoint,
        spotpy.algorithms.dream.sample,
        spotpy.algorithms.dream.update_last_half_like,
        spotpy.algorithms.dream.update_last_half_param_data_const_n,
        spotpy.algorithms.dream.update_last_half_param_data_inc_n,
        spotpy.algorithms.dream.update_mcmc_status,
        spotpy.algorithms.dream.update_r_hat_data,
    ]
)
def spotpy_dream(
    iter_opt_index=0,
    N=int(5e5),
    c=0.1,
    step=0.5,
    beta=0.05,
    nChains=7,
    maxTime=np.inf,
    acc_eps_delta=0.05,
):
    N = int(N)

    opt_data = next(
        iter_opt_methods(
            indices=(
                iter_opt_index,
                iter_opt_index + 1,
            ),
            release_gpu_model=True,
        )
    )
    space, x0_0_1, loss_func, lowest_model_loss = itemgetter(
        "space", "x0_0_1", "loss_func", "lowest_model_loss"
    )(opt_data)

    class spotpy_setup(object):
        def __init__(self, x0_0_1, loss_func):
            self.params = [
                spotpy.parameter.Uniform(name, 0, 1, step, x0, 0, 1)
                for name, x0 in x0_0_1.items()
            ]
            self.loss_func = loss_func

        def parameters(self):
            return spotpy.parameter.generate(self.params)

        def simulation(self, vector):
            simulations = [self.loss_func(vector)]
            return simulations

        def evaluation(self):
            observations = [0]
            return observations

        def objectivefunction(self, simulation, evaluation):
            objectivefunction = -spotpy.objectivefunctions.rmse(evaluation, simulation)
            return objectivefunction

    sampler = spotpy.algorithms.dream(
        spotpy_setup(x0_0_1, loss_func),
        parallel="seq",
        random_state=0,
    )

    r_hat = sampler.sample(
        repetitions=N,
        c=c,
        beta=beta,
        nChains=nChains,
        maxTime=maxTime,
        acc_eps=lowest_model_loss + acc_eps_delta,
        runs_after_convergence=int(1e4),
        burnInNSamples=N // 2,
    )

    results = sampler.getdata()

    results_df = pd.DataFrame(results)

    return dict(r_hat=r_hat, results_df=results_df, space=space)


def plot_combined(param_df, name, n_samples, save_dir, max_n_plot=int(1e4)):
    """

    max_n_plot (int): Maximum number of points per chain. Will be subsampled if
        needed.

    """
    use_style()

    plt.ioff()
    fig, axes = plt.subplots(2, 1, figsize=(5, 3.5))

    # Plot chains.

    ax = axes[0]

    for chain_id, data in param_df.groupby("chain", as_index=False):
        xs = np.arange(len(data[name]))
        ys = data[name]
        if xs.size > max_n_plot:
            factor = math.ceil(xs.size / max_n_plot)
            xs = xs[::factor]
            ys = ys[::factor]
        ax.plot(xs, ys, c=f"C{chain_id}", linestyle="-")

    name = name.lstrip("par")

    if name.endswith("2") or name.endswith("3"):
        param_number = int(name[-1]) - 1
        par_name = name[:-1]
    else:
        par_name = name
        param_number = 0

    pft_group_name = pft_group_names[param_number]
    if par_name != "log_simulation_0":
        label = f"{get_plot_name_map_total()[par_name]} {pft_group_name}"
    else:
        label = par_name

    ax.set_ylabel(label)
    ax.set_xlabel("Iterations")
    ax.grid(alpha=0.5, linestyle="--", color=tuple([0.8] * 3))

    # Plot histogram.

    data_col = [col for col in param_df.columns if col != "chain"][0]
    hist_data = np.asarray(param_df[data_col])
    assert hist_data.ndim == 1
    hist_data = hist_data[hist_data.size // 2 :]  # Only use last 1/2.
    axes[1].hist(hist_data, bins=100, density=True)
    axes[1].set_xlabel(label)
    axes[1].set_ylabel("Density")

    for ax, label in zip(axes.ravel(), ascii_lowercase):
        ax.text(-0.01, 1.05, f"({label})", transform=ax.transAxes)

    # Save.

    plt.tight_layout()
    fig.savefig(save_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_spotpy_results_df(*, results_df, save_dir):
    # Plot loss and parameters.
    parameter_cols = ["log_simulation_0"] + [
        col for col in results_df if col.startswith("par")
    ]
    results_df["log_simulation_0"] = np.log(results_df["simulation_0"])
    del results_df["simulation_0"]

    results_df["chain"] = results_df["chain"].astype("int")

    # Parallel plotting.
    executor = ProcessPoolExecutor(max_workers=10)

    futures = []

    for name in tqdm(parameter_cols, desc="Saving plots", disable=True):
        param_df = results_df[["chain", name]]

        n_samples = param_df.shape[0]

        futures.append(
            executor.submit(plot_combined, param_df, name, n_samples, save_dir)
        )

    for f in tqdm(as_completed(futures), total=len(futures), desc="Saving plots"):
        # Get result here to be notified of any exceptions.
        f.result()

    executor.shutdown()


@mark_dependency
def get_cached_mcmc_chains(*, method_index):
    mcmc_kwargs = dict(
        iter_opt_index=method_index,
        N=int(2e5),
        beta=1.0,
        nChains=50,
        c=0.1,
        maxTime=5 * 60 * 60,
        acc_eps_delta=0.002,
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
