#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats
from mcmcplot import mcmcplot as mcp
from pymcmcstat.MCMC import MCMC

from python_inferno.stats import gen_correlated_samples, inverse_cdf_from_kde


def model(x, theta):
    a, b = theta
    return a * b * x


def ssfunc(theta, data):
    x_obs, y_obs = data.xdata[0], data.ydata[0]
    y_pred = model(x_obs, theta)

    # SSE
    return np.sum((y_obs - y_pred) ** 2)


if __name__ == "__main__":
    mcstat = MCMC(rngseed=0)

    x_obs = np.array([1, 2, 3, 4])[:, None]
    y_obs = np.array([2.1, 3.2, 4.2, 5.5])[:, None]

    mcstat.data.add_data_set(x_obs, y_obs)

    mcstat.parameters.add_model_parameter(name="a", theta0=1, minimum=0, maximum=5)
    mcstat.parameters.add_model_parameter(name="b", theta0=1, minimum=0, maximum=5)

    mcstat.simulation_options.define_simulation_options(
        nsimu=int(1e4),
        updatesigma=1,
    )

    mcstat.model_settings.define_model_settings(sos_function=ssfunc, sigma2=0.01**2)

    mcstat.run_simulation()

    # Retrieve results object.
    results = mcstat.simulation_results.results
    chain = results["chain"]
    s2chain = results["s2chain"]
    names = results["names"]  # parameter names

    # Print some results.
    mcstat.chainstats(chain, results)

    point_est_theta = np.mean(chain, axis=0)
    x_plot = np.linspace(x_obs.min(), x_obs.max(), 100)

    plt.figure()
    plt.plot(x_obs.ravel(), y_obs.ravel(), linestyle="", marker="x", label="data")
    plt.plot(x_plot, model(x_plot, point_est_theta), linestyle="-", label="model")
    plt.legend()

    # plot chain panel
    mcp.plot_chain_panel(chain, names)
    # The |'pairs'| options makes pairwise scatterplots of the columns of
    # the |chain|.
    pwfig = mcp.plot_pairwise_correlation_panel(
        chain, names, settings=dict(fig=dict(figsize=(4, 4)))
    )

    f, settings = mcp.plot_density_panel(
        chains=chain, hist_on=True, return_settings=True
    )

    # Recreate joint distribution from the marginals.

    corr, corr_p = scipy.stats.spearmanr(chain)

    new_data = gen_correlated_samples(
        N=int(1e3),
        n_params=chain.shape[1],
        F_invs=[inverse_cdf_from_kde(chain[:, i]) for i in range(chain.shape[1])],
        R=np.array([[1, corr], [corr, 1]]),
        rng=np.random.default_rng(0),
    )

    pwfig = mcp.plot_pairwise_correlation_panel(
        new_data, names, settings=dict(fig=dict(figsize=(4, 4)))
    )

    f, settings = mcp.plot_density_panel(
        chains=new_data, hist_on=True, return_settings=True
    )

    plt.show()
