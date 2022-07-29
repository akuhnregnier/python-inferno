# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize
import scipy.stats
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


def test_corr_stats():
    """Generate simple data using MCMC, then recreate the resulting distributions."""
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

    corr, corr_p = scipy.stats.spearmanr(chain)

    new_data = gen_correlated_samples(
        N=int(1e3),
        F_invs=[inverse_cdf_from_kde(chain[:, i]) for i in range(chain.shape[1])],
        R=np.array([[1, corr], [corr, 1]]),
        rng=np.random.default_rng(0),
    )

    new_corr, new_corr_p = scipy.stats.spearmanr(new_data)

    assert np.isclose(new_corr, corr, atol=5e-4)
    assert new_corr_p < 1e-9
