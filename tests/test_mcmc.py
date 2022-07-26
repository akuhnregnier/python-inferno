# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize
import scipy.stats
from pymcmcstat.MCMC import MCMC


def model(x, theta):
    a, b = theta
    return a * b * x


def ssfunc(theta, data):
    x_obs, y_obs = data.xdata[0], data.ydata[0]
    y_pred = model(x_obs, theta)

    # SSE
    return np.sum((y_obs - y_pred) ** 2)


def test_mcmc():
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

    point_est_theta = np.mean(chain, axis=0)
    std_theta = np.std(chain, axis=0)

    assert np.isclose(point_est_theta[0], 1.9, rtol=0.1)
    assert np.isclose(point_est_theta[1], 1.4, rtol=0.1)
    assert np.all(std_theta < 1.5)

    corr, corr_p = scipy.stats.spearmanr(chain)

    assert np.isclose(corr, -1, atol=0.1)
    assert corr_p < 1e-9
