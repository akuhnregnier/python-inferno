# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize
import scipy.stats
from numpy.linalg import cholesky
from scipy.interpolate import interp1d
from scipy.special import ndtr
from scipy.stats import gaussian_kde, norm


def gen_correlated_samples(*, N, n_params, F_invs, R, rng=None):
    """

    N - Number of samples.
    n_params - Number of parameters.
    F_invs - Inverse CDF functions for each parameter.
    R - target Spearman rank correlation matrix.

    Based on Mara et al. 2015, Iman and Conover 1982.

    """
    if rng is None:
        rng = np.random.default_rng()

    gen_norm = norm()
    gen_norm.random_state = rng

    Z_nc = gen_norm.rvs(size=(N, n_params))
    C_z, p_C_z = scipy.stats.spearmanr(Z_nc)
    if n_params == 2:
        C_z = np.array(
            [
                [1, C_z],
                [C_z, 1],
            ]
        )

    L = cholesky(R)
    Q = cholesky(C_z)

    Z_c = np.dot(Z_nc, np.dot(np.linalg.inv(Q).T, L.T))

    norm_cdf_z = norm().cdf(Z_c)

    X = np.empty_like(Z_nc)
    for i, F_inv in enumerate(F_invs):
        X[:, i] = F_inv(norm_cdf_z[:, i])

    return X


def inverse_cdf_from_kde(y):
    """Estimate inverse CDF using Gaussian KDE.

    y - input 1D data array

    """
    kde = gaussian_kde(y)

    xs = np.linspace(np.min(y), np.max(y), int(1e4))

    # Estimated CDF.
    cdfs = [ndtr(np.ravel(x - kde.dataset) / kde.factor).mean() for x in xs]

    uniq_xs = []
    uniq_cdfs = []

    last_cdf = None

    for x, cdf in zip(xs, cdfs):
        if last_cdf is not None:
            # Check.
            if not np.isclose(last_cdf, cdf, rtol=0, atol=1e-10):
                uniq_xs.append(x)
                uniq_cdfs.append(cdf)
                last_cdf = cdf
        else:
            uniq_xs.append(x)
            uniq_cdfs.append(cdf)
            last_cdf = cdf

    return interp1d(
        uniq_cdfs, uniq_xs, kind="cubic", bounds_error=False, fill_value="extrapolate"
    )
