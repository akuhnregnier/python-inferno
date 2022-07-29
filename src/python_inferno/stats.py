# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize
import scipy.stats
from loguru import logger
from numpy.linalg import cholesky
from scipy.interpolate import interp1d
from scipy.special import ndtr
from scipy.stats import gaussian_kde, norm
from tqdm import tqdm

from .cache import cache, mark_dependency


@mark_dependency
@cache
def gen_correlated_samples(*, N, F_invs, R, rng=None):
    """

    N - Number of samples.
    F_invs - Inverse CDF functions for each parameter.
    R - target Spearman rank correlation matrix.

    Based on Mara et al. 2015, Iman and Conover 1982.

    """
    assert len(F_invs) == R.shape[0] == R.shape[1]
    n_params = R.shape[0]
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


@mark_dependency
@cache
def inverse_cdf_from_kde(y, steps=int(1e4)):
    """Estimate inverse CDF using Gaussian KDE.

    y - input 1D data array

    """
    kde = gaussian_kde(y)

    xs = np.linspace(np.min(y), np.max(y), steps)

    # Estimated CDF.
    cdfs = [
        ndtr(np.ravel(x - kde.dataset) / kde.factor).mean()
        for x in tqdm(xs, desc="Estimating CDF")
    ]

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

    # NOTE Scale estimated inverse CDF slightly in order to ensure bounds are [0, 1].
    low_err = uniq_cdfs[0]
    upp_err = 1 - uniq_cdfs[-1]
    if low_err > 1e-4 or upp_err > 1e-4:
        logger.warning(f"CDF errs: {low_err:0.2e}, {upp_err:0.2e}.")
    uniq_cdfs = np.asarray(uniq_cdfs)
    uniq_cdfs -= np.min(uniq_cdfs)
    uniq_cdfs /= np.max(uniq_cdfs)

    return interp1d(
        uniq_cdfs,
        uniq_xs,
        kind="cubic",
        bounds_error=True,
    )


@mark_dependency
@cache(dependencies=[inverse_cdf_from_kde, gen_correlated_samples])
def gen_correlated_samples_from_chains(*, N, chains, icdf_steps=int(1e4), rng=None):
    if rng is None:
        rng = np.random.default_rng()

    corr, corr_p = scipy.stats.spearmanr(chains)

    F_invs = [
        inverse_cdf_from_kde(chain, steps=icdf_steps)
        for chain in tqdm(chains.T, desc="Approximating inverse CDFs")
    ]

    return gen_correlated_samples(N=N, F_invs=F_invs, R=corr, rng=rng)
