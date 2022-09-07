# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spotpy
from scipy.stats import norm
from spotpy.examples.spot_setup_standardnormal import spot_setup

assert spotpy.__version__ == "1.5.16.1", spotpy.__version__


if __name__ == "__main__":
    mean = 0.5
    std = 0.5

    sampler = spotpy.algorithms.dream(
        spot_setup(mean=mean, std=std),
        parallel="seq",
        random_state=0,
    )

    r_hat = sampler.sample(
        repetitions=10000,
        c=0.1,
        beta=1.0,
        nChains=10,
        maxTime=1000,
        acc_eps=0.3,
    )
    r_hat = np.asarray(r_hat)

    results = sampler.getdata()

    results_df = pd.DataFrame(results)
    results_df["chain"] = results_df["chain"].astype("int")

    fig, axes = plt.subplots(2, 2)

    ax = axes.ravel()[0]
    ax.plot(r_hat[:, 0])
    ax.set_title("r-hat")

    ax = axes.ravel()[1]
    chain_groups = results_df.groupby("chain", as_index=False)
    for chain_id, data in chain_groups:
        ax.plot(data["parx"], alpha=0.4)
    ax.set_title("x")

    ax = axes.ravel()[2]
    chain_groups = results_df.groupby("chain", as_index=False)
    for chain_id, data in chain_groups:
        ax.plot(-data["simulation_0"], alpha=0.4)
    ax.set_title("rmse")

    ax = axes.ravel()[3]
    ax.hist(results_df["parx"], bins=100, density=True)
    xs = np.linspace(np.min(results_df["parx"]), np.max(results_df["parx"]), 100)
    ax.plot(xs, norm(loc=mean, scale=std).pdf(xs))
    ax.set_title("x")
