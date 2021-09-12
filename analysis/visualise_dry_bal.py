#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from python_inferno.data import get_climatological_grouped_dry_bal, timestep

if __name__ == "__main__":
    dry_bal = get_climatological_grouped_dry_bal(
        rain_f=0.5,
        vpd_f=19,
    )

    # Select a single PFT.
    dry_bal = dry_bal[:, 0]

    rows = 6
    cols = 6
    N = rows * cols

    xs = np.arange(0, dry_bal.shape[0]) * timestep / (60 * 60 * 24)  # Convert to days.

    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)

    for ax, l in zip(
        axes.ravel(),
        np.random.default_rng(0).choice(dry_bal.shape[1], size=(N,), replace=False),
    ):
        ax.plot(xs, dry_bal[:, l])
        ax.set_title(l)

    margin = 0.05
    plt.subplots_adjust(left=margin, right=1 - margin, top=1 - margin, bottom=margin)

    # plt.figure()
    # # plt.hist(
    # #     dry_bal[:, 0].ravel(), bins=np.linspace(-1, 1, 10), density=True, label="0"
    # # )
    # plt.hist(
    #     dry_bal[:, :].ravel(),
    #     bins=np.linspace(-1, 1, 10),
    #     density=True,
    #     label="all",
    #     alpha=0.5,
    # )
    # plt.legend(loc="best")

    plt.show()
