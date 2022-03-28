#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from python_inferno.data import calc_litter_pool

if __name__ == "__main__":
    litter_tcs = np.geomspace(0.5e-10, 1e-7, 8)
    leaf_fs = np.geomspace(1e-10, 10, 8)

    stds = np.zeros((litter_tcs.size, leaf_fs.size))

    for i, litter_tc in enumerate(tqdm(litter_tcs)):
        for j, leaf_f in enumerate(leaf_fs):
            litter_pool = calc_litter_pool(
                litter_tc=litter_tc, leaf_f=leaf_f, verbose=False, Nt=100
            )
            stds[i, j] = np.mean(np.std(litter_pool, axis=0))

    plt.figure()
    plt.imshow(stds.T)
    plt.xticks(
        ticks=np.arange(litter_tcs.size),
        labels=[format(x, "0.1e") for x in litter_tcs],
        rotation=45,
    )
    plt.xlabel("litter_tc")
    plt.yticks(
        ticks=np.arange(leaf_fs.size),
        labels=[format(x, "0.1e") for x in leaf_fs],
    )
    plt.ylabel("leaf_fs")
    plt.colorbar()
    plt.show()

    ####

    # land_index = 0
    # pft_index = 0
    # Nt = 2000

    # litter_tcs = np.geomspace(1e-8, 1e-4, 7)
    # leaf_fs = np.geomspace(1e-5, 1e-1, 7)

    # data = np.zeros((litter_tcs.size, leaf_fs.size, Nt))

    # for i, litter_tc in enumerate(tqdm(litter_tcs)):
    #     for j, leaf_f in enumerate(leaf_fs):
    #         data[i, j] = calc_litter_pool(
    #             litter_tc=litter_tc, leaf_f=leaf_f, verbose=False, Nt=Nt
    #         )[:, pft_index, land_index]

    # fig, axes = plt.subplots(
    #     nrows=litter_tcs.size,
    #     ncols=leaf_fs.size,
    #     sharex=True,
    #     sharey=True,
    # )

    # for i, litter_tc in enumerate(litter_tcs):
    #     for j, leaf_f in enumerate(leaf_fs):
    #         ax = axes[i, j]
    #         ax.plot(data[i, j])

    #         if i == 0:
    #             ax.set_title("leaf_f=" + format(leaf_f, "0.1e"))

    #         if j == 0:
    #             ax.set_ylabel(
    #                 "litter_tc=" + format(litter_tc, "0.1e"), rotation=0, labelpad=40,
    #                 fontsize=8
    #             )

    # plt.show()
    #
    ######

    # land_index = 0
    # pft_index = 0

    # plt.figure()
    # plt.plot(
    #     calc_litter_pool(litter_tc=1e-8, leaf_f=1e-3, verbose=False, Nt=None)[
    #         :, pft_index, land_index
    #     ]
    # )
    # plt.show()
