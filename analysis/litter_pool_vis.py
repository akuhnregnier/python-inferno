#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from python_inferno.configuration import npft
from python_inferno.data import ProcMode, process_litter_pool

if __name__ == "__main__":
    litter_pool = process_litter_pool(
        litter_tc=tuple([1e-9] * npft),
        leaf_f=tuple([1e-3] * npft),
        proc_mode=ProcMode.SINGLE_PFT,
    )

    plt.figure()

    for l in [0, 1000, 7000]:
        plt.plot(litter_pool[:, l], label=l)

    plt.legend()
