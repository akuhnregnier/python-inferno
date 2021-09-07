# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from python_inferno.configuration import npft
from python_inferno.data import get_climatological_dry_bal, timestep

if __name__ == "__main__":
    dry_bal = get_climatological_dry_bal(
        # XXX TODO
        # vpd_f should be ~10 - 5000x ?? bigger than rain_f
        # rain_f=[0.31] * npft,
        # vpd_f=[15] * npft,
        rain_f=[0.3] * npft,
        vpd_f=[40] * npft,
    )

    plt.figure()
    xs = np.arange(0, dry_bal.shape[0]) * timestep / (60 * 60 * 24)  # Convert to days.
    for p in range(npft):
        plt.plot(xs, dry_bal[:, p, 0], label=p)
    plt.xlabel("days")
    plt.legend()
    plt.title("'dry_bal' variable (influenced by 'vpd_f' & 'rain_f')")
