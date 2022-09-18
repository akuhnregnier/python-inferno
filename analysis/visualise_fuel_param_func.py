#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from python_inferno.inferno import sigmoid
from python_inferno.plotting import use_style

if __name__ == "__main__":
    use_style()

    factors = [-1, 0.5, 3]
    centres = [0]
    shapes = [0.2, 1, 4]

    nplots = len(factors) * len(centres) * len(shapes)
    nrows = math.floor(nplots**0.5)
    ncols = math.ceil(nplots / nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=np.array([1, 1])
        + np.array([2.5, 1.5]) * np.array([ncols, nrows]) / 1.5,
        sharex=True,
        sharey=True,
    )

    x = np.linspace(-10, 10, 1000)

    i = 0
    for factor in factors:
        for centre in centres:
            for shape in shapes:
                ax = axes.ravel()[i]
                ax.plot(x, sigmoid(x, factor, centre, shape))
                ax.set_title(
                    rf"$\alpha={factor:0.1f}\ \beta={centre:0.1f}\ \gamma={shape:0.1f}$"
                )
                i += 1

    plt.tight_layout()
    fig.savefig(Path("~/tmp/sigmoid_curves.pdf").expanduser())
    plt.close(fig)
