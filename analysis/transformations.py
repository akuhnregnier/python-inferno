#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import partial
from pathlib import Path
from string import ascii_lowercase
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
from jules_output_analysis.data import regrid_to_n96e
from wildfires.data import GFEDv4
from wildfires.utils import get_land_mask, match_shape

from python_inferno.plotting import use_style


def arcsinh_func(data, factor):
    return np.arcsinh(factor * data)


if __name__ == "__main__":
    filterwarnings("ignore", ".*divide by zero.*")
    filterwarnings("ignore", ".*invalid units.*")
    filterwarnings("ignore", ".*may not be fully.*")
    filterwarnings("ignore", ".*axes.*")
    filterwarnings("ignore")
    use_style()

    gfed = GFEDv4()
    gfed = gfed.get_climatology_dataset(gfed.min_time, gfed.max_time)
    print(gfed)

    gfed.cube.data.mask |= match_shape(get_land_mask(), gfed.cube.shape)
    print("shape:", gfed.cube.shape)

    data = regrid_to_n96e(gfed.cube).data

    valid = data.data[~data.mask]

    xs = np.geomspace(1e-9, 1e10, 1000)

    plt.ioff()
    fig, axes = plt.subplots(3, 2, figsize=(6, 8))
    axes = axes.ravel()

    axes[0].plot(xs, arcsinh_func(xs, factor=1), label="arcsinh")
    axes[0].plot(xs, arcsinh_func(xs, factor=1e4), label="arcsinh(data x 1e4)")
    axes[0].plot(xs, arcsinh_func(xs, factor=1e8), label="arcsinh(data x 1e8)")
    axes[0].plot(xs, np.log(xs), label="log")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend()

    for (ax, (transformation, title, xlabel)) in zip(
        axes[1:],
        [
            (lambda data: data, "Raw BA Data (GFED4 Clim)", "raw BA"),
            (lambda data: np.log(data[data > 1e-9]), "Log (BA>1e-9)", "BA"),
            *(
                (
                    partial(arcsinh_func, factor=factor),
                    "Inverse Hyperbolic Sine",
                    f"(BA x {factor:0.1e})",
                )
                for factor in np.geomspace(1e4, 1e8, 3)
            ),
        ],
    ):
        ax.hist(transformation(valid), bins="auto")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)

    for ax, letter in zip(axes, ascii_lowercase):
        ax.text(-0.01, 1.05, f"({letter})", transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(Path("~/tmp/ba_transformations.pdf").expanduser())
    plt.close(fig)
