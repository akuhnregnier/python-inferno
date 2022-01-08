#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from wildfires.analysis import cube_plotting
from wildfires.data import GFEDv4
from wildfires.utils import ensure_datetime

if __name__ == "__main__":
    gfed = GFEDv4()
    gfed_cube = gfed.cube

    time_coord = gfed_cube.coord("time")
    dates = [
        ensure_datetime(time_coord.cell(i).point) for i in range(time_coord.shape[0])
    ]
    lats = gfed_cube.coord("latitude").points

    plot_data = {}
    starti = 0
    stepsize = 7
    maxi = 70
    for _i in range(starti, maxi, stepsize):
        end_index = lats.shape[0] - 1 - _i
        start_index = end_index - stepsize
        mean_lat = np.mean(lats[start_index:end_index])
        data = gfed_cube[:, start_index:end_index].data
        # Mean BA across all longitudes at given latitude(s), over time.
        plot_data[mean_lat] = np.mean(data, axis=(1, 2))

    # Index of the lowest latitude.
    max_index = start_index

    # Plot mean BA cross sections.
    cmap = plt.get_cmap("turbo")
    plt.figure()
    for (i, (lat, data)) in enumerate(plot_data.items()):
        plt.plot(
            dates,
            data,
            label=lat,
            c=cmap(float(i / len(plot_data))),
            ls="-" if np.any(data > 0) else "--",
        )
    plt.legend()
    plt.ylabel("BA")

    cube_plotting(gfed_cube[:, max_index:, :], log=True)
