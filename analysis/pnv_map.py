#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from wildfires.analysis import cube_plotting
from wildfires.utils import get_centres

from python_inferno.plotting import use_style
from python_inferno.pnv import get_pnv_mega_regions, pnv_csv_file

if __name__ == "__main__":
    use_style()

    mega_pnv_cube = get_pnv_mega_regions()

    region_codes = np.array(list(mega_pnv_cube.attributes["regions"]))
    bounds = np.array([*(region_codes - 0.5), region_codes[-1] + 0.5])

    # NOTE See also colors.from_levels_and_colors for combined colormap and norm
    # generation!
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=bounds.size - 1)

    cmap = "tab10"

    fig, ax = plt.subplots(
        1, 1, figsize=(8, 4), subplot_kw=dict(projection=ccrs.Robinson())
    )
    cube_plotting(
        mega_pnv_cube,
        fig=fig,
        ax=ax,
        title="",
        colorbar_kwargs=False,
        cmap=cmap,
        norm=norm,
    )

    # Add colorbar.

    height = 0.5
    width = 0.012

    box = ax.get_position()

    x0 = 0.91
    y0 = (box.ymin + box.ymax) / 2 - height / 2

    cb = mpl.colorbar.ColorbarBase(
        fig.add_axes([x0, y0, width, height]),
        cmap=plt.get_cmap(cmap),
        norm=norm,
        extend="neither",
        ticks=get_centres(bounds),
        spacing="uniform",
        orientation="vertical",
        label="MEGA PNV biome",
    )
    # y-axis for vertical cbar.
    cb.ax.set_yticklabels(list(mega_pnv_cube.attributes["regions_codes"]))

    fig.savefig(Path("~/tmp/mega_biomes.png").expanduser())
    plt.close(fig)

    pnv_df = pd.read_csv(pnv_csv_file, header=0, index_col=0)
    print(pnv_df)

    print("\nMega PNV\n")
    print(
        pd.Series(
            {
                mega_pnv_cube.attributes["regions"][number]: np.sum(
                    mega_pnv_cube.data == number
                )
                for number in mega_pnv_cube.attributes["regions"]
            }
        )
    )
