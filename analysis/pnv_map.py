#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wildfires.analysis import cube_plotting

from python_inferno.pnv import get_pnv_mega_regions, pnv_csv_file

if __name__ == "__main__":
    mpl.rc_file(Path(__file__).absolute().parent / "matplotlibrc")

    mega_pnv_cube = get_pnv_mega_regions()

    cube_plotting(mega_pnv_cube, fig=plt.figure(figsize=(8, 4)), title="Mega PNV")

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

    plt.show()
