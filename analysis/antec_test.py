# -*- coding: utf-8 -*-
from pathlib import Path

import iris
import matplotlib.pyplot as plt

from python_inferno.utils import exponential_average, temporal_nearest_neighbour_interp

if __name__ == "__main__":
    # Load data.
    cubes = iris.load(
        str(
            Path(
                # "~/tmp/new-with-antec3/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUP0.Monthly.2000.nc"
                # "~/tmp/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUP0.Monthly.2009.nc"
                "~/tmp/new-with-antec4/JULES-ES.1p0.vn5.4.50.CRUJRA1.365.HYDE33.SPINUP0.Instant.2001.nc"
            ).expanduser()
        )
    )

    # Extract the necessary variables.
    fuel_build_up = cubes.extract_cube("fuel_build")
    fapar_diag_pft = cubes.extract_cube("fapar")

    # data_time_coord = fapar_diag_pft.coord('time')

    land_index = 2200

    plt.figure()
    plt.plot(fapar_diag_pft[:, 0, 0, land_index].data, label="fapar")
    plt.plot(fuel_build_up[:, 0, 0, land_index].data, label="fuel build up")
    plt.plot(
        exponential_average(
            temporal_nearest_neighbour_interp(
                fapar_diag_pft[:, 0, 0, land_index].data, 4
            ),
            4.6283007e-04,
        )[::4],
        label="antec",
    )
    plt.legend()
    plt.show()
