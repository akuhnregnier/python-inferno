# -*- coding: utf-8 -*-
import iris
import matplotlib.pyplot as plt
import numpy as np
from iris.coord_categorisation import add_month_number, add_year
from jules_output_analysis.data import n96e_lats, n96e_lons
from jules_output_analysis.utils import convert_longitudes, get_1d_to_2d_indices
from wildfires.data import GFEDv4

from python_inferno.data import load_data

if __name__ == "__main__":
    (
        t1p5m_tile,
        q1p5m_tile,
        pstar,
        sthu_soilt_single,
        frac,
        c_soil_dpm_gb,
        c_soil_rpm_gb,
        canht,
        ls_rain,
        con_rain,
        fuel_build_up,
        fapar_diag_pft,
        jules_lats,
        jules_lons,
        gfed_ba_1d,
        obs_fapar_1d,
        jules_ba_gb,
        jules_time_coord,
        npp_pft,
        npp_gb,
    ) = load_data(N=None)

    # Define the ignition method (`ignition_method`).
    ignition_method = 1

    print(jules_time_coord.cell(-1).point)
    print(jules_time_coord.cell(0).point)
    cube = iris.cube.Cube(gfed_ba_1d, dim_coords_and_dims=[(jules_time_coord, 0)])

    print(cube)

    add_year(cube, "time")
    add_month_number(cube, "time")
    mon_avg = cube.aggregated_by(["year", "month_number"], iris.analysis.MEAN)

    gfed = GFEDv4()
    gfed.limit_months(jules_time_coord.cell(0).point, jules_time_coord.cell(-1).point)
    gfed.regrid(new_latitudes=n96e_lats, new_longitudes=n96e_lons, area_weighted=True)
    indices_1d_to_2d = get_1d_to_2d_indices(
        np.ma.getdata(jules_lats.points[0]),
        np.ma.getdata(convert_longitudes(jules_lons.points[0])),
        n96e_lats,
        n96e_lons,
    )
    mon_gfed_1d = np.ma.vstack(
        [data[indices_1d_to_2d][np.newaxis] for data in gfed.cube.data]
    )

    print(mon_gfed_1d.shape)
    assert mon_gfed_1d.shape == mon_avg.shape

    for i in np.argsort(np.max(np.abs(mon_avg.data.data - mon_gfed_1d.data), axis=0))[
        -10:
    ]:
        plt.figure()
        plt.plot(mon_avg.data[:, i])
        plt.plot(mon_gfed_1d[:, i])
        plt.show()
