# -*- coding: utf-8 -*-
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory

from python_inferno.configuration import land_pts
from python_inferno.data import load_data
from python_inferno.multi_timestep_inferno import multi_timestep_inferno
from python_inferno.precip_dry_day import calculate_inferno_dry_days
from python_inferno.utils import unpack_wrapped

memory = Memory(str(Path(os.environ["EPHEMERAL"]) / "joblib_cache"), verbose=10)


if __name__ == "__main__":
    (
        t1p5m_tile,
        q1p5m_tile,
        pstar,
        sthu_soilt,
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
        obs_fuel_build_up_1d,
        jules_ba_gb,
        jules_time_coord,
    ) = load_data(N=100)

    # Define the ignition method (`ignition_method`).
    ignition_method = 1

    timestep = 3600 * 4

    ba, dry_bal = unpack_wrapped(multi_timestep_inferno)(
        t1p5m_tile=t1p5m_tile,
        q1p5m_tile=q1p5m_tile,
        pstar=pstar,
        sthu_soilt=sthu_soilt,
        frac=frac,
        c_soil_dpm_gb=c_soil_dpm_gb,
        c_soil_rpm_gb=c_soil_rpm_gb,
        canht=canht,
        ls_rain=ls_rain,
        con_rain=con_rain,
        # Not used for ignition mode 1.
        pop_den=np.zeros((land_pts,)) - 1,
        flash_rate=np.zeros((land_pts,)) - 1,
        ignition_method=ignition_method,
        fuel_build_up=np.repeat(
            np.expand_dims(obs_fuel_build_up_1d.data, 1), repeats=13, axis=1
        ),
        fapar_diag_pft=np.repeat(
            np.expand_dims(obs_fapar_1d.data, 1), repeats=13, axis=1
        ),
        dry_days=unpack_wrapped(calculate_inferno_dry_days)(
            ls_rain, con_rain, threshold=1.0, timestep=timestep
        ),
        flammability_method=2,
        fapar_factor=-4.83e1,
        fapar_centre=4.0e-1,
        fuel_build_up_factor=1.01e1,
        fuel_build_up_centre=3.76e-1,
        temperature_factor=8.01e-2,
        temperature_centre=2.82e2,
        dry_day_factor=0.05,
        dry_day_centre=-0.9,
        dryness_method=2,
        rain_f=0.5,
        vpd_f=2500,
        dry_bal_factor=-1.1,
        dry_bal_centre=-0.98,
        return_dry_bal=True,
    )

    combined_mask = gfed_ba_1d.mask | obs_fapar_1d.mask | obs_fuel_build_up_1d.mask
    combined_mask = np.expand_dims(combined_mask, axis=1)
    combined_mask = np.repeat(combined_mask, axis=1, repeats=13)

    dry_bal = np.ma.MaskedArray(dry_bal, mask=combined_mask)

    valid = np.where(~np.any(combined_mask, axis=(0, 1)))[0]

    plt.figure()
    xs = np.arange(0, dry_bal.shape[0]) * timestep / (60 * 60 * 24)  # Convert to days.
    for p in range(dry_bal.shape[1]):
        plt.plot(xs, dry_bal[:, p, valid[0]], label=p)
    plt.xlabel("days")
    plt.legend()
    plt.title("'dry_bal' variable (influenced by 'vpd_f' & 'rain_f')")
