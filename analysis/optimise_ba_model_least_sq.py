# -*- coding: utf-8 -*-
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from joblib import Memory
from scipy.optimize import least_squares
from sklearn.metrics import r2_score

from python_inferno.configuration import land_pts
from python_inferno.data import load_data
from python_inferno.multi_timestep_inferno import multi_timestep_inferno
from python_inferno.precip_dry_day import calculate_inferno_dry_days

memory = Memory(str(Path(os.environ["EPHEMERAL"]) / "joblib_cache"), verbose=10)


@memory.cache(ignore=["verbose"])
def optimize_ba(
    *,
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
    obs_fuel_build_up_1d_data,
    obs_fapar_1d_data,
    combined_mask,
    gfed_ba_1d_data,
    verbose=2,
):
    # Define the parameters to optimise and their respective bounds.
    opt_bounds = OrderedDict(
        fapar_factor=(-50, -1),
        fapar_centre=(0.25, 0.7),
        fuel_build_up_factor=(1, 30),
        fuel_build_up_centre=(0.2, 0.6),
        temperature_factor=(0.05, 0.25),
        temperature_centre=(250, 350),
        dry_day_factor=(0.01, 0.12),
        dry_day_centre=(100, 400),
        dry_day_threshold=(1e-6, 1e-4),
    )

    # Specified initial parameters - optional.
    opt_x0 = dict(
        fapar_factor=-4.83e1,
        fapar_centre=4.0e-1,
        fuel_build_up_factor=1.01e1,
        fuel_build_up_centre=3.76e-1,
        temperature_factor=8.01e-2,
        temperature_centre=2.82e2,
        dry_day_factor=2.0e-2,
        dry_day_centre=1.73e2,
        dry_day_threshold=2.83e-5,
    )

    # Default values.
    kwargs = dict(
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
        ignition_method=1,
        fuel_build_up=np.repeat(
            np.expand_dims(obs_fuel_build_up_1d_data, 1), repeats=13, axis=1
        ),
        fapar_diag_pft=np.repeat(
            np.expand_dims(obs_fapar_1d_data, 1), repeats=13, axis=1
        ),
        dry_days=calculate_inferno_dry_days(
            ls_rain, con_rain, threshold=4.3e-5, timestep=3600 * 4
        ),
        flammability_method=2,
        dryness_method=1,
        fapar_factor=-30.0,
        fapar_centre=0.3,
        fuel_build_up_factor=25.0,
        fuel_build_up_centre=0.4,
        temperature_factor=0.08,
        temperature_centre=300.0,
        dry_day_factor=0.05,
        dry_day_centre=400.0,
    )

    def to_optimise(x):
        """Function to optimise."""
        # Insert new parameters into the kwargs.
        for name, value in zip(opt_bounds, x):
            if name == "dry_day_threshold":
                kwargs["dry_days"] = calculate_inferno_dry_days(
                    ls_rain, con_rain, threshold=value, timestep=3600 * 4
                )
            else:
                kwargs[name] = value

        model_ba = multi_timestep_inferno(**kwargs)

        if np.all(np.isclose(model_ba, 0, rtol=0, atol=1e-15)):
            r2 = -100.0
        else:
            # Compute R2 score after normalising each by their mean.
            y_true = gfed_ba_1d_data[~combined_mask]
            y_pred = model_ba[~combined_mask]

            y_true /= np.mean(y_true)
            y_pred /= np.mean(y_pred)

            r2 = r2_score(y_true=y_true, y_pred=y_pred)

        print("R2:", r2)

        # We want R2=1, but the optimisation algorithm wants to
        # minimise the return value.
        return -(r2 - 1)

    return least_squares(
        to_optimise,
        # Start in the middle of each range.
        x0=[opt_x0.get(name, np.mean(values)) for name, values in opt_bounds.items()],
        bounds=tuple(zip(*opt_bounds.values())),
        verbose=verbose,
        max_nfev=20,
    )


def main():
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
    ) = load_data(N=None)

    combined_mask = gfed_ba_1d.mask | obs_fapar_1d.mask | obs_fuel_build_up_1d.mask

    out = optimize_ba(
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
        obs_fuel_build_up_1d_data=obs_fuel_build_up_1d.data,
        obs_fapar_1d_data=obs_fapar_1d.data,
        combined_mask=combined_mask,
        gfed_ba_1d_data=gfed_ba_1d.data,
    )
    return out


if __name__ == "__main__":
    out = main()
    print(out.x)
