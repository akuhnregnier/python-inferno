#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import string
from collections import OrderedDict
from pathlib import Path

import numpy as np
from joblib import Memory
from scipy.optimize import basinhopping
from sklearn.metrics import r2_score
from wildfires.dask_cx1.dask_rf import safe_write

from python_inferno.configuration import land_pts
from python_inferno.cx1 import run
from python_inferno.data import load_data, timestep
from python_inferno.multi_timestep_inferno import multi_timestep_inferno
from python_inferno.precip_dry_day import calculate_inferno_dry_days
from python_inferno.utils import unpack_wrapped

memory = Memory(str(Path(os.environ["EPHEMERAL"]) / "joblib_cache"), verbose=10)


class Recorder:
    def __init__(self, record_dir=None):
        """Initialise."""
        self.xs = []
        self.fvals = []
        self.filename = Path(record_dir) / (
            "".join(random.choices(string.ascii_lowercase, k=20)) + ".pkl"
        )
        self.filename.parent.mkdir(exist_ok=True)
        print("filename:", self.filename)

    def record(self, x, fval):
        """Record parameters and function value."""
        self.xs.append(x)
        self.fvals.append(fval)

    def dump(self):
        """Dump the recorded values to file."""
        safe_write((self.xs, self.fvals), self.filename)


# @memory.cache(ignore=["verbose", "record_dir"])
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
    dryness_method=1,
    verbose=True,
    seed=None,
    record_dir=None,
):
    # Define the parameters to optimise by giving their starting values and expected
    # min / max values.
    opt_x0 = OrderedDict(
        fapar_factor=-4.83e1,
        fapar_centre=4.0e-1,
        fuel_build_up_factor=1.01e1,
        fuel_build_up_centre=3.76e-1,
        temperature_factor=8.01e-2,
        temperature_centre=2.82e2,
    )
    opt_range = OrderedDict(
        fapar_factor=(-5e1, -5),
        fapar_centre=(3e-1, 6e-1),
        fuel_build_up_factor=(2, 2e1),
        fuel_build_up_centre=(3e-1, 5e-1),
        temperature_factor=(1e-2, 1e-1),
        temperature_centre=(2.5e2, 3e2),
    )

    if dryness_method == 1:
        opt_x0.update(
            OrderedDict(
                dry_day_factor=2.0e-2,
                dry_day_centre=1.73e2,
                dry_day_threshold=1.0,
            )
        )
        opt_range.update(
            OrderedDict(
                dry_day_factor=(1e-2, 4e-2),
                dry_day_centre=(0.5e2, 5e2),
                dry_day_threshold=(0.5, 1.5),
            )
        )
    elif dryness_method == 2:
        opt_x0.update(
            OrderedDict(
                dry_bal_centre=-0.5,
                dry_bal_factor=-3,
                rain_f=0.15,
                vpd_f=1000,
            )
        )
        opt_range.update(
            OrderedDict(
                dry_bal_centre=(-1, 1),
                dry_bal_factor=(-10, -1),
                rain_f=(0.01, 0.2),
                vpd_f=(500, 1500),
            )
        )
    else:
        raise ValueError(f"Unsupported `dryness_method` {dryness_method}.")

    def transform_param(name, value):
        """Transform parameter closer to a [0, 1] range."""
        min_p, max_p = sorted(opt_range[name])
        return (value - min_p) / (max_p - min_p)

    def inv_transform_param(name, value):
        """Inverse transform of the above."""
        min_p, max_p = sorted(opt_range[name])
        return value * (max_p - min_p) + min_p

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
        dry_days=unpack_wrapped(calculate_inferno_dry_days)(
            ls_rain, con_rain, threshold=1.0, timestep=timestep
        ),
        flammability_method=2,
        dryness_method=dryness_method,
        fapar_factor=-4.83e1,
        fapar_centre=4.0e-1,
        fuel_build_up_factor=1.01e1,
        fuel_build_up_centre=3.76e-1,
        temperature_factor=8.01e-2,
        temperature_centre=2.82e2,
        dry_day_factor=2.0e-2,
        dry_day_centre=1.73e2,
        rain_f=1,
        vpd_f=5e5,
        dry_bal_factor=1,
        dry_bal_centre=0,
        timestep=timestep,
    )

    if record_dir is not None:
        recorder = Recorder(record_dir=record_dir)
    else:
        recorder = None

    def basinhopping_callback(x, f, accept):
        print("callback")
        if recorder is not None:
            print("dump")
            r2 = 1 - f
            recorder.record(
                {name: inv_transform_param(name, val) for name, val in zip(opt_x0, x)},
                r2,
            )

            # Update record in file.
            recorder.dump()

    def to_optimise(x):
        """Function to optimise."""
        # Insert new parameters into the kwargs.
        for name, _value in zip(opt_x0, x):
            # Invert the transform.
            value = inv_transform_param(name, _value)

            if dryness_method == 1 and name == "dry_day_threshold":
                kwargs["dry_days"] = calculate_inferno_dry_days(
                    ls_rain, con_rain, threshold=value, timestep=timestep
                )
            else:
                kwargs[name] = value

        model_ba = unpack_wrapped(multi_timestep_inferno)(**kwargs)

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

    return (
        basinhopping(
            to_optimise,
            # Map into normalised intervals.
            x0=list(transform_param(name, val) for name, val in opt_x0.items()),
            disp=verbose,
            minimizer_kwargs=dict(
                method="bfgs",
                jac=None,
                options=dict(maxiter=12, disp=verbose, eps=0.04),
            ),
            seed=seed,
            niter_success=10,
            callback=basinhopping_callback,
        ),
        opt_x0.keys(),
    )


def main(*args, **kwargs):
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
        dryness_method=2,
        seed=None,
        record_dir=Path(os.environ["EPHEMERAL"]) / "opt_record",
    )
    return out


if __name__ == "__main__":
    run(main, [None] * 1000, cx1_kwargs=dict(walltime="24:00:00", ncpus=2, mem="10GB"))
