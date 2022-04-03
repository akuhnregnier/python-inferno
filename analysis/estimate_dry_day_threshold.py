#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.metrics import r2_score

from python_inferno.analysis.dry_day_analysis import dry_day_calc, prepare_data
from python_inferno.metrics import calculate_factor_r2
from python_inferno.utils import tqdm

memory = Memory(str(Path(os.environ["EPHEMERAL"]) / "joblib_cache"), verbose=10)


if __name__ == "__main__":
    (
        ls_rain,
        con_rain,
        calc_mask,
        jules_time_coord,
        mon_era_dd_1d,
        era_dd,
        jules_lats,
        jules_lons,
    ) = prepare_data()

    results = {}
    for threshold in tqdm(np.linspace(0.5, 2, 60)):
        mon_avg_inferno_dry_days, mon_era_dd_1d = dry_day_calc(
            ls_rain=ls_rain,
            con_rain=con_rain,
            calc_mask=calc_mask,
            jules_time_coord=jules_time_coord,
            mon_era_dd_1d=mon_era_dd_1d,
            threshold=threshold,
        )

        shared_mask = np.ma.getmaskarray(mon_era_dd_1d) | np.ma.getmaskarray(
            mon_avg_inferno_dry_days
        )

        y_true = np.ma.getdata(mon_era_dd_1d)[~shared_mask]
        y_pred = np.ma.getdata(mon_avg_inferno_dry_days)[~shared_mask]

        factor = calculate_factor_r2(y_true=y_true, y_pred=y_pred)

        results[threshold] = {
            "r2": r2_score(y_true=y_true, y_pred=factor * y_pred),
            "r2_raw": r2_score(y_true=y_true, y_pred=y_pred),
            "factor": factor,
        }

    results = pd.DataFrame(results).T

    thresholds = results.index.values

    fig, ax = plt.subplots()
    ax.plot(thresholds, results["r2"], label="r2")
    # ax.plot(thresholds, results['r2_raw'], label='r2_raw')
    # ax.legend(loc='best')
    ax.set_xlabel("threshold")
    ax.set_ylabel("r2")
    ax.grid()

    ax2 = ax.twinx()
    ax2.plot(thresholds, results["factor"], c="C2")
    ax2.set_ylabel("factor", color="C2")

    fig.savefig(Path("~/tmp/dry_day_threshold_estimate.png").expanduser())
