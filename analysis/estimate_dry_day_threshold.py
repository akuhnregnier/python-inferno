#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from python_inferno.analysis.dry_day_analysis import dry_day_calc, prepare_data
from python_inferno.metrics import calculate_factor_r2
from python_inferno.plotting import use_style
from python_inferno.utils import tqdm


def get_thres_results(
    *,
    ls_rain,
    con_rain,
    calc_mask,
    jules_time_coord,
    mon_era_dd_1d,
    threshold,
):
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

    return {
        "r2": r2_score(y_true=y_true, y_pred=factor * y_pred),
        "r2_raw": r2_score(y_true=y_true, y_pred=y_pred),
        "factor": factor,
    }


if __name__ == "__main__":
    use_style()

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

    thresholds = np.linspace(0.5, 2, 60)

    futures = []
    results = {}
    with ProcessPoolExecutor() as executor:
        for threshold in thresholds:
            futures.append(
                executor.submit(
                    get_thres_results,
                    ls_rain=ls_rain,
                    con_rain=con_rain,
                    calc_mask=calc_mask,
                    jules_time_coord=jules_time_coord,
                    mon_era_dd_1d=mon_era_dd_1d,
                    threshold=threshold,
                )
            )

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

        for threshold, f in zip(thresholds, futures):
            results[threshold] = f.result()

    results = pd.DataFrame(results).T

    thresholds = results.index.values

    fig, ax = plt.subplots(figsize=(4, 2.5))

    ax.plot(thresholds, results["r2"], label="r2")
    # ax.plot(thresholds, results['r2_raw'], label='r2_raw')
    # ax.legend(loc='best')
    ax.set_xlabel("Threshold")
    ax.set_ylabel(r"$\mathrm{R}^2$")
    ax.grid(True, which="major", axis="x")

    ax2 = ax.twinx()
    ax2.plot(thresholds, results["factor"], c="C2")
    ax2.set_ylabel("Factor", color="C2")
    ax2.grid(False)

    fig.savefig(Path("~/tmp/dry_day_threshold_estimate.pdf").expanduser())
