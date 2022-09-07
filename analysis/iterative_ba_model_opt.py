#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm

from python_inferno.iter_opt import iterative_ba_model_opt, vis_result
from python_inferno.model_params import get_model_params

if __name__ == "__main__":
    mpl.rc_file(Path(__file__).absolute().parent / "matplotlibrc")
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    save_dir = Path("~/tmp/iter-opt-models").expanduser()
    save_dir.mkdir(parents=False, exist_ok=True)

    df, method_iter = get_model_params(
        record_dir=Path(os.environ["EPHEMERAL"]) / "opt_record",
        progress=False,
        verbose=False,
    )

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 8))

    for i, method_data in enumerate(method_iter()):
        full_opt_loss = method_data[4]
        params = method_data[5]
        method_name = method_data[6]

        ax = axes.ravel()[i]
        ax.set_title(method_name)

        method_dir = save_dir / method_name
        method_dir.mkdir(parents=False, exist_ok=True)

        for key, marker, colors, opt_kwargs in (
            ("50,1", "^", ("C0", "C1"), dict(maxiter=50, niter_success=1)),
            ("1000,5", "x", ("C2", "C3"), dict(maxiter=1000, niter_success=5)),
        ):
            results, aic_results, cv_results = iterative_ba_model_opt(
                params=params, **opt_kwargs
            )

            n_params, aics, arcsinh_aics = zip(
                *(
                    (n_params, data["aic"], data["arcsinh_aic"])
                    for (n_params, data) in aic_results.items()
                )
            )
            aics = np.asarray(aics)
            arcsinh_aics = np.asarray(arcsinh_aics)

            opt_dir = method_dir / key.replace(",", "_")
            opt_dir.mkdir(parents=False, exist_ok=True)

            for n, result in tqdm(results.items(), desc="Plotting model vis"):
                vis_result(
                    result=result,
                    dryness_method=int(params["dryness_method"]),
                    fuel_build_up_method=int(params["fuel_build_up_method"]),
                    save_key=str(n),
                    save_dir=opt_dir,
                )

            n_params, losses = zip(
                *((n, float(loss)) for (n, (loss, _)) in results.items())
            )

            ms = 6
            aic_ms = 8

            ax.plot(
                n_params,
                losses,
                marker=marker,
                linestyle="",
                label=key,
                ms=ms,
                c=colors[0],
            )

            min_aic_i = np.argmin(aics)
            min_arcsinh_aic_i = np.argmin(arcsinh_aics)

            print("AIC indices:", min_aic_i, min_arcsinh_aic_i)

            ax.plot(
                n_params[min_aic_i],
                losses[min_aic_i],
                marker=marker,
                linestyle="",
                ms=aic_ms,
                c="r",
                zorder=4,
            )
            ax.plot(
                n_params[min_arcsinh_aic_i],
                losses[min_arcsinh_aic_i],
                marker=marker,
                linestyle="",
                ms=aic_ms,
                c="g",
                zorder=5,
            )

            # CV plotting.
            cv_n_params, cv_test_losses = zip(*cv_results.items())
            ax.plot(
                cv_n_params,
                cv_test_losses,
                marker=marker,
                linestyle="",
                label=f"{key} CV",
                ms=ms,
                c=colors[1],
            )

            print(method_name, key)
            for (n, (loss, _)) in results.items():
                if n in cv_results:
                    # print(n, loss - cv_results[n])
                    print(n, cv_results[n])

        ax.hlines(full_opt_loss, 0, max(n_params), colors="g")
        ax.legend()

        if i in (0, 2):
            ax.set_ylabel("performance (loss)")
        if i in (2, 3):
            ax.set_xlabel("nr. of optimised parameters")

    fig.savefig(save_dir / "iter_model_perf.png")
    plt.close(fig)
