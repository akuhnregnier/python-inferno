#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import matplotlib as mpl
from loguru import logger

from python_inferno.hyperopt import get_space_template
from python_inferno.iter_opt import ALWAYS_OPTIMISED, IGNORED
from python_inferno.model_params import get_model_params, get_param_uncertainties

if __name__ == "__main__":
    mpl.rc_file(Path(__file__).absolute().parent / "matplotlibrc")
    save_dir = Path("~/tmp/ba-model-param-uncertainties/").expanduser()
    save_dir.mkdir(exist_ok=True, parents=False)

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    df, method_iter = get_model_params(progress=False, verbose=False)

    for (
        dryness_method,
        fuel_build_up_method,
        df_sel,
        min_index,
        min_loss,
        params,
        exp_name,
        exp_key,
    ) in method_iter():
        assert int(params["include_temperature"]) == 1

        logger.info(exp_name)
        logger.info(exp_key)

        plot_save_dir = save_dir / exp_key
        plot_save_dir.mkdir(exist_ok=True, parents=False)

        space_template = get_space_template(
            dryness_method=dryness_method,
            fuel_build_up_method=fuel_build_up_method,
            include_temperature=1,
        )

        param_names = [
            key for key in space_template if key not in ALWAYS_OPTIMISED.union(IGNORED)
        ]

        # Filter columns.
        new_cols = [
            col
            for col in df_sel.columns
            if any(name_root in col for name_root in param_names)
        ] + ["loss"]

        get_param_uncertainties(
            df_sel=df_sel[new_cols],
            exp_name=exp_name,
            plot=True,
            save_dir=plot_save_dir,
        )
