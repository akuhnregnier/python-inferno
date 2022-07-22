#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import sys
from itertools import islice
from pathlib import Path

import numpy as np
from loguru import logger

from python_inferno.ba_model import BAModel, GPUBAModel
from python_inferno.data import load_jules_lats_lons
from python_inferno.model_params import get_model_params

TEST_DATA_DIR = Path(__file__).parent.parent / "tests" / "test_data"


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    jules_lats, jules_lons = load_jules_lats_lons()

    record_dir = Path(os.environ["EPHEMERAL"]) / "opt_record"
    df, method_iter = get_model_params(
        record_dir=record_dir, progress=True, verbose=False
    )

    params_dict = {}
    model_ba_dict = {}

    for (
        dryness_method,
        fuel_build_up_method,
        df_sel,
        min_index,
        min_loss,
        params,
        exp_name,
        exp_key,
    ) in islice(method_iter(), 0, None):
        logger.info(exp_name)
        logger.info(exp_key)

        model_bas = dict()

        for name, model_class in [
            ("python", BAModel),
            ("metal", GPUBAModel),
        ]:
            # `**params` is used twice here because the functions simply use the
            # kwargs they require, ignoring the rest.
            ba_model = model_class(**params)

            model_bas[name] = ba_model.run(**params)["model_ba"]

            if name == "metal":
                ba_model._gpu_inferno.release()

        params_dict[exp_key] = params
        model_ba_dict[exp_key] = model_bas

        diffs = model_bas["python"] - model_bas["metal"]
        print(
            "Diffs: "
            + ", ".join(
                f"{descr}: {diff:0.1e}"
                for descr, diff in [
                    ("np.abs(np.mean(diffs))", np.abs(np.mean(diffs))),
                    ("np.mean(np.abs(diffs))", np.mean(np.abs(diffs))),
                    ("np.max(np.abs(diffs))", np.max(np.abs(diffs))),
                ]
            )
        )

    TEST_DATA_DIR.mkdir(parents=False, exist_ok=True)

    suffix = "litter_v2_clim_data_v2"

    with (TEST_DATA_DIR / f"best_params_{suffix}.pkl").open("wb") as f:
        pickle.dump(params_dict, f, protocol=-1)

    with (TEST_DATA_DIR / f"model_ba_{suffix}.pkl").open("wb") as f:
        pickle.dump(model_ba_dict, f, protocol=-1)
