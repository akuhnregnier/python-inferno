#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from collections import defaultdict
from itertools import islice
from pathlib import Path
from time import time

import numpy as np
from loguru import logger

from python_inferno.ba_model import BAModel, GPUBAModel
from python_inferno.data import load_jules_lats_lons
from python_inferno.model_params import get_model_params

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    jules_lats, jules_lons = load_jules_lats_lons()

    # XXX - 'opt_record_bak' vs. 'opt_record'
    record_dir = Path(os.environ["EPHEMERAL"]) / "opt_record_bak"
    df, method_iter = get_model_params(
        record_dir=record_dir, progress=True, verbose=False
    )

    for (
        _,
        _,
        _,
        _,
        _,
        params,
        exp_name,
        _,
    ) in islice(method_iter(), 0, None):
        logger.info(exp_name)

        outputs = dict()
        times = defaultdict(list)

        for name, model_class in [
            ("python", BAModel),
            ("metal", GPUBAModel),
        ]:
            # `**params` is used twice here because the functions simply use the
            # kwargs they require, ignoring the rest.
            ba_model = model_class(**params)

            for i in range(100 if name == "python" else 1000):
                start = time()

                outputs[name] = ba_model.run(
                    **{
                        **dict(
                            fapar_weight=1,
                            dryness_weight=1,
                            temperature_weight=1,
                            fuel_weight=1,
                        ),
                        **params,
                    }
                )
                times[name].append(time() - start)

            if name == "metal":
                ba_model._gpu_inferno.release()

        for name, time_vals in times.items():
            assert time_vals

            if len(time_vals) > 1:
                sel_time_vals = time_vals[1:]
                if len(sel_time_vals) == 1:
                    std = 0
                    mean = sel_time_vals[0]
                else:
                    std = np.std(sel_time_vals)
                    mean = np.mean(sel_time_vals)
            else:
                std = 0
                mean = time_vals[0]

            extra = ""
            if std / mean > 0.1:
                extra = "!!"

            print(f"Time taken by '{name:<10}': {mean:0.1e} ± {std:0.1e} {extra}")

        diffs = outputs["python"]["model_ba"] - outputs["metal"]["model_ba"]
        print(
            "Diffs: "
            + ", ".join(
                format(d, "0.1e")
                for d in [
                    np.mean(diffs),
                    np.mean(np.abs(diffs)),
                    np.max(diffs),
                    np.max(np.abs(diffs)),
                ]
            )
        )
