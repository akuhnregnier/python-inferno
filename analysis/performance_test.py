#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from collections import defaultdict
from functools import partial
from itertools import islice
from pathlib import Path
from time import time

import numpy as np
from loguru import logger

from python_inferno.ba_model import get_pred_ba_prep
from python_inferno.data import load_jules_lats_lons
from python_inferno.model_params import get_model_params
from python_inferno.multi_timestep_inferno import _multi_timestep_inferno
from python_inferno.py_gpu_inferno import run_single_shot
from python_inferno.utils import unpack_wrapped

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
        dryness_method,
        fuel_build_up_method,
        df_sel,
        min_index,
        min_loss,
        params,
        exp_name,
        exp_key,
    ) in islice(method_iter(), 0, 1):
        logger.info(exp_name)

        outputs = dict()
        times = defaultdict(list)

        for name, function in [
            ("python", partial(get_pred_ba_prep, _func=_multi_timestep_inferno)),
            ("metal", partial(get_pred_ba_prep, _func=run_single_shot)),
        ]:
            for i in range(10):
                start = time()

                outputs[name] = unpack_wrapped(function)(**params)
                times[name].append(time() - start)

        for name, time_vals in times.items():
            print(f"Times taken by '{name:<10}': {time_vals}.")

        for name, output_tup in outputs.items():
            print(name)
            model_ba = output_tup[0]
            print(model_ba.shape, np.unique(model_ba))
