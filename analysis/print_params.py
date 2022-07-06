#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from itertools import islice
from pathlib import Path
from pprint import pprint

from loguru import logger

from python_inferno.model_params import get_model_params

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    record_dir = Path(os.environ["EPHEMERAL"]) / "opt_record"
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
    ) in islice(method_iter(), 0, None):
        assert int(params["include_temperature"]) == 1

        logger.info(exp_name)
        pprint(params)
