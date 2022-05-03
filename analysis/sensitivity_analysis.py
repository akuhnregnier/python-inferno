#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from SALib.analyze import sobol
from SALib.sample import saltelli
from tqdm import tqdm

from python_inferno.ba_model import BAModel
from python_inferno.cache import cache
from python_inferno.data import load_jules_lats_lons, subset_sthu_soilt_inplace
from python_inferno.model_params import get_model_params


def append_to_bounds(bounds_list, values):
    new_bounds = [np.min(values), np.max(values)]
    if np.isclose(new_bounds[0], new_bounds[1]):
        print("Close bounds:", new_bounds[0])
        bounds_list.append([new_bounds[0], new_bounds[0] + 1])
    else:
        bounds_list.append(new_bounds)


@cache
def get_sobol_sis(*, params, land_index_args):
    params = {
        **dict(
            fapar_weight=1,
            dryness_weight=1,
            temperature_weight=1,
            fuel_weight=1,
        ),
        **params,
    }
    ba_model = BAModel(**params)

    group_vars = [
        "t1p5m_tile",
        "q1p5m_tile",
        "pstar",
        "sthu_soilt",  # Will be subset to have only temporal & land dimensions.
        # "frac",
        "c_soil_dpm_gb",
        "c_soil_rpm_gb",
        "canht",
        "ls_rain",
        "con_rain",
        "fuel_build_up",
        "fapar_diag_pft",
        "grouped_dry_bal",
        "dry_days",
        "litter_pool",
        # Extra param.
        "obs_pftcrop_1d",
    ]

    data_arrs = dict(**ba_model.data_dict, obs_pftcrop_1d=ba_model.obs_pftcrop_1d)
    data_arrs = subset_sthu_soilt_inplace(data_arrs)

    outputs = []

    # for land_index in range(land_pts):
    for land_index in tqdm(range(*land_index_args), desc="land"):
        var_indices = {}

        param_bounds = []
        param_names = []
        param_groups = []

        count = 0

        for name in group_vars:
            # Shape tuple without the temporal & land sizes.
            sub_shape = data_arrs[name].shape[1:-1]
            assert (
                len(sub_shape) <= 1
            ), f"Should only be PFT size at most, got {name}: {sub_shape}."

            n_sub_vars = sub_shape[0] if sub_shape else 1

            var_indices[name] = (count, count + n_sub_vars)

            if sub_shape:
                for pft_i in range(n_sub_vars):
                    vals = data_arrs[name][:, pft_i, land_index]
                    append_to_bounds(param_bounds, vals)
                    param_names.append(f"{name}_{pft_i}")
                    param_groups.append(name)
            else:
                vals = data_arrs[name][:, land_index]
                append_to_bounds(param_bounds, vals)
                param_names.append(name)
                param_groups.append(name)

            count += n_sub_vars

        problem = dict(
            groups=param_groups,
            names=param_names,
            num_vars=len(param_names),
            bounds=param_bounds,
        )

        param_values = saltelli.sample(problem, 2**6)
        assert len(param_values.shape) == 2

        Y = np.zeros(param_values.shape[0])
        for i in tqdm(range(param_values.shape[0]), desc="samples"):
            # Insert the data accordingly.
            for (name, (lower_i, upper_i)) in var_indices.items():
                assert upper_i <= param_values.shape[1]
                if (upper_i - lower_i) == 1:
                    # Replace all temporal values with this value.
                    data_arrs[name][:, land_index] = param_values[i, lower_i]
                else:
                    for index_j, j in enumerate(range(lower_i, upper_i)):
                        # Replace all temporal values with this value.
                        data_arrs[name][:, index_j, land_index] = param_values[i, j]

            # Update data variables.
            for key, var in ba_model.data_dict.items():
                if key == "sthu_soilt":
                    # TODO - Resolve this inconsistency everywhere this variable is
                    # used, i.e. simply return the single layer version originally.
                    x = data_arrs[key]
                    assert x.ndim == 2
                    updated = np.repeat(
                        data_arrs[key].reshape(x.shape[0], 1, 1, x.shape[1]), 4, axis=1
                    )
                else:
                    updated = data_arrs[key]

                assert ba_model.data_dict[key].shape == updated.shape
                ba_model.data_dict[key] = updated

            ba_model.obs_pftcrop_1d = data_arrs["obs_pftcrop_1d"]

            # Run the model with the new variables.
            model_ba = ba_model.run(
                land_point=land_index,
                **params,
            )["model_ba"]

            Y[i] = model_ba[0, land_index]

        Si = sobol.analyze(problem, Y, print_to_console=False)

        outputs.append(Si)
    return outputs


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

        sobol_sis = get_sobol_sis(params=params, land_index_args=(0, 3))

        group_names = list(sobol_sis[0].to_df()[0].index.values)

        data = {}
        for name in group_names:
            vals = [si.to_df()[0]["ST"][name] for si in sobol_sis]
            data[name] = dict(
                mean=np.nanmean(vals),
                std=np.nanstd(vals),
            )
        df = pd.DataFrame(data).T
        print(df)
