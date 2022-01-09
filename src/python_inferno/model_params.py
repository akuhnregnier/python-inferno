# -*- coding: utf-8 -*-
import pickle
from enum import Enum
from itertools import product
from pprint import pprint

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import get_exp_key, get_exp_name

NoVal = Enum("NoVal", ["NoVal"])


def check_params(params, key, value=NoVal.NoVal):
    if all(key in p for p in params):
        if value is not NoVal.NoVal:
            if all(p[key] == value for p in params):
                return True
        else:
            return True
    return False


def get_model_params(*, record_dir, verbose=False, progress=True):
    assert record_dir.is_dir()

    global_params = []
    global_losses = []

    for fname in record_dir.glob("*"):
        with fname.open("rb") as f:
            params, losses = pickle.load(f)

        if check_params(params, "dryness_method", 1):
            assert check_params(params, "dry_day_factor")
        elif check_params(params, "dryness_method", 2):
            assert check_params(params, "dry_bal_factor")
        else:
            raise ValueError("dryness_method")

        if check_params(params, "fuel_build_up_method", 1):
            assert check_params(params, "fuel_build_up_factor")
        elif check_params(params, "fuel_build_up_method", 2):
            assert check_params(params, "litter_pool_factor")
        else:
            raise ValueError("fuel_build_up_method")

        if check_params(params, "include_temperature", 1):
            assert check_params(params, "temperature_factor")
        elif check_params(params, "include_temperature", 0):
            assert not check_params(params, "temperature_factor")
        else:
            raise ValueError("include_temperature")

        for ps, loss in zip(params, losses):
            if loss > 0.95:
                # Skip poor samples.
                continue

            global_params.append(ps)
            global_losses.append(loss)

    df = pd.DataFrame(global_params)
    df["loss"] = global_losses

    cat_names = ["dryness_method", "fuel_build_up_method", "include_temperature"]

    for name in cat_names:
        df[name] = df[name].astype("int")

    if verbose:
        print(df.head())
        print("\nNumber of trials:\n")
        print(df.groupby(cat_names).size())

        print("\nMinimum loss by parametrisation approach:\n")
        print(df.groupby(cat_names)["loss"].min())

    def method_iter():
        for dryness_method, fuel_build_up_method in tqdm(
            list(product([1, 2], [1, 2])), disable=not progress, desc="Methods"
        ):
            sel = (df["dryness_method"] == dryness_method) & (
                df["fuel_build_up_method"] == fuel_build_up_method
            )
            if not np.any(sel):
                continue

            df_sel = df[sel]
            min_index = df_sel["loss"].argmin()
            min_loss = df_sel.iloc[min_index]["loss"]

            params = {
                key: val
                for key, val in df_sel.iloc[min_index].to_dict().items()
                if not pd.isna(val) and key not in ("loss",)
            }
            if verbose:
                pprint(params)

            exp_name = get_exp_name(
                dryness_method=dryness_method, fuel_build_up_method=fuel_build_up_method
            )

            exp_key = get_exp_key(
                dryness_method=dryness_method, fuel_build_up_method=fuel_build_up_method
            )

            yield (
                dryness_method,
                fuel_build_up_method,
                df_sel,
                min_index,
                min_loss,
                params,
                exp_name,
                exp_key,
            )

    return df, method_iter
