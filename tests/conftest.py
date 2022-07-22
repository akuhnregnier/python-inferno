# -*- coding: utf-8 -*-
import pickle
from pathlib import Path

import pytest

TEST_DATA_DIR = Path(__file__).parent / "test_data"
SUFFIX = "litter_v2_clim_data_v2"


@pytest.fixture
def params_model_ba():
    with (TEST_DATA_DIR / f"best_params_{SUFFIX}.pkl").open("rb") as f:
        params_dict = pickle.load(f)

    with (TEST_DATA_DIR / f"model_ba_{SUFFIX}.pkl").open("rb") as f:
        model_ba_dict = pickle.load(f)

    return [[params_dict[key], model_ba_dict[key]] for key in params_dict]


@pytest.fixture
def model_params():
    with (TEST_DATA_DIR / f"best_params_{SUFFIX}.pkl").open("rb") as f:
        params_dict = pickle.load(f)

    return params_dict
