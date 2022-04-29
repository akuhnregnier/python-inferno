# -*- coding: utf-8 -*-
import pickle
from pathlib import Path

import pytest

TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture
def params_model_ba():
    with (TEST_DATA_DIR / "best_params.pkl").open("rb") as f:
        params_dict = pickle.load(f)

    with (TEST_DATA_DIR / "model_ba.pkl").open("rb") as f:
        model_ba_dict = pickle.load(f)

    return [[params_dict[key], model_ba_dict[key]] for key in params_dict]


@pytest.fixture
def model_params():
    with (TEST_DATA_DIR / "best_params.pkl").open("rb") as f:
        params_dict = pickle.load(f)

    return params_dict
