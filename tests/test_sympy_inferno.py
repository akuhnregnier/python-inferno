# -*- coding: utf-8 -*-
from itertools import islice

import numpy as np
import pytest
from numba import njit
from sympy import lambdify, symarray, symbols

from python_inferno.ba_model import BAModel
from python_inferno.configuration import avg_ba
from python_inferno.sympy_inferno import get_grid_cell_metrics, wrap_lambdify_kwargs


def test_wrap_lambdify_kwargs():
    x, y = symbols("x y")
    orig_f = lambdify([x, y], x, dummify=True)
    assert orig_f(1, 2) == 1

    wrapped_f = wrap_lambdify_kwargs(["x", "y"])(orig_f)

    assert wrapped_f(x=1, y=2) == 1
    assert wrapped_f(y=1, x=2) == 2


def test_wrap_lambdify_kwargs_symarray():
    arr = symarray("arr", (2,))
    arr2 = symarray("arr2", (2,))
    orig_f = njit(lambdify([arr, arr2], arr[0] - arr2[1], dummify=True))
    assert orig_f(np.array([1, 2]), np.array([3, 4])) == -3

    wrapped_f = wrap_lambdify_kwargs(["x", "y"])(orig_f)

    assert wrapped_f(x=np.array([1, 2]), y=np.array([3, 4])) == -3
    assert wrapped_f(y=np.array([1, 2]), x=np.array([3, 4])) == 1


@pytest.mark.parametrize("index", range(4))
def test_sympy_inferno(index, model_params):
    params = next(islice(iter(model_params.values()), index, index + 1))

    ba_model = BAModel(**params)
    proc_params = ba_model.process_kwargs(**params)
    ba_model.run(**params)["model_ba"]

    all_crop = ba_model.obs_pftcrop_1d
    assert not np.any(np.ma.getmaskarray(all_crop))
    all_crop = np.ma.getdata(all_crop)

    land_index = 1000

    obs_ba = ba_model.mon_avg_gfed_ba_1d[:, land_index]
    assert not np.any(np.ma.getmaskarray(obs_ba))
    obs_ba = np.ma.getdata(obs_ba)

    symbol_dict, grid_cell_nme, grid_cell_mpd = get_grid_cell_metrics(
        avg_weights=ba_model._cons_monthly_avg.weights, obs_ba=obs_ba
    )

    l_grid_cell_nme = wrap_lambdify_kwargs(list(symbol_dict))(
        (lambdify(list(symbol_dict.values()), grid_cell_nme))
    )

    nme = l_grid_cell_nme(
        **proc_params,
        crop_f=params["crop_f"],
        avg_ba=avg_ba,
        temp_l=ba_model.data_dict["t1p5m_tile"][:, :, land_index],
        fapar=ba_model.data_dict["fapar_diag_pft"][:, :, land_index],
        dry_days=ba_model.data_dict["dry_days"][:, land_index],
        fuel_build_up=ba_model.data_dict["fuel_build_up"][:, :, land_index],
        crop=all_crop[:, land_index],
        frac=ba_model.data_dict["frac"][:, :, land_index],
    )
