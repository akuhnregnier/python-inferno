#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
from itertools import islice
from pathlib import Path

import black
import numpy as np
from sympy import lambdify, pycode

from python_inferno.ba_model import BAModel
from python_inferno.configuration import avg_ba
from python_inferno.sympy_inferno import get_grid_cell_metrics, wrap_lambdify_kwargs

# XXX
# sys.path.append("/tmp")
#
# from sym_nme import sym_nme


# @cache - caching these return values does not work!
def wrapper(
    symbol_dict,
    grid_cell_nme,
    grid_cell_mpd,
):
    lam_nme = lambdify(list(symbol_dict.values()), grid_cell_nme)
    lam_mpd = lambdify(list(symbol_dict.values()), grid_cell_mpd)

    l_grid_cell_nme = wrap_lambdify_kwargs(list(symbol_dict))(lam_nme)
    l_grid_cell_mpd = wrap_lambdify_kwargs(list(symbol_dict))(lam_mpd)

    return l_grid_cell_nme, l_grid_cell_mpd


def gen_sympy_func(*, func_name, symbol_dict, func_expr):
    args_str = ",\n".join(["*"] + list(symbol_dict))
    body = ["import math", f"def {func_name}({args_str}):"]
    for name, val in symbol_dict.items():
        if hasattr(val, "__iter__"):
            if isinstance(val, np.ndarray) and val.ndim > 1:
                body.append(" " * 4 + f"{repr(list(val.ravel()))} = {name}.ravel()")
            else:
                body.append(" " * 4 + f"{repr(list(val))} = {name}")
            # Empty line.
            body.append("")

    code = "\n".join(body) + "\n\n" + " " * 4 + "return " + pycode(func_expr)
    return black.format_str(code, mode=black.Mode())


if __name__ == "__main__":
    with (
        Path(__file__).parent.parent
        / "tests"
        / "test_data"
        / f"best_params_litter_v2.pkl"
    ).open("rb") as f:
        model_params = pickle.load(f)

    index = 0

    params = next(islice(iter(model_params.values()), index, index + 1))

    ba_model = BAModel(**params)
    proc_params = ba_model.process_kwargs(**params)
    pred_ba = ba_model.run(**params)["model_ba"]

    all_crop = ba_model.obs_pftcrop_1d
    assert not np.any(np.ma.getmaskarray(all_crop))
    all_crop = np.ma.getdata(all_crop)

    land_index = 1000

    obs_ba = ba_model.mon_avg_gfed_ba_1d[:, land_index]
    assert not np.any(np.ma.getmaskarray(obs_ba))
    obs_ba = np.ma.getdata(obs_ba)

    avg_weights = (
        ba_model._cons_monthly_avg.weights
        / ba_model._cons_monthly_avg.weights.sum(axis=0)
    )

    (
        symbol_dict,
        grid_cell_nme,
        grid_cell_mpd,
    ) = get_grid_cell_metrics(avg_weights=avg_weights, obs_ba=obs_ba)

    # code = gen_sympy_func(
    #     func_name="sym_nme", symbol_dict=symbol_dict, func_expr=grid_cell_nme
    # )

    # with open("/tmp/sym_nme.py", "w") as f:
    #     f.write(code)

    for name in list(proc_params):
        if "litter_pool" in name or "dry_bal" in name:
            del proc_params[name]

    sym_kwargs = dict(
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

    # import sys

    # sys.exit(0)

    # l_grid_cell_nme, l_grid_cell_mpd = wrapper(
    #     symbol_dict,
    #     grid_cell_nme,
    #     grid_cell_mpd,
    # )

    # sym_nme = l_grid_cell_nme(**sym_kwargs)

    # sym_mpd = l_grid_cell_mpd(**sym_kwargs)

    # pred = ba_model._cons_monthly_avg.cons_monthly_average_data(pred_ba)[:, land_index]
    # ref_nme = nme_simple(
    #     pred=np.arcsinh(ARCSINH_FACTOR * pred), obs=np.arcsinh(ARCSINH_FACTOR * obs_ba)
    # )

    # ref_mpd = mpd(obs=obs_ba[:, None], pred=pred[:, None])
