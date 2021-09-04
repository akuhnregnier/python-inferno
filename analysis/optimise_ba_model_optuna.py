#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc

import optuna

from python_inferno.optimisation import gen_to_optimise
from python_inferno.optuna import OptunaSpace
from python_inferno.space import generate_space

space_template = dict(
    fapar_factor=(3, [(-50, -1)], "suggest_float"),
    fapar_centre=(3, [(-0.1, 1.1)], "suggest_float"),
    fuel_build_up_n_samples=(3, [(100, 1300, 400)], "suggest_int"),
    fuel_build_up_factor=(3, [(0.5, 30)], "suggest_float"),
    fuel_build_up_centre=(3, [(0.0, 0.5)], "suggest_float"),
    temperature_factor=(3, [(0.07, 0.2)], "suggest_float"),
    temperature_centre=(3, [(260, 295)], "suggest_float"),
    rain_f=(3, [(0.8, 2.0)], "suggest_float"),
    vpd_f=(3, [(400, 2200)], "suggest_float"),
    dry_bal_factor=(3, [(-100, -1)], "suggest_float"),
    dry_bal_centre=(3, [(-3, 3)], "suggest_float"),
    # Averaged samples between ~1 week and ~1 month (4 hrs per sample).
    average_samples=(1, [(40, 160, 60)], "suggest_int"),
)

space = OptunaSpace(
    generate_space(space_template), remap_float_to_0_1=True, replicate_pft_groups=True
)


def fail_func(*args, **kwargs):
    raise optuna.exceptions.TrialPruned


def success_func(loss, *args, **kwargs):
    return loss


to_optimise = gen_to_optimise(
    fail_func=fail_func,
    success_func=success_func,
)


def objective(trial):
    gc.collect()

    suggested_params = space.suggest(trial)
    loss = to_optimise(suggested_params)

    gc.collect()
    return loss


if __name__ == "__main__":
    study_name = "optuna5"
    study = optuna.load_study(
        sampler=optuna.samplers.CmaEsSampler(restart_strategy="ipop"),
        study_name=f"{study_name}",
        storage=f"mysql://alex@maritimus.webredirect.org/{study_name}",
    )
    study.optimize(objective, n_trials=5000)
