# -*- coding: utf-8 -*-
import optuna

from python_inferno.optuna import OptunaSpace


def test_optuna():
    def objective(trial):
        space = OptunaSpace({"x": ("suggest_float", -10, 10)})
        x = space.suggest(trial)["x"]
        return (x - 2) ** 2

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    assert abs(study.best_params["x"] - 2) < 0.5
