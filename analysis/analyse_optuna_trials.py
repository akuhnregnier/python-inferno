#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optimise_ba_model_optuna import space

if __name__ == "__main__":
    exp_key = "optuna5"
    study = optuna.load_study(
        sampler=optuna.samplers.CmaEsSampler(),
        study_name=f"{exp_key}",
        storage=f"mysql://alex@maritimus.webredirect.org/{exp_key}",
    )
    df = study.trials_dataframe()

    fig_dir = Path(f"~/tmp/trials_{exp_key}").expanduser()
    if fig_dir.parent.exists():
        fig_dir.mkdir(exist_ok=True, parents=False)
        save = True
    else:
        save = False

    losses = np.array(df["value"][df["value"] < 2])

    plt.figure()
    plt.plot(losses)
    plt.ylabel("Loss")
    plt.xlabel("Iteration")

    # Calculate and plot the minimum loss up to and including the current iteration.
    min_losses = []
    for loss in losses:
        if not min_losses and loss is not None:
            # First valid loss.
            min_losses.append(loss)
        else:
            if loss is None or loss > min_losses[-1]:
                min_losses.append(min_losses[-1])
            else:
                min_losses.append(loss)

    plt.figure()
    plt.plot(min_losses)
    plt.ylabel("Minimum loss")
    plt.xlabel("Iteration")

    pprint(space.remap_if_needed(study.best_params))

    if save:
        plt.savefig(fig_dir / "min_loss_evolution.png")

        with (fig_dir / "argmin.json").open("w") as f:
            json.dump(space.remap_if_needed(study.best_params), f, indent=4)


max_loss = 0.87
values = df["value"]
selection = values <= max_loss
sel_values = values[selection]

for col in [col for col in df.columns if col.startswith("params_")]:
    name = col.replace("params_", "")
    plt.figure()
    params = np.array(
        [space.remap_if_needed({name: val})[name] for val in df[col].values]
    )

    plt.plot(
        params[selection],
        sel_values,
        linestyle="",
        marker="x",
    )
    plt.xlabel(name)
    plt.ylabel("value")
    plt.title(name)
    plt.show()

    if save:
        plt.savefig(fig_dir / f"{name}_values.png")
