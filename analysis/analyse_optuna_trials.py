#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna

if __name__ == "__main__":
    exp_key = "optuna2"
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
    if save:
        plt.savefig(fig_dir / "min_loss_evolution.png")

        with (fig_dir / "argmin.json").open("w") as f:
            json.dump(study.best_params, f, indent=4)
