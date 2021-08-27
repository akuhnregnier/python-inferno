#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hyperopt.mongoexp import MongoTrials

if __name__ == "__main__":
    exp_key = "exp20_shrink"
    fig_dir = Path(f"~/tmp/trials_{exp_key}").expanduser()
    fig_dir.mkdir(exist_ok=True, parents=False)

    trials = MongoTrials(
        "mongo://maritimus.webredirect.org:1234/ba/jobs", exp_key=exp_key
    )

    losses = np.array(trials.losses())

    min_loss = np.min([l for l in losses if l is not None])

    raw_selection = np.array([l is not None and l < 1000 for l in trials.losses()])
    selection = np.array(
        [l is not None and l < min_loss * 1.15 for l in trials.losses()]
    )

    for name, vals in trials.vals.items():
        vals = np.array(vals)

        plt.figure()
        plt.title(name)
        plt.plot(
            vals[selection], losses[selection], marker="o", alpha=0.2, linestyle=""
        )
        plt.xlabel(name)
        plt.ylabel("loss")

        new_dir = fig_dir / "selection"
        new_dir.mkdir(exist_ok=True)
        plt.savefig(new_dir / f"{name}.png")

        plt.figure()
        plt.title(name)
        plt.plot(
            vals[raw_selection],
            losses[raw_selection],
            marker="o",
            alpha=0.2,
            linestyle="",
        )
        plt.xlabel(name)
        plt.ylabel("loss")

        new_dir = fig_dir / "raw"
        new_dir.mkdir(exist_ok=True)
        plt.savefig(new_dir / f"{name}.png")

    # Calculate and plot the minimum loss up to and including the current iteration.
    min_losses = []
    for loss in trials.losses():
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
    plt.savefig(fig_dir / "min_loss_evolution.png")

    with (fig_dir / "argmin.json").open("w") as f:
        json.dump(trials.argmin, f, indent=4)
