# -*- coding: utf-8 -*-
import math
import os
from itertools import islice, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .ba_model import GPUConsAvgScoreBAModel
from .cache import mark_dependency
from .hyperopt import HyperoptSpace, get_space_template
from .metrics import Metrics
from .model_params import get_model_params
from .space import generate_space_spec


@mark_dependency
def get_sse_func(
    *,
    space,
    score_model,
):
    def sse_func_with_discrete(theta, data=None):
        scores = score_model.get_scores(
            requested=(Metrics.ARCSINH_SSE,),
            **space.inv_map_float_to_0_1(
                dict(zip(space.continuous_param_names, theta))
            ),
        )

        sse = float(scores["arcsinh_sse"])

        if np.isnan(sse):
            return int(1e10)

        return sse

    return sse_func_with_discrete


@mark_dependency
def iter_opt_methods(indices=None, release_gpu_model=False):
    df, method_iter = get_model_params(
        record_dir=Path(os.environ["EPHEMERAL"]) / "opt_record",
        progress=False,
        verbose=False,
    )

    if indices is None:
        indices = (0, None)

    for i, (
        dryness_method,
        fuel_build_up_method,
        df_sel,
        min_index,
        min_loss,
        params,
        exp_name,
        exp_key,
    ) in enumerate(islice(method_iter(), *indices)):
        assert int(params["include_temperature"]) == 1

        score_model = GPUConsAvgScoreBAModel(_uncached_data=False, **params)

        space = HyperoptSpace(
            generate_space_spec(
                get_space_template(
                    dryness_method=score_model.dryness_method,
                    fuel_build_up_method=score_model.fuel_build_up_method,
                    include_temperature=score_model.include_temperature,
                )
            )
        )

        x0_0_1 = space.map_float_to_0_1(
            {key: params[key] for key in space.continuous_param_names}
        )

        sse_func = get_sse_func(space=space, score_model=score_model)

        yield dict(
            score_model=score_model, space=space, x0_0_1=x0_0_1, sse_func=sse_func
        )

        if release_gpu_model:
            score_model.release()
            del score_model


def plot_chains(*, chains, names, figsize=(24, 14)):
    n_params = chains.shape[1]

    ncols = math.ceil(math.sqrt(n_params))
    nrows = math.ceil(n_params / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)

    for chain, ax, name in zip(chains.T, axes.ravel(), names):
        ax.plot(chain)
        ax.set_ylabel(name)

    for ax in axes.ravel()[len(names) :]:
        ax.set_axis_off()

    for j in range(ncols):
        axes[-1, j].set_xlabel("iteration")

    plt.tight_layout()

    return fig


def plot_pairwise_grid(*, chains, names, save_dir, nbins=12, filename="pairs"):
    N = len(names)
    assert chains.shape[1] == N

    bin_edges = np.linspace(np.min(chains), np.max(chains), nbins)

    plt.ioff()
    fig, axes = plt.subplots(N, N, figsize=(0.2 + (0.52 * N), 0.6 * N))

    for row, col in tqdm(list(product(range(N), range(N))), desc="Pairs"):
        ax = axes[row, col]

        ax.set_xticks([])
        ax.set_yticks([])

        if col == 0:
            # Row label.
            ax.set_ylabel(names[row], rotation="horizontal", ha="right", va="center")
        if row == col:
            # Column label.
            ax.xaxis.set_label_position("top")
            ax.set_xlabel(names[col], rotation=45, ha="left", va="bottom")

        if col < row:
            if chains.shape[0] < 100:
                ax.plot(
                    chains[:, col], chains[:, row], linestyle="", marker="o", alpha=0.7
                )
                ax.grid(linestyle="--", color=tuple([0.6] * 3), alpha=0.5)
            else:
                ax.hexbin(
                    chains[:, col],
                    chains[:, row],
                    gridsize=(6, 6),
                    bins="log",
                    extent=(0, 1, 0, 1),
                )
        elif col == row:
            # Plot histogram - this is *not* normalised in any way. Its height is
            # designed to fill up as much of the [0, 1] space as possible.
            n, bins = np.histogram(chains[:, row], bin_edges)
            ax.bar(bins[:-1], n / np.max(n), align="edge", width=np.diff(bins))
        elif col > row:
            ax.set_axis_off()
            continue

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("square")

    plt.tight_layout(pad=1, h_pad=0.1, w_pad=0.1, rect=(0, 0, 1, 1))

    fig.savefig(save_dir / filename)

    plt.close(fig)
