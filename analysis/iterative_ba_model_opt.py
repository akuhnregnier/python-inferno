#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm

from python_inferno.cache import IN_STORE, NotCachedError, cache
from python_inferno.hyperopt import HyperoptSpace, get_space_template
from python_inferno.iter_opt import (
    ALWAYS_OPTIMISED,
    IGNORED,
    _next_configurations_iter,
    any_match,
    configuration_to_hyperopt_space_spec,
    format_configurations,
    get_always_optimised,
    get_ignored,
    get_sigmoid_names,
    get_weight_sigmoid_names_map,
    match,
    next_configurations_iter,
    reorder,
)
from python_inferno.model_params import get_model_params
from python_inferno.space import generate_space_spec
from python_inferno.space_opt import space_opt


def mp_space_opt(*, q, **kwargs):
    q.put(space_opt(**kwargs))


@cache(
    dependencies=[
        _next_configurations_iter,
        any_match,
        configuration_to_hyperopt_space_spec,
        format_configurations,
        generate_space_spec,
        get_always_optimised,
        get_ignored,
        get_sigmoid_names,
        get_space_template,
        get_weight_sigmoid_names_map,
        match,
        reorder,
        space_opt,
    ]
)
def iterative_ba_model_opt(
    *,
    params,
    maxiter=60,
    niter_success=15,
):
    dryness_method = int(params["dryness_method"])
    fuel_build_up_method = int(params["fuel_build_up_method"])
    include_temperature = int(params["include_temperature"])

    # NOTE Full space template.
    space_template = get_space_template(
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=include_temperature,
    )

    discrete_param_names = HyperoptSpace(
        generate_space_spec(space_template)
    ).discrete_param_names

    # NOTE Constant.
    defaults = dict(
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=include_temperature,
    )
    discrete_params = {}

    # Most basic config possible.
    # Keys specify which parameters are potentially subject to optimisation. All other
    # keys will be taken from the optimal configuration as set out in `params`.
    start_config = defaults.copy()

    base_spec = {}

    for key in space_template:
        if key in ALWAYS_OPTIMISED:
            base_spec.update(generate_space_spec({key: space_template[key]}))
        elif key in IGNORED:
            if key in discrete_param_names:
                # NOTE Also constant.
                for pft_key in (f"{key}{suffix}" for suffix in ("", "2", "3")):
                    if pft_key in params:
                        discrete_params[pft_key] = params[pft_key]

                assert key in discrete_params, "At least 1 key should be present"
            else:
                raise ValueError(key)
        else:
            start_config[key] = 0

    q = Queue()

    steps_prog = tqdm(desc="Steps", position=0)

    results = {}
    init_n_params = 1  # TODO - This initial value should depends on `base_spec`.

    steps = 0

    while True:
        local_best_config = defaultdict(lambda: None)
        local_best_loss = defaultdict(lambda: np.inf)

        for (configuration, n_new) in tqdm(
            list(next_configurations_iter(start_config)),
            desc="Step-configs",
            position=1,
        ):
            n_params = init_n_params + n_new

            space_spec, constants = configuration_to_hyperopt_space_spec(configuration)
            space = HyperoptSpace({**base_spec, **space_spec})

            opt_kwargs = dict(
                space=space,
                dryness_method=dryness_method,
                fuel_build_up_method=fuel_build_up_method,
                include_temperature=include_temperature,
                discrete_params=discrete_params,
                opt_record_dir="test",
                defaults={**defaults, **constants},
                minimizer_options=dict(maxiter=maxiter),
                basinhopping_options=dict(niter_success=niter_success),
                verbose=False,
                _uncached_data=False,
            )

            is_cached = False
            try:
                if space_opt.check_in_store(**opt_kwargs) is IN_STORE:
                    is_cached = True
            except NotCachedError:
                pass

            if is_cached:
                loss = space_opt(**opt_kwargs)
            else:
                # Avoid memory leaks by running each trial in a new process.
                p = Process(target=mp_space_opt, kwargs={"q": q, **opt_kwargs})
                p.start()
                loss = q.get()
                p.join()

            logger.info(f"loss: {loss}")
            if loss < local_best_loss[n_params]:
                logger.info(f"New best loss: {loss}.")
                local_best_loss[n_params] = loss
                local_best_config[n_params] = configuration

                if n_params not in results:
                    # New `n_params`.
                    results[n_params] = (loss, configuration)
                else:
                    # Check the old loss.
                    if loss < results[n_params][0]:
                        # Only update if the new loss is lower.
                        results[n_params] = (loss, configuration)

            steps_prog.refresh()

        if not local_best_config:
            # No configurations were explored.
            break

        best_n_params = min(
            (loss, n_params) for (n_params, loss) in local_best_loss.items()
        )[1]

        start_config = {**start_config, **local_best_config[best_n_params]}
        init_n_params = best_n_params

        steps += 1
        steps_prog.update()

    q.close()
    q.join_thread()

    return results


@dataclass
class Box:
    color: str

    def plot(self, *, ax, xy, width, height, **kwargs):
        rect = plt.Rectangle(xy, width, height, **kwargs)
        ax.add_patch(rect)
        return rect


@dataclass
class BoxElement:
    colors: list[str | tuple]


def vis_result(
    *,
    result,
    dryness_method,
    fuel_build_up_method,
    save_key,
    save_dir,
):
    spec = result[1]

    weight_to_sigmoid_names_map = get_weight_sigmoid_names_map(
        dryness_method=dryness_method, fuel_build_up_method=fuel_build_up_method
    )
    # Invert the above.
    categories_map = dict(crop="crop_f") if "crop_f" in spec else {}
    for weight_name, sigmoid_names in weight_to_sigmoid_names_map.items():
        # NOTE Assuming only 3 parameters here, fixed.
        assert len(sigmoid_names) == 3
        categories_map[weight_name.replace("_weight", "")] = [weight_name] + list(
            sigmoid_names
        )

    source_colors = plt.get_cmap("tab10").colors

    cat_color_map = dict(
        temperature=source_colors[3],
        fapar=source_colors[2],
        dryness=source_colors[1],
        fuel=source_colors[0],
        crop=source_colors[4],
    )
    weight_color_map = {
        1: (0, 0, 0),
        0: (0.5, 0.5, 0.5),
    }
    opt_color_map = {
        "X": source_colors[6],
        "Y": source_colors[8],
        "Z": source_colors[9],
    }
    sigmoid_details = ("weight", "factor", "centre", "shape")

    boxes = {}

    for category in cat_color_map:
        keys = categories_map[category]
        if isinstance(keys, str):
            assert keys == "crop_f"
            if spec[keys] != (0, 1):
                continue

            assert spec[keys] == (0, 1)
            boxes[category] = {
                "box": Box(color=cat_color_map[category]),
                "elements": {
                    "crop": BoxElement(
                        colors=[opt_color_map["X"]],
                    )
                },
            }
            continue

        weight_key = keys[0]
        param_keys = keys[1:]
        weights = spec[weight_key]

        if weights == 0:
            continue

        boxes[category] = {
            "box": Box(color=cat_color_map[category]),
            "elements": {},
        }

        weight_colors = []

        for weight in weights:
            if weight in (0, 1):
                weight_colors.append(weight_color_map[weight])
            else:
                assert len(weight) == 3
                assert weight[:2] == (0, 1)
                weight_colors.append(opt_color_map[weight[2]])

        boxes[category]["elements"]["weight"] = BoxElement(colors=weight_colors)

        for key in param_keys:
            param_type = key.split("_")[-1]
            colors = []
            for weight, opt_key in zip(weights, spec[key][0]):
                if weight == 0:
                    colors.append(weight_color_map[0])
                else:
                    colors.append(opt_color_map[opt_key])

            boxes[category]["elements"][param_type] = BoxElement(colors=colors)

    fig, ax = plt.subplots(figsize=(13, 3))

    n_cat = len(cat_color_map)
    padding = 0.03
    total_pad = padding * (n_cat - 1)
    total_width = 1.0 - total_pad
    cat_width = total_width / n_cat

    cat_xs = [i * (padding + cat_width) for i in range(n_cat)]

    nested_padding = 0.015
    nested_total_x_pad = nested_padding * (len(opt_color_map) + 1)
    nested_width = (cat_width - nested_total_x_pad) / len(opt_color_map)
    nested_height = nested_width

    cat_height = (
        len(sigmoid_details) * nested_height
        + (len(sigmoid_details) + 1) * nested_padding
    )

    for (cat_i, (x, (category, box_dict))) in enumerate(zip(cat_xs, boxes.items())):
        ax.annotate(
            category,
            (x + cat_width / 2.0, cat_height + nested_padding),
            color="k",
            fontsize=14,
            ha="center",
            va="baseline",
        )

        crop_plot = set(box_dict["elements"]) == {"crop"}

        if crop_plot:
            x += cat_width / 2.0
            x -= nested_width / 2.0 + nested_padding

            y = cat_height / 2.0
            y -= nested_height / 2.0 + nested_padding

            width = nested_width + 2.0 * nested_padding
            height = nested_height + 2.0 * nested_padding
        else:
            y = 0
            width = cat_width
            height = cat_height

        # Outer box.
        rect = box_dict["box"].plot(
            ax=ax,
            xy=(x, y),
            width=width,
            height=height,
            color=box_dict["box"].color,
            zorder=1,
        )

        if crop_plot:
            element = box_dict["elements"]["crop"]
            assert len(element.colors) == 1
            color = element.colors[0]
            ax.add_patch(
                plt.Rectangle(
                    (x + nested_padding, y + nested_padding),
                    nested_width,
                    nested_height,
                    color=color,
                    zorder=2,
                )
            )
        else:
            # Nested details.
            for i, sigmoid_detail in enumerate(sigmoid_details):
                element = box_dict["elements"][sigmoid_detail]

                nested_y = (
                    cat_height - (i + 1) * nested_padding - (i + 1) * nested_height
                )

                if cat_i == 0:
                    ax.annotate(
                        sigmoid_detail,
                        (-padding, nested_y + nested_height / 2.0),
                        color="k",
                        fontsize=14,
                        ha="right",
                        va="center",
                    )

                for j, color in enumerate(element.colors):
                    nested_x = x + (j + 1) * nested_padding + j * nested_width

                    ax.add_patch(
                        plt.Rectangle(
                            (nested_x, nested_y),
                            nested_width,
                            nested_height,
                            color=color,
                            zorder=2,
                        )
                    )

    legend_x = 1.15

    opt_colors = list(opt_color_map[key] for key in opt_color_map)

    for (i, key) in enumerate(
        (
            weight_color_map[1],
            weight_color_map[0],
            opt_colors,
        )
    ):
        if key == weight_color_map[1]:
            text = "weight=1"
        elif key == weight_color_map[0]:
            text = "weight=0"
        elif key == opt_colors:
            text = "optimised"
        else:
            raise ValueError

        y = cat_height - (i + 1) * nested_padding - (i + 1) * nested_height

        if isinstance(key, list):
            x = legend_x - 3 * (nested_width + nested_padding)
            for j, k in enumerate(key):
                x += nested_width + nested_padding
                ax.add_patch(
                    plt.Rectangle(
                        (x, y), nested_width, nested_height, color=k, zorder=2
                    )
                )
        else:
            ax.add_patch(
                plt.Rectangle(
                    (legend_x, y), nested_width, nested_height, color=key, zorder=2
                )
            )

        ax.annotate(
            text,
            (legend_x + nested_width + padding, y + nested_height / 2.0),
            color="k",
            fontsize=14,
            ha="left",
            va="center",
        )

    ax.add_patch(
        plt.Rectangle(
            (
                legend_x - 2 * (nested_width + nested_padding) - padding,
                y - padding,
            ),
            3 * (nested_width + nested_padding) + padding + 0.18,
            3 * nested_height + 2 * nested_padding + 2 * padding,
            fill=False,
        )
    )

    ax.axis("equal")
    ax.set_axis_off()
    ax.set_ylim(0, cat_height * 1.1)
    fig.savefig(save_dir / f"{save_key}.png")
    plt.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    save_dir = Path("~/tmp/iter-opt-models").expanduser()
    save_dir.mkdir(parents=False, exist_ok=True)

    df, method_iter = get_model_params(
        record_dir=Path(os.environ["EPHEMERAL"]) / "opt_record",
        progress=False,
        verbose=False,
    )

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 8))

    for i, method_data in enumerate(method_iter()):
        full_opt_loss = method_data[4]
        params = method_data[5]
        method_name = method_data[6]

        ax = axes.ravel()[i]
        ax.set_title(method_name)

        method_dir = save_dir / method_name
        method_dir.mkdir(parents=False, exist_ok=True)

        for key, marker, opt_kwargs in (
            ("30,5", "+", dict(maxiter=30, niter_success=5)),
            ("60,15", "x", dict(maxiter=60, niter_success=15)),
        ):
            results = iterative_ba_model_opt(params=params, **opt_kwargs)

            opt_dir = method_dir / key.replace(",", "_")
            opt_dir.mkdir(parents=False, exist_ok=True)

            for n, result in tqdm(results.items(), desc="Plotting model vis"):
                vis_result(
                    result=result,
                    dryness_method=int(params["dryness_method"]),
                    fuel_build_up_method=int(params["fuel_build_up_method"]),
                    save_key=str(n),
                    save_dir=opt_dir,
                )

            n_params, losses = zip(
                *((n_params, float(loss)) for (n_params, (loss, _)) in results.items())
            )
            ax.plot(
                n_params,
                losses,
                marker=marker,
                linestyle="",
                label=key,
                ms=8,
            )
        ax.hlines(full_opt_loss, 0, max(n_params), colors="g")
        ax.legend()
        ax.grid()

        if i in (0, 2):
            ax.set_ylabel("performance (loss)")
        if i in (2, 3):
            ax.set_xlabel("nr. of optimised parameters")

    fig.savefig(save_dir / "iter_model_perf.png")
    plt.close(fig)
