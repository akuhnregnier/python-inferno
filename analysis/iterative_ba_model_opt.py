#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm

from python_inferno.ba_model import ARCSINH_FACTOR, GPUConsAvgBAModel
from python_inferno.cache import IN_STORE, NotCachedError, cache
from python_inferno.cv import get_ba_cv_splits
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
    get_next_x0,
    get_sigmoid_names,
    get_weight_sigmoid_names_map,
    match,
    next_configurations_iter,
    reorder,
)
from python_inferno.metrics import Metrics
from python_inferno.model_params import get_model_params
from python_inferno.space import generate_space_spec
from python_inferno.space_opt import (
    calculate_split_loss,
    space_opt,
    split_min_space_opt,
)


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
        get_next_x0,
        get_sigmoid_names,
        get_space_template,
        get_weight_sigmoid_names_map,
        match,
        reorder,
        space_opt,
        split_min_space_opt,
        calculate_split_loss,
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

    # BA Model for AIC and CV calculations.
    ba_model = GPUConsAvgBAModel(
        _uncached_data=False,
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=include_temperature,
        **discrete_params,
    )

    # CV setup.
    assert np.ma.isMaskedArray(ba_model.mon_avg_gfed_ba_1d)
    assert not np.any(ba_model.mon_avg_gfed_ba_1d.mask)
    gfed_ba_1d = ba_model.mon_avg_gfed_ba_1d.data

    train_grids, test_grids, test_grid_map = get_ba_cv_splits(gfed_ba_1d)

    q = Queue()

    steps_prog = tqdm(desc="Steps", position=0)

    results = {}
    aic_results = {}
    cv_results = {}
    init_n_params = 1  # TODO - This initial value should depends on `base_spec`.

    steps = 0

    # x0 variables.
    x0_dict = None
    prev_spec = None
    prev_constants = None
    x0_dict_vals = {}
    prev_spec_vals = {}
    prev_constants_vals = {}
    # 'Real' (i.e. not [0, 1]) params for BA calculation.
    param_vals = {}
    space_vals = {}
    default_vals = {}
    constant_vals = {}
    x0_vals = {}

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

            if x0_dict is not None:
                assert prev_spec is not None
                assert prev_constants is not None
                x0 = get_next_x0(
                    new_space=space,
                    x0_dict=x0_dict,
                    prev_spec=prev_spec,
                    prev_constants=prev_constants,
                )
            else:
                x0 = None

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
                x0=x0,
                return_res=True,
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
                res = space_opt(**opt_kwargs)
            else:
                # Avoid memory leaks by running each trial in a new process.
                p = Process(target=mp_space_opt, kwargs={"q": q, **opt_kwargs})
                p.start()
                res = q.get()
                p.join()

            loss = res.fun

            logger.info(f"loss: {loss}")
            if loss < local_best_loss[n_params]:
                logger.info(f"New best loss: {loss}.")
                local_best_loss[n_params] = loss
                local_best_config[n_params] = configuration

                x0_vals[n_params] = res.x

                # x0 values are in [0, 1].
                x0_dict_vals[n_params] = {
                    key: val for key, val in zip(space.continuous_param_names, res.x)
                }
                prev_spec_vals[n_params] = space_spec
                prev_constants_vals[n_params] = constants
                space_vals[n_params] = space
                default_vals[n_params] = defaults
                constant_vals[n_params] = constants

                param_vals[n_params] = {
                    **space.inv_map_float_to_0_1(x0_dict_vals[n_params]),
                    **discrete_params,
                    **defaults,
                    **constants,
                }

                assert n_params == len(space.continuous_param_names)

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

        # Set up x0 variables for the next iteration / AIC / CV calculations.
        x0_dict = x0_dict_vals[best_n_params]
        prev_spec = prev_spec_vals[best_n_params]
        prev_constants = prev_constants_vals[best_n_params]
        params = param_vals[best_n_params]
        space = space_vals[best_n_params]
        defaults = default_vals[best_n_params]
        constants = constant_vals[best_n_params]
        x0 = x0_vals[best_n_params]

        # AIC calculation.

        try:
            # Calculate BA, scores.

            # ba_model = BAModel(**params)  # NOTE Further exceptions are raised here
            scores = ba_model.calc_scores(
                model_ba=ba_model.run(**params)["model_ba"],
                requested=(
                    Metrics.MPD,
                    Metrics.ARCSINH_NME,
                    Metrics.SSE,
                    Metrics.ARCSINH_SSE,
                ),
                n_params=len(discrete_params) + best_n_params,
            )["scores"]

            aic_results[best_n_params] = {
                "aic": scores["aic"],
                "arcsinh_aic": scores["arcsinh_aic"],
            }

            # NOTE Parameters only change minutely most of the time,
            # resulting in exactly 0 performance changes - local minimisation failure?
            # pprint(x0_dict)
        except Exception:
            logger.exception("Exception during AIC.")

        # CV.

        try:
            test_losses = []

            for train_grid, test_grid in zip(train_grids, test_grids):
                # Optimise model on training set using the previous minimum as a
                # starting point.
                cv_res = split_min_space_opt(
                    space=space,
                    dryness_method=dryness_method,
                    fuel_build_up_method=fuel_build_up_method,
                    include_temperature=include_temperature,
                    discrete_params=discrete_params,
                    train_grid=train_grid,
                    defaults={**defaults, **constants},
                    x0=x0,
                    minimizer_options=dict(maxiter=maxiter),
                )

                # x0 values are in [0, 1].
                cv_x0_dict = {
                    key: val for key, val in zip(space.continuous_param_names, cv_res.x)
                }

                # Test.
                test_gfed_ba_1d = np.ascontiguousarray(gfed_ba_1d[:, test_grid])
                test_arcsinh_y_true = np.arcsinh(
                    ARCSINH_FACTOR * test_gfed_ba_1d.ravel()
                )
                test_loss = calculate_split_loss(
                    pred_ba=ba_model.run(
                        **space.inv_map_float_to_0_1(cv_x0_dict),
                        **defaults,
                        **constants,
                    )["model_ba"],
                    point_grid=test_grid,
                    sel_true_1d=test_gfed_ba_1d,
                    sel_arcsinh_y_true=test_arcsinh_y_true,
                )

                test_losses.append(test_loss)

            cv_results[best_n_params] = np.mean(test_losses)
        except Exception:
            logger.exception("Exception during CV.")

        # Next loop setup.
        start_config = {**start_config, **local_best_config[best_n_params]}
        init_n_params = best_n_params

        steps += 1
        steps_prog.update()

    q.close()
    q.join_thread()

    ba_model.release()

    return results, aic_results, cv_results


@cache(dependencies=[get_weight_sigmoid_names_map])
def vis_result(
    *,
    result,
    dryness_method,
    fuel_build_up_method,
    save_key,
    save_dir,
):
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
    mpl.rc_file(Path(__file__).absolute().parent / "matplotlibrc")
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

        for key, marker, colors, opt_kwargs in (
            ("50,1", "^", ("C0", "C1"), dict(maxiter=50, niter_success=1)),
            ("1000,5", "x", ("C2", "C3"), dict(maxiter=1000, niter_success=5)),
        ):
            results, aic_results, cv_results = iterative_ba_model_opt(
                params=params, **opt_kwargs
            )

            n_params, aics, arcsinh_aics = zip(
                *(
                    (n_params, data["aic"], data["arcsinh_aic"])
                    for (n_params, data) in aic_results.items()
                )
            )
            aics = np.asarray(aics)
            arcsinh_aics = np.asarray(arcsinh_aics)

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
                *((n, float(loss)) for (n, (loss, _)) in results.items())
            )

            ms = 6
            aic_ms = 8

            ax.plot(
                n_params,
                losses,
                marker=marker,
                linestyle="",
                label=key,
                ms=ms,
                c=colors[0],
            )

            min_aic_i = np.argmin(aics)
            min_arcsinh_aic_i = np.argmin(arcsinh_aics)

            print("AIC indices:", min_aic_i, min_arcsinh_aic_i)

            ax.plot(
                n_params[min_aic_i],
                losses[min_aic_i],
                marker=marker,
                linestyle="",
                ms=aic_ms,
                c="r",
                zorder=4,
            )
            ax.plot(
                n_params[min_arcsinh_aic_i],
                losses[min_arcsinh_aic_i],
                marker=marker,
                linestyle="",
                ms=aic_ms,
                c="g",
                zorder=5,
            )

            # CV plotting.
            cv_n_params, cv_test_losses = zip(*cv_results.items())
            ax.plot(
                cv_n_params,
                cv_test_losses,
                marker=marker,
                linestyle="",
                label=f"{key} CV",
                ms=ms,
                c=colors[1],
            )

            print(method_name, key)
            for (n, (loss, _)) in results.items():
                if n in cv_results:
                    # print(n, loss - cv_results[n])
                    print(n, cv_results[n])

        ax.hlines(full_opt_loss, 0, max(n_params), colors="g")
        ax.legend()

        if i in (0, 2):
            ax.set_ylabel("performance (loss)")
        if i in (2, 3):
            ax.set_xlabel("nr. of optimised parameters")

    fig.savefig(save_dir / "iter_model_perf.png")
    plt.close(fig)