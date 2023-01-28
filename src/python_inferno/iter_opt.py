# -*- coding: utf-8 -*-
"""
For PFT-dependent parameters, require either 1 or 3 values (but 1, 2, or 3 distinct
values).
For only 1 distinct value, the single provided value is automatically expanded like
0.5 -> [0.5, 0.5, 0.5].
If only 2 distinct values are required, one value must reference another,
e.g. [0.5, 0.6, Index_0] where the 3rd value will be equal to the first.

Since there are 3 PFT groups, let i (0-indexed) be in (0, 1, 2).

Parameters are categorised according to which 'model version' they are part of,
selected by the 'switches' 'dryness_method', 'include_temperature', and
'fuel_build_up_method'.

All parameters except for `overall_scale` and `crop_f` 'belong' to such a category.
These are always included and do not correspond to any sigmoid.

Within each model version, there may be several mutually exclusive groups of
parameters, e.g. `dry_day_X` for `dryness_method=1`, vs. `dry_bal_X`, `rain_f`,
`vpd_f` for `dryness_method=2`.

Let weights be denoted by X_w[i]. If X_w[i] = 0, corresponding sigmoid parameters[i]
'X_s[i]', (e.g. X_shape[i]) will be meaningless and should therefore not be optimised.

Each sigmoid parameter X_s[i] can be shared amongst multiple pft groups as stated
above.

Let there be 'K' parameters in each set of sigmoid parameters (e.g. shape, location,
etc...).

Total number of parameters for each scenario, 'P'.

For each sigmoid, the hierarchy of complexity is as follows (from lowest to highest,
i.e. fewest to most parameters):
    -> X_w = 0 for all PFTs:
        - all X_s do not matter and are therefore not optimised (P=0)
    -> X_w = 1 for all PFTs:
        - 1 set of X_s across all PFTs (P=K)
    -> Mixture of X_w = 0, = 1 for PFTs
        - 1--2 sets of X_s
    -> 1 >= X_w > 0:
        - 1 value per parameter, shared across PFTs
        - 2 value(s) per parameter, may or may not be shared across PFTs
        - 3 values per parameter, never shared across PFTs

For 3 categories (i.e. `N_pft_groups`), can have (complexity):
    XXX (1)
    XYY (2)
    XXY (2)
    XYX (2)
    XYZ (3)

Thus, for each X_s, progression of complexity could be:
    XXX -> XYY -> XXY -> XYX -> XYZ
    - The 3 middle combinations have the equivalent complexity (number of parameters),
      but will yield different performance. Thus, when going from 1 to 2 parameters
      for this particular X_s[i], all 3 combinations are tried.

Combined progression:
    X_w = 0 for all i, i.e. 000
    X_w = 111, 101, 011, 110, 001, 010, 100 with all X_s shared.
    X_w = 111, 101, 011, 110 with some X_s shared, complexity depends on
        specific parameter sharing.
    X_w = 111 with no X_s shared, i.e. most general case with highest complexity.

If a parameter is not optimised, fix at value determined using optimisation with all
parameters?

Always optimised:
 - overall_scale

Conditional:
 - crop_f
 - average_samples ?? (or fixed from optimised full run?)

"""
import os
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from itertools import combinations, product
from multiprocessing import Process, Queue
from operator import add
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hyperopt import hp
from loguru import logger
from tqdm import tqdm

from .ba_model import ARCSINH_FACTOR, GPUConsAvgBAModel
from .cache import IN_STORE, NotCachedError, cache, mark_dependency
from .cv import get_ba_cv_splits
from .data import get_processed_climatological_data
from .hyperopt import HyperoptSpace, get_space_template
from .metrics import Metrics
from .plotting import get_plot_name_map_total, plot_label_case
from .space import generate_space_spec
from .space_opt import calculate_split_loss, space_opt, split_min_space_opt

Mode = Enum("Mode", ["R", "N"])


def plot_text(x: str) -> str:
    return plot_label_case(get_plot_name_map_total().get(x, x))


@mark_dependency
def get_always_optimised():
    return set(("overall_scale",))


ALWAYS_OPTIMISED = get_always_optimised()


@mark_dependency
def get_ignored():
    return set(
        (
            "overall_scale",
            "average_samples",
            "rain_f",
            "vpd_f",
            "fuel_build_up_n_samples",
            "litter_tc",
            "leaf_f",
        )
    )


IGNORED = get_ignored()


@mark_dependency
def reorder(s):
    assert isinstance(s, str)
    assert len(s) >= 2
    assert set("XYZ").issuperset(set(s))

    remaining_indices = list(range(len(s)))

    new_s = [""] * len(s)

    count = 0
    while remaining_indices:
        for i, letter in enumerate(s):
            if i not in remaining_indices:
                continue

            for j, l2 in enumerate(s):
                if l2 != letter:
                    continue

                new_s[j] = str(count)
                remaining_indices.remove(j)
            count += 1

    new_s = "".join(new_s)
    for old, new in zip("012"[: len(s)], "XYZ"):
        new_s = new_s.replace(old, new)
    return new_s


@mark_dependency
def match(spec, pattern):
    assert isinstance(pattern, str)
    assert {"0", "1"}.issuperset(set(pattern))
    assert len(spec) == len(pattern)

    for s, p in zip(spec, pattern):
        if p == "1" and (s == 1 or isinstance(s, tuple)):
            if isinstance(s, tuple) and len(s) != 3:
                raise ValueError
        elif p == "0" and s == 0:
            pass
        else:
            return False
    return True


@mark_dependency
def any_match(spec, patterns):
    for pattern in patterns:
        if match(spec, pattern):
            return True
    return False


@mark_dependency
def get_sigmoid_names(prefix):
    return tuple(f"{prefix}_{suffix}" for suffix in ("factor", "centre", "shape"))


@mark_dependency
def get_weight_sigmoid_names_map(*, dryness_method, fuel_build_up_method):
    out = {
        "fapar_weight": get_sigmoid_names("fapar"),
        "temperature_weight": get_sigmoid_names("temperature"),
    }
    if dryness_method == 1:
        out["dryness_weight"] = get_sigmoid_names("dry_day")
    elif dryness_method == 2:
        # "rain_f", "vpd_f" ignored (kept fixed).
        out["dryness_weight"] = get_sigmoid_names("dry_bal")

    if fuel_build_up_method == 1:
        # "fuel_build_up_n_samples" kept fixed.
        out["fuel_weight"] = get_sigmoid_names("fuel_build_up")
    elif fuel_build_up_method == 2:
        # "litter_tc", "leaf_f" kept fixed.
        out["fuel_weight"] = get_sigmoid_names("litter_pool")

    for vals in out.values():
        assert not any(v in IGNORED for v in vals)

    return out


@mark_dependency
def format_configurations(iterator_func):
    def get_formatted_configs(*args, **kwargs):
        for config, count in iterator_func(*args, **kwargs):
            # 'Format' config.
            for key, spec in config.items():
                if (
                    isinstance(spec, tuple)
                    and len(spec) == 2
                    and isinstance(spec[0], str)
                    and len(spec[0]) == 3
                    and set("XYZ").issuperset(spec[0])
                ):
                    # Reorganise in order 'X', 'Y', 'Z'.
                    unique_letters = set(spec[0])
                    if len(unique_letters) == 1:
                        # Nothing to format.
                        continue
                    else:
                        new_spec = list(spec)
                        new_spec[0] = reorder(spec[0])
                        config[key] = tuple(new_spec)
                elif "_weight" in key:
                    if not isinstance(spec, tuple):
                        continue
                    opt_keys = [s[2] for s in spec if isinstance(s, tuple)]

                    if not opt_keys or set(opt_keys) == {"X"}:
                        pass
                    else:
                        # Potentially need to reorder keys.
                        ordered_opt_keys = list(reorder("".join(opt_keys)))

                        new_spec = list(spec)

                        for i, s in enumerate(spec):
                            if isinstance(s, tuple):
                                new_vals = list(new_spec[i])
                                new_vals[2] = ordered_opt_keys.pop(0)
                                new_spec[i] = tuple(new_vals)

                        # Check that we have consumed all of them as expected. There
                        # should be a 1-1 mapping from unordered to ordered keys.
                        assert not ordered_opt_keys

                        config[key] = tuple(new_spec)

            yield config, count

    return get_formatted_configs


@mark_dependency
def _next_configurations_iter(start):
    """Get next possible configurations.

    Args:
        start (dict): Dict of parameters and their optimisation criterion.

    Yields:
        iterator of (dict, int): New specs and number of new parameters (diff).


    """
    start = deepcopy(start)
    dryness_method = start.pop("dryness_method")
    fuel_build_up_method = start.pop("fuel_build_up_method")
    include_temperature = start.pop("include_temperature")

    assert include_temperature

    # Most complex template possible.
    space_template = get_space_template(
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=include_temperature,
    )
    weight_to_sigmoid_names_map = get_weight_sigmoid_names_map(
        dryness_method=dryness_method, fuel_build_up_method=fuel_build_up_method
    )
    # Invert the above.
    sigmoid_to_weight_names_map = {}
    for weight_name, sigmoid_names in weight_to_sigmoid_names_map.items():
        # NOTE Assuming only 3 parameters here, fixed.
        assert len(sigmoid_names) == 3
        for sigmoid_name in sigmoid_names:
            sigmoid_to_weight_names_map[sigmoid_name] = weight_name

    for key, spec in start.items():
        if key in IGNORED:
            # NOTE These values are not optimised.
            continue

        if key == "crop_f" and spec == 0:
            new = deepcopy(start)
            new[key] = (0, 1)  # Enable optimisation.
            yield (new, 1)
        elif "_weight" in key and spec == 0:
            # 'Kick-start' optimisation of this sigmoid group by assigning some
            # non-zero weights.
            for weights in (
                (1, 1, 1),
                (1, 0, 1),
                (1, 1, 0),
                (0, 1, 1),
                (0, 0, 1),
                (0, 1, 0),
                (1, 0, 0),
            ):
                new = deepcopy(start)
                new[key] = weights
                # All X_s shared.
                for name in weight_to_sigmoid_names_map[key]:
                    new[name] = ("XXX", space_template[name][1][0])

                yield (new, len(weight_to_sigmoid_names_map[key]))
        elif "_weight" in key and any_match(
            spec,
            (
                # Weights (1 denotes either literal 1 or (0, 1) optimisation).
                # 100 -> 110 (or any permutation thereof, e.g. 001 -> 011).
                # For 3 X_s
                # X--, X--, X--
                # -> XY- (new param), XX- (replicate), XX- (replicate)
                # -> XX- (replicate), XY- (new param), XX- (replicate)
                # -> XX- (replicate), XX- (replicate), XY- (new param)
                # i.e. per row only 1 new param (new param (N), appears once)
                # -> N, R, R
                # -> R, N, R
                # -> R, R, N
                # For columns mapping of original state to choices:
                # - X-- -> XY- (N) or XX- (R)
                "001",
                "010",
                "100",
            ),
        ):
            # Add another '1' weight to a previously '0' index.
            zero_indices = [i for i, val in enumerate(spec) if val == 0]

            sigmoid_names = weight_to_sigmoid_names_map[key]

            for name in sigmoid_names:
                assert start[name][0] == "XXX"

            for new_index in zero_indices:
                for modes in product(
                    (Mode.R, Mode.N),
                    repeat=3,
                ):
                    if modes.count(Mode.N) != 1:
                        # Only allow 1 new parameter.
                        continue

                    new = deepcopy(start)

                    # Add new weight.
                    new_weights_spec = list(new[key])
                    new_weights_spec[new_index] = 1
                    new[key] = tuple(new_weights_spec)

                    for name, mode in zip(sigmoid_names, modes):
                        new_vals = list(new[name][0])
                        if mode == Mode.R:
                            new_vals[new_index] = "X"
                        elif mode == Mode.N:
                            new_vals[new_index] = "Y"

                        new_spec = list(new[name])
                        new_spec[0] = "".join(new_vals)
                        new[name] = tuple(new_spec)

                    yield (new, 1)
        elif "_weight" in key and any_match(
            spec,
            (
                # Weights (1 denotes either literal 1 or (0, 1) optimisation).
                # 110 -> 111 (or any permutation thereof, e.g. 101 -> 111).
                # For 3 X_s
                # XX-, XY-, XX-
                # -> XXY (new param), XYX / XYY (2 choices), XXX (replicate)
                # -> XXX (replicate), XYZ (new param), XXX (replicate)
                # -> XXX (replicate), XYX / XYY (2 choices), XXY (new param)
                # i.e. per row only 1 new param (new param (N), appears once)
                # -> N, A/B (i.e. R), R
                # -> R,     N       , R
                # -> R, A/B (i.e. R), N
                # For columns mapping of original state to choices:
                # - XX- -> XXY (N) or XXX (R)
                # - XY- -> XYX / XYY (A/B) or XYZ (N)
                "110",
                "011",
                "101",
            ),
        ):
            # Add another '1' weight to a previously '0' index.
            zero_indices = [i for i, val in enumerate(spec) if val == 0]
            valid_indices = [i for i, val in enumerate(spec) if val != 0]
            assert len(zero_indices) == 1
            assert len(zero_indices) + len(valid_indices) == 3

            new_index = zero_indices[0]

            sigmoid_names = weight_to_sigmoid_names_map[key]

            # Verification of expected share strings.
            for name in sigmoid_names:
                share_str = start[name][0]
                assert (
                    (share_str.count("X") == 2 and share_str.count("Y") == 1)
                    or (share_str.count("X") == 1 and share_str.count("Y") == 2)
                    or (share_str.count("X") == 3 and share_str.count("Y") == 0)
                ) and {"X", "Y"}.issuperset(set(share_str))

            for modes in product(
                (Mode.R, Mode.N),
                repeat=3,
            ):
                if modes.count(Mode.N) != 1:
                    # Only allow 1 new parameter.
                    continue

                new_vals_dict = {}

                for name, mode in zip(sigmoid_names, modes):
                    new_vals = list(deepcopy(start[name])[0])

                    valid_share = "".join(sorted([new_vals[i] for i in valid_indices]))
                    assert valid_share in ("XX", "XY")

                    if mode == Mode.R:
                        if valid_share == "XX":
                            new_vals[new_index] = "X"
                        elif valid_share == "XY":
                            # NOTE 2 options here, will be iterated over.
                            new_vals[new_index] = ["X", "Y"]
                    elif mode == Mode.N:
                        if valid_share == "XX":
                            new_vals[new_index] = "Y"
                        elif valid_share == "XY":
                            new_vals[new_index] = "Z"

                    new_vals_dict[name] = new_vals

                stop = False

                while not stop:
                    new = deepcopy(start)

                    # Add new weight.
                    new_weights_spec = list(new[key])
                    new_weights_spec[new_index] = 1
                    new[key] = tuple(new_weights_spec)

                    for name, new_vals in new_vals_dict.items():
                        valid_vals = list(new[name][0])

                        new_item = new_vals[new_index]

                        if isinstance(new_item, list):
                            if new_item:
                                valid_vals[new_index] = new_item.pop()
                            else:
                                stop = True
                                break
                        else:
                            # Simply copy if it is a string.
                            valid_vals[new_index] = new_item

                        new_spec = list(new[name])
                        new_spec[0] = "".join(valid_vals)
                        new[name] = tuple(new_spec)
                    else:
                        # If no break encountered.
                        yield (new, 1)

                    if not any(
                        isinstance(new_vals[new_index], list)
                        for new_vals in new_vals_dict.values()
                    ):
                        break

        if (
            "_weight" in key
            and isinstance(spec, tuple)
            and set((0, 1)).issuperset(set(spec))
        ):
            # Optimise either a single weight parameter, a pair of weight parameters,
            # or all three simultaneously.

            weight_indices = [i for i, s in enumerate(spec) if s == 1]

            for indices in list(
                reduce(
                    add, (list(combinations(weight_indices, n)) for n in range(1, 4))
                )
            ):
                new = deepcopy(start)

                new_spec = list(spec)
                for i in indices:
                    new_spec[i] = (0, 1, "X")

                new[key] = tuple(new_spec)
                yield (new, 1)
        elif "_weight" in key and isinstance(spec, tuple) and spec.count(0) == 1:
            # Optimise individual weights parameters.

            existing_opt_keys_unique = {s[2] for s in spec if isinstance(s, tuple)}

            if existing_opt_keys_unique == set("XY"):
                # Would already be as complicated as possible here.
                pass
            else:
                assert {"X"}.issuperset(existing_opt_keys_unique)
                valid_keys = set("XY")
                possible_keys = valid_keys - existing_opt_keys_unique
                assert possible_keys
                # NOTE New key for a previously '1' weight that wasn't being optimised.
                new_opt_key = sorted(list(possible_keys))[0]

                for i, weight_val in enumerate(spec):
                    if weight_val == 1:
                        # Make this subject to optimisation.
                        new = deepcopy(start)

                        new_spec = list(spec)
                        new_spec[i] = (0, 1, new_opt_key)

                        new[key] = tuple(new_spec)

                        yield (new, 1)
        elif "_weight" in key and isinstance(spec, tuple) and spec.count(0) == 0:
            # Optimise individual weights parameters.

            existing_opt_keys_unique = {s[2] for s in spec if isinstance(s, tuple)}

            if existing_opt_keys_unique == set("XYZ"):
                # Would already be as complicated as possible here.
                pass
            else:
                assert set("XY").issuperset(existing_opt_keys_unique)
                valid_keys = set("XYZ")
                possible_keys = valid_keys - existing_opt_keys_unique
                assert possible_keys
                # NOTE New key for a previously '1' weight that wasn't being optimised.
                new_opt_key = sorted(list(possible_keys))[0]

                # Optimise single individual weights.
                for i, weight_val in enumerate(spec):
                    if weight_val == 1:
                        # Make this subject to optimisation.
                        new = deepcopy(start)

                        new_spec = list(spec)
                        new_spec[i] = (0, 1, new_opt_key)

                        new[key] = tuple(new_spec)

                        yield (new, 1)

                if spec.count(1) == 2:
                    # Optimise pair.
                    new = deepcopy(start)

                    new_spec = list(spec)

                    for i, weight_val in enumerate(spec):
                        if weight_val == 1:
                            new_spec[i] = (0, 1, new_opt_key)

                    new[key] = tuple(new_spec)

                    yield (new, 1)

                existing_key_counts = {
                    key: sum(
                        (
                            weight_val[2] == key
                            if isinstance(weight_val, Iterable)
                            else False
                        )
                        for weight_val in spec
                    )
                    for key in existing_opt_keys_unique
                }

                max_count_key = max(
                    existing_key_counts, key=existing_key_counts.__getitem__
                )

                if existing_key_counts[max_count_key] >= 2:
                    # Scope for complexity by increasing weight optimisation
                    # granularity.
                    possible_indices = [
                        i
                        for i, weight_val in enumerate(spec)
                        if isinstance(weight_val, Iterable)
                        and weight_val[2] == max_count_key
                    ]
                    if len(possible_indices) == 2:
                        possible_indices = possible_indices[:1]

                    for index in possible_indices:
                        new_spec = list(spec)
                        new_spec[index] = (0, 1, new_opt_key)
                        new = deepcopy(start)
                        new[key] = tuple(new_spec)
                        yield (new, 1)

        if (
            key in sigmoid_to_weight_names_map
            and start[sigmoid_to_weight_names_map[key]] != 0
        ):
            weight_spec = start[sigmoid_to_weight_names_map[key]]
            # Locations where values actually matter.
            weight_indices = [i for i, s in enumerate(weight_spec) if s != 0]
            # Number of such locations.
            n_locs = len(weight_indices)

            existing_opt_keys = [spec[0][i] for i in weight_indices]
            existing_opt_keys_unique = set(existing_opt_keys)

            if (n_locs == 0) or (len(existing_opt_keys_unique) == n_locs):
                # Does not matter, or already as complicated as possible.
                pass
            else:
                # Scope for more complexity.
                assert set("XY").issuperset(existing_opt_keys_unique)
                valid_keys = set("XYZ")
                possible_keys = valid_keys - existing_opt_keys_unique
                assert possible_keys
                # NOTE New key.
                new_opt_key = sorted(list(possible_keys))[0]
                assert new_opt_key in ("Y", "Z")

                # For 'XXX' `max_existing_n=3`.
                # For 'XXY', `max_existing_n=2`, and only a single index should be tried.
                # For 2 locs, only option is 'XX' -> `max_existing_n=2`.
                # Pairs of indices are never attempted, since changing pairs is
                # equivalent to changing single indices,
                # e.g. XXX -> XYY is equivalent to XXX -> YXX -> XYY without loss of
                # generality.
                max_existing_n = max(
                    existing_opt_keys.count(opt_key)
                    for opt_key in existing_opt_keys_unique
                )
                assert max_existing_n > 1

                target_opt_key = None
                for opt_key in existing_opt_keys_unique:
                    if existing_opt_keys.count(opt_key) == max_existing_n:
                        target_opt_key = opt_key

                assert target_opt_key

                assert "".join(sorted(existing_opt_keys)) in (
                    "XXX",
                    "XXY",
                    "XYY",
                    "XX",
                    "YY",
                )

                target_indices = [
                    weight_indices[i]
                    for i, opt_key in enumerate(existing_opt_keys)
                    if opt_key == target_opt_key
                ]

                if max_existing_n == 2:
                    # Sufficient to try 1 index here, since result will be equivalent,
                    # e.g. XYY -> XYZ ~= XYY -> XZY -> XYZ wlog.
                    target_indices = target_indices[:1]

                for index in target_indices:
                    new = deepcopy(start)

                    new_share_spec = list(spec[0])

                    new_share_spec[index] = new_opt_key

                    new_spec = list(spec)
                    new_spec[0] = "".join(new_share_spec)
                    new[key] = tuple(new_spec)

                    yield (new, 1)


@format_configurations
def next_configurations_iter(start):
    return _next_configurations_iter(start)


@mark_dependency
def configuration_to_hyperopt_space_spec(configuration):
    space_spec = {}
    constants = {}

    for key, spec in configuration.items():
        if key == "crop_f":
            if spec in (0, 1):
                constants[key] = spec
            else:
                assert len(spec) == 2
                space_spec[key] = (hp.uniform, *spec)
        elif spec in (0, 1):
            constants[key] = spec
        elif (
            isinstance(spec, tuple)
            and len(spec) == 2
            and isinstance(spec[0], str)
            and len(spec[0]) == 3
            and set("XYZ").issuperset(spec[0])
        ):
            assert len(spec[1]) == 2
            space_spec.update(
                generate_space_spec({key: (spec[0], [spec[1]], hp.uniform)})
            )
        elif "_weight" in key:
            assert len(spec) == 3

            prev_weights = {}

            for weight_spec_val, suffix in zip(spec, ["", "2", "3"]):
                new_key = f"{key}{suffix}"
                if weight_spec_val in (0, 1):
                    constants[new_key] = weight_spec_val
                else:
                    assert weight_spec_val[:2] == (0, 1)
                    assert weight_spec_val[2] in ("X", "Y", "Z")

                    # Share weights as appropriate (see `generate_space_spec`).
                    if weight_spec_val[2] in prev_weights:
                        space_spec[new_key] = prev_weights[weight_spec_val[2]]
                    else:
                        space_spec[new_key] = (hp.uniform, 0, 1)
                        prev_weights[weight_spec_val[2]] = new_key

    # NOTE: `space_spec` defines variables that are optimised, `constants` are not.
    return space_spec, constants


@mark_dependency
def get_next_x0(*, new_space, x0_dict, prev_spec, prev_constants):
    x0 = []
    for key in new_space.continuous_param_names:
        if key in x0_dict:
            x0.append(x0_dict[key])
        else:
            # Add missing entries to replicate the exact previous minimum.
            if key in prev_spec:
                # String mapping.
                x0.append(x0_dict[prev_spec[key]])
            elif key in prev_constants:
                # String mapping.
                x0.append(prev_constants[key])
            elif key.strip("2") in prev_spec:
                # Transition akin to XXX -> XYZ
                x0.append(x0_dict[key.strip("2")])
            elif key.strip("3") in prev_spec:
                # Transition akin to XXX -> XYZ
                x0.append(x0_dict[key.strip("3")])
            else:
                raise ValueError(f"key '{key}' not found.")

    return x0


def mp_space_opt(*, q, **kwargs):
    q.put(space_opt(**kwargs))


@cache(
    dependencies=[
        _next_configurations_iter,
        any_match,
        calculate_split_loss,
        configuration_to_hyperopt_space_spec,
        format_configurations,
        generate_space_spec,
        get_always_optimised,
        get_ignored,
        get_next_x0,
        get_processed_climatological_data,
        get_sigmoid_names,
        get_space_template,
        get_weight_sigmoid_names_map,
        match,
        reorder,
        space_opt,
        split_min_space_opt,
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
                opt_record_dir=Path(os.environ["EPHEMERAL"]) / "newrun_iter_opt",
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
            plot_text(category),
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
    fig.savefig(save_dir / f"{save_key}")
    plt.close()
