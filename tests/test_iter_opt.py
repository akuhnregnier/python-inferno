# -*- coding: utf-8 -*-
from hyperopt import hp

from python_inferno.iter_opt import (
    configuration_to_hyperopt_space_spec,
    next_configurations_iter,
    reorder,
)


def test_reorder():
    assert reorder("XXX") == "XXX"
    assert reorder("XXX") == "XXX"
    assert reorder("YXY") == "XYX"
    assert reorder("ZXY") == "XYZ"
    assert reorder("ZYX") == "XYZ"
    assert reorder("ZXX") == "XYY"

    assert reorder("XX") == "XX"
    assert reorder("XY") == "XY"
    assert reorder("YX") == "XY"
    assert reorder("YY") == "XX"


def test_next_configurations_iter1():
    out = list(
        next_configurations_iter(
            dict(
                include_temperature=1,
                dryness_method=1,
                fuel_build_up_method=1,
                crop_f=0,
                dryness_weight=0,
            )
        )
    )

    # Ensure outputs are unique.
    assert len(set((tuple(o[0].items()), o[1]) for o in out)) == len(out)
    assert out == [
        ({"crop_f": (0, 1), "dryness_weight": 0}, 1),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.6)),
                "dry_day_shape": ("XXX", (0.1, 30.0)),
                "dryness_weight": (1, 1, 1),
            },
            3,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.6)),
                "dry_day_shape": ("XXX", (0.1, 30.0)),
                "dryness_weight": (1, 0, 1),
            },
            3,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.6)),
                "dry_day_shape": ("XXX", (0.1, 30.0)),
                "dryness_weight": (1, 1, 0),
            },
            3,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.6)),
                "dry_day_shape": ("XXX", (0.1, 30.0)),
                "dryness_weight": (0, 1, 1),
            },
            3,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.6)),
                "dry_day_shape": ("XXX", (0.1, 30.0)),
                "dryness_weight": (0, 0, 1),
            },
            3,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.6)),
                "dry_day_shape": ("XXX", (0.1, 30.0)),
                "dryness_weight": (0, 1, 0),
            },
            3,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.6)),
                "dry_day_shape": ("XXX", (0.1, 30.0)),
                "dryness_weight": (1, 0, 0),
            },
            3,
        ),
    ]


def test_next_configurations_iter2():
    out = list(
        next_configurations_iter(
            {
                "include_temperature": 1,
                "dryness_method": 1,
                "fuel_build_up_method": 1,
                "crop_f": (0, 1),
                "dryness_weight": 0,
            }
        )
    )

    # Ensure outputs are unique.
    assert len(set((tuple(o[0].items()), o[1]) for o in out)) == len(out)
    assert out == [
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.6)),
                "dry_day_shape": ("XXX", (0.1, 30.0)),
                "dryness_weight": (1, 1, 1),
            },
            3,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.6)),
                "dry_day_shape": ("XXX", (0.1, 30.0)),
                "dryness_weight": (1, 0, 1),
            },
            3,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.6)),
                "dry_day_shape": ("XXX", (0.1, 30.0)),
                "dryness_weight": (1, 1, 0),
            },
            3,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.6)),
                "dry_day_shape": ("XXX", (0.1, 30.0)),
                "dryness_weight": (0, 1, 1),
            },
            3,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.6)),
                "dry_day_shape": ("XXX", (0.1, 30.0)),
                "dryness_weight": (0, 0, 1),
            },
            3,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.6)),
                "dry_day_shape": ("XXX", (0.1, 30.0)),
                "dryness_weight": (0, 1, 0),
            },
            3,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.6)),
                "dry_day_shape": ("XXX", (0.1, 30.0)),
                "dryness_weight": (1, 0, 0),
            },
            3,
        ),
    ]


def test_next_configurations_iter3():
    out = list(
        next_configurations_iter(
            {
                "include_temperature": 1,
                "dryness_method": 1,
                "fuel_build_up_method": 1,
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 0, 0),
            }
        )
    )

    # Ensure outputs are unique.
    assert len(set((tuple(o[0].items()), o[1]) for o in out)) == len(out)
    assert out == [
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.2)),
                "dry_day_shape": ("XYX", (0.1, 20.0)),
                "dryness_weight": (1, 1, 0),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, 0),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XYX", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, 0),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.2)),
                "dry_day_shape": ("XXY", (0.1, 20.0)),
                "dryness_weight": (1, 0, 1),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 0, 1),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 0, 1),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), 0, 0),
            },
            1,
        ),
    ]


def test_next_configurations_iter4():
    out = list(
        next_configurations_iter(
            {
                "include_temperature": 1,
                "dryness_method": 1,
                "fuel_build_up_method": 1,
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (0, 1, 0),
            }
        )
    )

    # Ensure outputs are unique.
    assert len(set((tuple(o[0].items()), o[1]) for o in out)) == len(out)
    assert out == [
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": (1, 1, 0),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYY", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, 0),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XYY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, 0),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.2)),
                "dry_day_shape": ("XXY", (0.1, 20.0)),
                "dryness_weight": (0, 1, 1),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (0, 1, 1),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (0, 1, 1),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXX", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (0, (0, 1, "X"), 0),
            },
            1,
        ),
    ]


def test_next_configurations_iter5():
    out = list(
        next_configurations_iter(
            {
                "include_temperature": 1,
                "dryness_method": 1,
                "fuel_build_up_method": 1,
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (0, 1, (0, 1, "X")),
            }
        )
    )

    # Ensure outputs are unique.
    assert len(set((tuple(o[0].items()), o[1]) for o in out)) == len(out)
    assert out == [
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (0, 1, (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XYX", (0.1, 20.0)),
                "dryness_weight": (0, 1, (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XYX", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": (1, 1, (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": (1, 1, (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYY", (100, 200)),
                "dry_day_factor": ("XYX", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYY", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (0, (0, 1, "X"), (0, 1, "Y")),
            },
            1,
        ),
    ]


def test_next_configurations_iter6():
    out = list(
        next_configurations_iter(
            {
                "include_temperature": 1,
                "dryness_method": 1,
                "fuel_build_up_method": 1,
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, 1),
            }
        )
    )

    # Ensure outputs are unique.
    assert len(set((tuple(o[0].items()), o[1]) for o in out)) == len(out)
    assert out == [
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYY", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, 1),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, 1),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, 1),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, 1),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": (1, 1, 1),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XYX", (0.1, 20.0)),
                "dryness_weight": (1, 1, 1),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXY", (0.1, 20.0)),
                "dryness_weight": (1, 1, 1),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), 1, 1),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, (0, 1, "X"), 1),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "X"), 1),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), 1, (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, (0, 1, "X"), (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "X"), (0, 1, "X")),
            },
            1,
        ),
    ]


def test_next_configurations_iter7():
    out = list(
        next_configurations_iter(
            {
                "include_temperature": 1,
                "dryness_method": 1,
                "fuel_build_up_method": 1,
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, (0, 1, "X")),
            }
        )
    )

    # Ensure outputs are unique.
    assert len(set((tuple(o[0].items()), o[1]) for o in out)) == len(out)
    assert out == [
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYY", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, 1, (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": (1, 1, (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XYX", (0.1, 20.0)),
                "dryness_weight": (1, 1, (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXY", (0.1, 20.0)),
                "dryness_weight": (1, 1, (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), 1, (0, 1, "Y")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, (0, 1, "X"), (0, 1, "Y")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "X"), (0, 1, "Y")),
            },
            1,
        ),
    ]


def test_next_configurations_iter8():
    out = list(
        next_configurations_iter(
            {
                "include_temperature": 1,
                "dryness_method": 1,
                "fuel_build_up_method": 1,
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, (0, 1, "X"), (0, 1, "Y")),
            }
        )
    )

    # Ensure outputs are unique.
    assert len(set((tuple(o[0].items()), o[1]) for o in out)) == len(out)
    assert out == [
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYY", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, (0, 1, "X"), (0, 1, "Y")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, (0, 1, "X"), (0, 1, "Y")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, (0, 1, "X"), (0, 1, "Y")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": (1, (0, 1, "X"), (0, 1, "Y")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": (1, (0, 1, "X"), (0, 1, "Y")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XYX", (0.1, 20.0)),
                "dryness_weight": (1, (0, 1, "X"), (0, 1, "Y")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXY", (0.1, 20.0)),
                "dryness_weight": (1, (0, 1, "X"), (0, 1, "Y")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            },
            1,
        ),
    ]


def test_next_configurations_iter9():
    out = list(
        next_configurations_iter(
            {
                "include_temperature": 1,
                "dryness_method": 1,
                "fuel_build_up_method": 1,
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            }
        )
    )

    # Ensure outputs are unique.
    assert len(set((tuple(o[0].items()), o[1]) for o in out)) == len(out)
    assert out == [
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYY", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XXX", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XYX", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXX", (100, 200)),
                "dry_day_factor": ("XXY", (0.0, 0.2)),
                "dry_day_shape": ("XXY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            },
            1,
        ),
    ]


def test_next_configurations_iter10():
    out = list(
        next_configurations_iter(
            {
                "include_temperature": 1,
                "dryness_method": 1,
                "fuel_build_up_method": 1,
                "crop_f": 0,
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            }
        )
    )

    # Ensure outputs are unique.
    assert len(set((tuple(o[0].items()), o[1]) for o in out)) == len(out)
    assert out == [
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            },
            1,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XYZ", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            },
            1,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYZ", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            },
            1,
        ),
    ]


def test_next_configurations_iter11():
    out = list(
        next_configurations_iter(
            {
                "include_temperature": 1,
                "dryness_method": 1,
                "fuel_build_up_method": 1,
                "crop_f": 0,
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_centre": ("XXX", (0, 1)),
                "fuel_build_up_factor": ("XXX", (0, 1)),
                "fuel_build_up_shape": ("XXY", (0, 1)),
                "fuel_weight": (1, 0, 1),
            }
        )
    )

    # Ensure outputs are unique.
    assert len(set((tuple(o[0].items()), o[1]) for o in out)) == len(out)
    assert out == [
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_centre": ("XXX", (0, 1)),
                "fuel_build_up_factor": ("XXX", (0, 1)),
                "fuel_build_up_shape": ("XXY", (0, 1)),
                "fuel_weight": (1, 0, 1),
            },
            1,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XYZ", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_centre": ("XXX", (0, 1)),
                "fuel_build_up_factor": ("XXX", (0, 1)),
                "fuel_build_up_shape": ("XXY", (0, 1)),
                "fuel_weight": (1, 0, 1),
            },
            1,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYZ", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_centre": ("XXX", (0, 1)),
                "fuel_build_up_factor": ("XXX", (0, 1)),
                "fuel_build_up_shape": ("XXY", (0, 1)),
                "fuel_weight": (1, 0, 1),
            },
            1,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_centre": ("XYY", (0, 1)),
                "fuel_build_up_factor": ("XXX", (0, 1)),
                "fuel_build_up_shape": ("XXY", (0, 1)),
                "fuel_weight": (1, 0, 1),
            },
            1,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_centre": ("XXX", (0, 1)),
                "fuel_build_up_factor": ("XYY", (0, 1)),
                "fuel_build_up_shape": ("XXY", (0, 1)),
                "fuel_weight": (1, 0, 1),
            },
            1,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_centre": ("XXX", (0, 1)),
                "fuel_build_up_factor": ("XXX", (0, 1)),
                "fuel_build_up_shape": ("XYZ", (0, 1)),
                "fuel_weight": (1, 1, 1),
            },
            1,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_centre": ("XYX", (0, 1)),
                "fuel_build_up_factor": ("XXX", (0, 1)),
                "fuel_build_up_shape": ("XYY", (0, 1)),
                "fuel_weight": (1, 1, 1),
            },
            1,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_centre": ("XYX", (0, 1)),
                "fuel_build_up_factor": ("XXX", (0, 1)),
                "fuel_build_up_shape": ("XXY", (0, 1)),
                "fuel_weight": (1, 1, 1),
            },
            1,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_centre": ("XXX", (0, 1)),
                "fuel_build_up_factor": ("XYX", (0, 1)),
                "fuel_build_up_shape": ("XYY", (0, 1)),
                "fuel_weight": (1, 1, 1),
            },
            1,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_centre": ("XXX", (0, 1)),
                "fuel_build_up_factor": ("XYX", (0, 1)),
                "fuel_build_up_shape": ("XXY", (0, 1)),
                "fuel_weight": (1, 1, 1),
            },
            1,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_centre": ("XXX", (0, 1)),
                "fuel_build_up_factor": ("XXX", (0, 1)),
                "fuel_build_up_shape": ("XXY", (0, 1)),
                "fuel_weight": ((0, 1, "X"), 0, 1),
            },
            1,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_centre": ("XXX", (0, 1)),
                "fuel_build_up_factor": ("XXX", (0, 1)),
                "fuel_build_up_shape": ("XXY", (0, 1)),
                "fuel_weight": (1, 0, (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": 0,
                "dry_day_centre": ("XXY", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYY", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_centre": ("XXX", (0, 1)),
                "fuel_build_up_factor": ("XXX", (0, 1)),
                "fuel_build_up_shape": ("XXY", (0, 1)),
                "fuel_weight": ((0, 1, "X"), 0, (0, 1, "X")),
            },
            1,
        ),
    ]


def test_next_configurations_iter12():
    out = list(
        next_configurations_iter(
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYZ", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYZ", (0.1, 20.0)),
                "dryness_method": 1,
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fapar_centre": ("XYZ", (-0.1, 1.1)),
                "fapar_factor": ("XYZ", (-50, -1)),
                "fapar_shape": ("XYZ", (0.1, 20.0)),
                "fapar_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_method": 2,
                "fuel_weight": ((0, 1, "X"), (0, 1, "X"), (0, 1, "X")),
                "include_temperature": 1,
                "litter_pool_centre": ("XYZ", (10, 5000)),
                "litter_pool_factor": ("XYZ", (0.001, 0.1)),
                "litter_pool_shape": ("XYZ", (0.1, 20.0)),
                "temperature_centre": ("XYZ", (280, 320)),
                "temperature_factor": ("XYZ", (0.19, 0.3)),
                "temperature_shape": ("XYZ", (0.1, 20.0)),
                "temperature_weight": ((0, 1, "X"), (0, 1, "X"), (0, 1, "X")),
            }
        )
    )

    # Ensure outputs are unique.
    assert len(set((tuple(o[0].items()), o[1]) for o in out)) == len(out)

    assert out == [
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYZ", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYZ", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fapar_centre": ("XYZ", (-0.1, 1.1)),
                "fapar_factor": ("XYZ", (-50, -1)),
                "fapar_shape": ("XYZ", (0.1, 20.0)),
                "fapar_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Y")),
                "litter_pool_centre": ("XYZ", (10, 5000)),
                "litter_pool_factor": ("XYZ", (0.001, 0.1)),
                "litter_pool_shape": ("XYZ", (0.1, 20.0)),
                "temperature_centre": ("XYZ", (280, 320)),
                "temperature_factor": ("XYZ", (0.19, 0.3)),
                "temperature_shape": ("XYZ", (0.1, 20.0)),
                "temperature_weight": ((0, 1, "X"), (0, 1, "X"), (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYZ", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYZ", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fapar_centre": ("XYZ", (-0.1, 1.1)),
                "fapar_factor": ("XYZ", (-50, -1)),
                "fapar_shape": ("XYZ", (0.1, 20.0)),
                "fapar_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "X")),
                "litter_pool_centre": ("XYZ", (10, 5000)),
                "litter_pool_factor": ("XYZ", (0.001, 0.1)),
                "litter_pool_shape": ("XYZ", (0.1, 20.0)),
                "temperature_centre": ("XYZ", (280, 320)),
                "temperature_factor": ("XYZ", (0.19, 0.3)),
                "temperature_shape": ("XYZ", (0.1, 20.0)),
                "temperature_weight": ((0, 1, "X"), (0, 1, "X"), (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYZ", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYZ", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fapar_centre": ("XYZ", (-0.1, 1.1)),
                "fapar_factor": ("XYZ", (-50, -1)),
                "fapar_shape": ("XYZ", (0.1, 20.0)),
                "fapar_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_weight": ((0, 1, "X"), (0, 1, "X"), (0, 1, "Y")),
                "litter_pool_centre": ("XYZ", (10, 5000)),
                "litter_pool_factor": ("XYZ", (0.001, 0.1)),
                "litter_pool_shape": ("XYZ", (0.1, 20.0)),
                "temperature_centre": ("XYZ", (280, 320)),
                "temperature_factor": ("XYZ", (0.19, 0.3)),
                "temperature_shape": ("XYZ", (0.1, 20.0)),
                "temperature_weight": ((0, 1, "X"), (0, 1, "X"), (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYZ", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYZ", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fapar_centre": ("XYZ", (-0.1, 1.1)),
                "fapar_factor": ("XYZ", (-50, -1)),
                "fapar_shape": ("XYZ", (0.1, 20.0)),
                "fapar_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_weight": ((0, 1, "X"), (0, 1, "X"), (0, 1, "X")),
                "litter_pool_centre": ("XYZ", (10, 5000)),
                "litter_pool_factor": ("XYZ", (0.001, 0.1)),
                "litter_pool_shape": ("XYZ", (0.1, 20.0)),
                "temperature_centre": ("XYZ", (280, 320)),
                "temperature_factor": ("XYZ", (0.19, 0.3)),
                "temperature_shape": ("XYZ", (0.1, 20.0)),
                "temperature_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Y")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYZ", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYZ", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fapar_centre": ("XYZ", (-0.1, 1.1)),
                "fapar_factor": ("XYZ", (-50, -1)),
                "fapar_shape": ("XYZ", (0.1, 20.0)),
                "fapar_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_weight": ((0, 1, "X"), (0, 1, "X"), (0, 1, "X")),
                "litter_pool_centre": ("XYZ", (10, 5000)),
                "litter_pool_factor": ("XYZ", (0.001, 0.1)),
                "litter_pool_shape": ("XYZ", (0.1, 20.0)),
                "temperature_centre": ("XYZ", (280, 320)),
                "temperature_factor": ("XYZ", (0.19, 0.3)),
                "temperature_shape": ("XYZ", (0.1, 20.0)),
                "temperature_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "X")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYZ", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYZ", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fapar_centre": ("XYZ", (-0.1, 1.1)),
                "fapar_factor": ("XYZ", (-50, -1)),
                "fapar_shape": ("XYZ", (0.1, 20.0)),
                "fapar_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_weight": ((0, 1, "X"), (0, 1, "X"), (0, 1, "X")),
                "litter_pool_centre": ("XYZ", (10, 5000)),
                "litter_pool_factor": ("XYZ", (0.001, 0.1)),
                "litter_pool_shape": ("XYZ", (0.1, 20.0)),
                "temperature_centre": ("XYZ", (280, 320)),
                "temperature_factor": ("XYZ", (0.19, 0.3)),
                "temperature_shape": ("XYZ", (0.1, 20.0)),
                "temperature_weight": ((0, 1, "X"), (0, 1, "X"), (0, 1, "Y")),
            },
            1,
        ),
    ]


def test_next_configurations_iter13():
    out = list(
        next_configurations_iter(
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYZ", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYZ", (0.1, 20.0)),
                "dryness_method": 1,
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fapar_centre": ("XYZ", (-0.1, 1.1)),
                "fapar_factor": ("XYZ", (-50, -1)),
                "fapar_shape": ("XYZ", (0.1, 20.0)),
                "fapar_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_method": 2,
                "fuel_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "include_temperature": 1,
                "litter_pool_centre": ("XYZ", (10, 5000)),
                "litter_pool_factor": ("XYZ", (0.001, 0.1)),
                "litter_pool_shape": ("XYZ", (0.1, 20.0)),
                "temperature_centre": ("XYZ", (280, 320)),
                "temperature_factor": ("XYZ", (0.19, 0.3)),
                "temperature_shape": ("XYZ", (0.1, 20.0)),
                "temperature_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Y")),
            }
        )
    )

    # Ensure outputs are unique.
    assert len(set((tuple(o[0].items()), o[1]) for o in out)) == len(out)

    assert out == [
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYZ", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYZ", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fapar_centre": ("XYZ", (-0.1, 1.1)),
                "fapar_factor": ("XYZ", (-50, -1)),
                "fapar_shape": ("XYZ", (0.1, 20.0)),
                "fapar_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "litter_pool_centre": ("XYZ", (10, 5000)),
                "litter_pool_factor": ("XYZ", (0.001, 0.1)),
                "litter_pool_shape": ("XYZ", (0.1, 20.0)),
                "temperature_centre": ("XYZ", (280, 320)),
                "temperature_factor": ("XYZ", (0.19, 0.3)),
                "temperature_shape": ("XYZ", (0.1, 20.0)),
                "temperature_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            },
            1,
        )
    ]


def test_next_configurations_iter14():
    for opt_keys in ["XXY", "XYX", "XYY"]:
        out = list(
            next_configurations_iter(
                {
                    "crop_f": (0, 1),
                    "dry_day_centre": ("XYZ", (100, 200)),
                    "dry_day_factor": ("XYZ", (0.0, 0.2)),
                    "dry_day_shape": ("XYZ", (0.1, 20.0)),
                    "dryness_method": 1,
                    "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                    "fapar_centre": ("XYZ", (-0.1, 1.1)),
                    "fapar_factor": ("XYZ", (-50, -1)),
                    "fapar_shape": ("XYZ", (0.1, 20.0)),
                    "fapar_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                    "fuel_build_up_method": 2,
                    "fuel_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                    "include_temperature": 1,
                    "litter_pool_centre": (opt_keys, (10, 5000)),
                    "litter_pool_factor": ("XYZ", (0.001, 0.1)),
                    "litter_pool_shape": ("XYZ", (0.1, 20.0)),
                    "temperature_centre": ("XYZ", (280, 320)),
                    "temperature_factor": ("XYZ", (0.19, 0.3)),
                    "temperature_shape": ("XYZ", (0.1, 20.0)),
                    "temperature_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                }
            )
        )

        # Ensure outputs are unique.
        assert len(set((tuple(o[0].items()), o[1]) for o in out)) == len(out)

        assert out == [
            (
                {
                    "crop_f": (0, 1),
                    "dry_day_centre": ("XYZ", (100, 200)),
                    "dry_day_factor": ("XYZ", (0.0, 0.2)),
                    "dry_day_shape": ("XYZ", (0.1, 20.0)),
                    "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                    "fapar_centre": ("XYZ", (-0.1, 1.1)),
                    "fapar_factor": ("XYZ", (-50, -1)),
                    "fapar_shape": ("XYZ", (0.1, 20.0)),
                    "fapar_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                    "fuel_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                    "litter_pool_centre": ("XYZ", (10, 5000)),
                    "litter_pool_factor": ("XYZ", (0.001, 0.1)),
                    "litter_pool_shape": ("XYZ", (0.1, 20.0)),
                    "temperature_centre": ("XYZ", (280, 320)),
                    "temperature_factor": ("XYZ", (0.19, 0.3)),
                    "temperature_shape": ("XYZ", (0.1, 20.0)),
                    "temperature_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                },
                1,
            )
        ]


def test_next_configurations_iter15():
    out = list(
        next_configurations_iter(
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYZ", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYZ", (0.1, 20.0)),
                "dryness_method": 1,
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fapar_centre": ("XYZ", (-0.1, 1.1)),
                "fapar_factor": ("XYZ", (-50, -1)),
                "fapar_shape": ("XYZ", (0.1, 20.0)),
                "fapar_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_method": 2,
                "fuel_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "include_temperature": 1,
                "litter_pool_centre": ("XXX", (10, 5000)),
                "litter_pool_factor": ("XYZ", (0.001, 0.1)),
                "litter_pool_shape": ("XYZ", (0.1, 20.0)),
                "temperature_centre": ("XYZ", (280, 320)),
                "temperature_factor": ("XYZ", (0.19, 0.3)),
                "temperature_shape": ("XYZ", (0.1, 20.0)),
                "temperature_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            }
        )
    )

    # Ensure outputs are unique.
    assert len(set((tuple(o[0].items()), o[1]) for o in out)) == len(out)

    assert out == [
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYZ", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYZ", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fapar_centre": ("XYZ", (-0.1, 1.1)),
                "fapar_factor": ("XYZ", (-50, -1)),
                "fapar_shape": ("XYZ", (0.1, 20.0)),
                "fapar_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "litter_pool_centre": ("XYY", (10, 5000)),
                "litter_pool_factor": ("XYZ", (0.001, 0.1)),
                "litter_pool_shape": ("XYZ", (0.1, 20.0)),
                "temperature_centre": ("XYZ", (280, 320)),
                "temperature_factor": ("XYZ", (0.19, 0.3)),
                "temperature_shape": ("XYZ", (0.1, 20.0)),
                "temperature_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYZ", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYZ", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fapar_centre": ("XYZ", (-0.1, 1.1)),
                "fapar_factor": ("XYZ", (-50, -1)),
                "fapar_shape": ("XYZ", (0.1, 20.0)),
                "fapar_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "litter_pool_centre": ("XYX", (10, 5000)),
                "litter_pool_factor": ("XYZ", (0.001, 0.1)),
                "litter_pool_shape": ("XYZ", (0.1, 20.0)),
                "temperature_centre": ("XYZ", (280, 320)),
                "temperature_factor": ("XYZ", (0.19, 0.3)),
                "temperature_shape": ("XYZ", (0.1, 20.0)),
                "temperature_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            },
            1,
        ),
        (
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYZ", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYZ", (0.1, 20.0)),
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fapar_centre": ("XYZ", (-0.1, 1.1)),
                "fapar_factor": ("XYZ", (-50, -1)),
                "fapar_shape": ("XYZ", (0.1, 20.0)),
                "fapar_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "litter_pool_centre": ("XXY", (10, 5000)),
                "litter_pool_factor": ("XYZ", (0.001, 0.1)),
                "litter_pool_shape": ("XYZ", (0.1, 20.0)),
                "temperature_centre": ("XYZ", (280, 320)),
                "temperature_factor": ("XYZ", (0.19, 0.3)),
                "temperature_shape": ("XYZ", (0.1, 20.0)),
                "temperature_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            },
            1,
        ),
    ]


def test_next_configurations_iter16():
    out = list(
        next_configurations_iter(
            {
                "crop_f": (0, 1),
                "dry_day_centre": ("XYZ", (100, 200)),
                "dry_day_factor": ("XYZ", (0.0, 0.2)),
                "dry_day_shape": ("XYZ", (0.1, 20.0)),
                "dryness_method": 1,
                "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fapar_centre": ("XYZ", (-0.1, 1.1)),
                "fapar_factor": ("XYZ", (-50, -1)),
                "fapar_shape": ("XYZ", (0.1, 20.0)),
                "fapar_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "fuel_build_up_method": 2,
                "fuel_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
                "include_temperature": 1,
                "litter_pool_centre": ("XYZ", (10, 5000)),
                "litter_pool_factor": ("XYZ", (0.001, 0.1)),
                "litter_pool_shape": ("XYZ", (0.1, 20.0)),
                "temperature_centre": ("XYZ", (280, 320)),
                "temperature_factor": ("XYZ", (0.19, 0.3)),
                "temperature_shape": ("XYZ", (0.1, 20.0)),
                "temperature_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            }
        )
    )

    # Ensure outputs are unique.
    assert len(set((tuple(o[0].items()), o[1]) for o in out)) == len(out)

    assert out == []


def test_configuration_to_hyperopt_space_spec():
    out = configuration_to_hyperopt_space_spec(
        {
            "crop_f": 0,
            "dry_day_centre": ("XXY", (100, 200)),
            "dry_day_factor": ("XYZ", (0.0, 0.2)),
            "dry_day_shape": ("XYY", (0.1, 20.0)),
            "dryness_weight": ((0, 1, "X"), (0, 1, "Y"), (0, 1, "Z")),
            "fuel_build_up_centre": ("XXX", (0, 1)),
            "fuel_build_up_factor": ("XXX", (0, 1)),
            "fuel_build_up_shape": ("XXY", (0, 1)),
            "fuel_weight": ((0, 1, "X"), 0, (0, 1, "X")),
        }
    )

    assert out == (
        {
            "dry_day_centre": (hp.uniform, 100, 200),
            "dry_day_centre2": "dry_day_centre",
            "dry_day_centre3": (hp.uniform, 100, 200),
            "dry_day_factor": (hp.uniform, 0.0, 0.2),
            "dry_day_factor2": (hp.uniform, 0.0, 0.2),
            "dry_day_factor3": (hp.uniform, 0.0, 0.2),
            "dry_day_shape": (hp.uniform, 0.1, 20.0),
            "dry_day_shape2": (hp.uniform, 0.1, 20.0),
            "dry_day_shape3": "dry_day_shape2",
            "dryness_weight": (hp.uniform, 0, 1),
            "dryness_weight2": (hp.uniform, 0, 1),
            "dryness_weight3": (hp.uniform, 0, 1),
            "fuel_build_up_centre": (hp.uniform, 0, 1),
            "fuel_build_up_factor": (hp.uniform, 0, 1),
            "fuel_build_up_shape": (hp.uniform, 0, 1),
            "fuel_build_up_shape2": "fuel_build_up_shape",
            "fuel_build_up_shape3": (hp.uniform, 0, 1),
            "fuel_weight": (hp.uniform, 0, 1),
            "fuel_weight3": "fuel_weight",
        },
        {"crop_f": 0, "fuel_weight2": 0},
    )
