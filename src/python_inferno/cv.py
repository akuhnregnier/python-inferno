# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .cache import cache, mark_dependency
from .configuration import land_pts


@mark_dependency
@cache
def get_ba_cv_splits(gfed_ba):
    cat_boundaries = [-1, 1e-4, 1e-2, 0.1, 1]
    ba_categories = np.digitize(gfed_ba.ravel(), cat_boundaries)
    # The non-overlapping groups for sampling are simply the grid point indices.
    groups = (
        np.zeros_like(gfed_ba, dtype=np.int64)
        + np.arange(land_pts).reshape(1, land_pts)
    ).ravel()

    cv = StratifiedGroupKFold(n_splits=5, random_state=0, shuffle=True)

    test_grid_map = {}

    train_grids = []
    test_grids = []

    for (split_i, (train_ids, test_ids)) in enumerate(
        cv.split(np.ones((groups.size, 1)), ba_categories, groups)
    ):
        train_grid = []
        test_grid = []

        # For each grid cell, 12 months should be present.
        assert train_ids.size % 12 == 0
        assert test_ids.size % 12 == 0

        for land_pt in range(land_pts):
            s = np.sum((test_ids % land_pts) == land_pt)

            if s == 12:
                test_grid_map[land_pt] = split_i
                test_grid.append(land_pt)
            elif s == 0:
                # Points must be either in the test or train set.
                train_grid.append(land_pt)
            else:
                raise ValueError

        train_grids.append(np.array(train_grid))
        test_grids.append(np.array(test_grid))

    train_grids = tuple(train_grids)
    test_grids = tuple(test_grids)

    return train_grids, test_grids, test_grid_map
