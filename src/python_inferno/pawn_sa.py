# -*- coding: utf-8 -*-
from functools import partial

from SALib.analyze import pawn

from .cache import cache, mark_dependency
from .sensitivity_analysis import (
    GPUSampleMetricSA,
    analyse_sis,
    batched_sis_calc,
    get_gpu_sample_metric_sa_class_dependencies,
)

analyse_sis = partial(
    analyse_sis,
    method_name="PAWN",
    method_keys=["minimum", "mean", "median", "maximum", "CV"],
    sort_key="median",
)


class GPUPAWNSA(GPUSampleMetricSA):

    salib_analyze_func = pawn.pawn_analyze

    @mark_dependency
    def gen_problem(self, *, all_names):
        return dict(
            names=all_names,
            num_vars=len(all_names),
        )

    @mark_dependency
    def calc_sa_indices(
        self,
        *,
        problem,
        X,
        Y,
        verbose,
    ):
        return type(self).salib_analyze_func(
            problem=problem,
            X=X,
            Y=Y,
            S=10,
            print_to_console=False,
            seed=0,
        )


# NOTE Due to complications with cpp dependencies, cache should be reset manually when
# needed.
@cache(
    dependencies=get_gpu_sample_metric_sa_class_dependencies(GPUPAWNSA),
    ignore=["n_batches"],
)
def pawn_sis_calc(
    *,
    n_batches=1,
    params,
    dryness_method,
    fuel_build_up_method,
    N,
    chain_data,
    chain_names,
    land_points,
):
    return batched_sis_calc(
        n_batches=n_batches,
        params=params,
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        N=N,
        chain_data=chain_data,
        chain_names=chain_names,
        land_points=land_points,
        sa_class=GPUPAWNSA,
    )
