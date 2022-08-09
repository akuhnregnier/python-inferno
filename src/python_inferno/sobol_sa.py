# -*- coding: utf-8 -*-
from collections import defaultdict
from copy import deepcopy
from functools import partial
from operator import itemgetter

import numpy as np
from loguru import logger
from SALib.analyze import sobol
from SALib.sample import saltelli
from tqdm import tqdm

from .ba_model import BAModel
from .cache import cache, mark_dependency
from .configuration import Dims, N_pft_groups, npft
from .data import get_data_yearly_stdev
from .hyperopt import get_space_template
from .iter_opt import ALWAYS_OPTIMISED, IGNORED
from .mcmc import iter_opt_methods
from .metrics import calculate_phase
from .sensitivity_analysis import (
    GPUSAMixin,
    LandChecksFailed,
    SAMetric,
    SensitivityAnalysis,
    _sis_calc,
    analyse_sis,
    batched_sis_calc,
)
from .spotpy_mcmc import spotpy_dream
from .utils import map_keys


class SobolQCError(RuntimeError):
    """Raised when Sobol values fail checks."""


class SAParams:
    @mark_dependency
    def __init__(self, *, dryness_method, fuel_build_up_method, data_stats_params):
        self.initialise()

        # Param bound setup.
        space_template = get_space_template(
            dryness_method=dryness_method,
            fuel_build_up_method=fuel_build_up_method,
            include_temperature=1,
        )

        param_names = [
            key for key in space_template if key not in ALWAYS_OPTIMISED.union(IGNORED)
        ]

        key_mapping = {
            # TARGET: SOURCE
            "t1p5m_tile": "t1p5m",
            # NOTE - 'fapar_diag_pft' is the name used in `data_dict` and similar
            # structures which are passed around internally, named as such for
            # historical reasons. NPP is the variable that is actually being used.
            "fapar_diag_pft": "npp_pft",
            # NOTE See above. Also, distributions of NPP and antecedent NPP are
            # expected to be incredibly similar.
            "fuel_build_up": "npp_pft",
            "obs_pftcrop_1d": "obs_pftcrop_1d",
            "litter_pool": "litter_pool",
            "dry_days": "dry_days",
            "grouped_dry_bal": "grouped_dry_bal",
        }

        # Get parameter bounds from MCMC chains.

        method_index = {(1, 1): 0, (1, 2): 1, (2, 1): 2, (2, 2): 3}[
            (dryness_method, fuel_build_up_method)
        ]

        mcmc_kwargs = dict(
            iter_opt_index=method_index,
            # 1e5 - 15 mins with beta=0.05
            # 2e5 - 50 mins with beta=0.05 - due to decreasing acceptance rate over time!
            N=int(2e5),
            beta=0.05,
        )
        assert spotpy_dream.check_in_store(**mcmc_kwargs), str(mcmc_kwargs)
        dream_results = spotpy_dream(**mcmc_kwargs)
        results_df = dream_results["results_df"]
        space = dream_results["space"]

        # Analysis of results.
        names = space.continuous_param_names

        # Generate array of chain values, transform back to original ranges.
        chains = np.hstack(
            [
                space.inv_map_float_to_0_1(
                    {name: np.asarray(results_df[f"par{name}"])}
                )[name].reshape(-1, 1)
                for name in names
            ]
        )

        # Extract last N/2 values from the chains.
        chains = chains[chains.shape[0] // 2 :]

        # Get mean and standard deviations.
        means = np.mean(chains, axis=0)
        std = np.std(chains, axis=0)

        mins = np.min(chains, axis=0)
        maxs = np.max(chains, axis=0)

        min_bounds = np.clip(means - std, mins, maxs)
        max_bounds = np.clip(means + std, mins, maxs)

        self.param_bounds = {
            name: (min_bound, max_bound)
            for name, min_bound, max_bound in zip(names, min_bounds, max_bounds)
        }

        # NOTE Missing keys will be silently ignored. The keys present in
        # `data_yearly_stdev` are therefore determined by `data_stats_params`
        # indirectly.
        self.data_yearly_stdev = map_keys(
            get_data_yearly_stdev(params=data_stats_params), key_mapping
        )

    def initialise(self):
        self._variable_indices = []
        self._variable_bounds = []
        self._variable_names = []
        self._variable_groups = []
        self._dists = []

    @mark_dependency
    def add_param_vars(
        self,
        *,
        name,
        param_data,
    ):
        assert param_data.shape == (N_pft_groups,)

        slices = [(pft_i,) for pft_i in range(N_pft_groups)]
        dims = (Dims.PFT,)

        for (i, s) in enumerate(slices):
            i_name = f"{name}{i + 1}" if i > 0 else name

            bounds = self.param_bounds[i_name]

            if "_weight" in name or name == "crop_f":
                # NOTE Don't expect crop_f to actually show up here since it is not
                # expected to be subject to SA (but crop fraction variable is).
                bounds = list(np.clip(bounds, 0, 1))

            self._variable_indices.append({"name": name, "slice": s, "dims": dims})
            self._variable_bounds.append(bounds)
            self._dists.append("unif")
            self._variable_names.append(f"{name}_{i}")
            # Group all PFTs.
            self._variable_groups.append(name)

        self._check()

    @mark_dependency
    def add_vars(
        self,
        *,
        name: str,
        data: np.ndarray,
        land_index: int,
        max_pft=npft,
    ):
        if data.ndim == 3:
            slices = [
                (slice(None), pft_i, land_index)
                for pft_i in range(min(max_pft, data.shape[1]))
            ]
            dims = (Dims.TIME, Dims.PFT, Dims.LAND)
        elif data.ndim == 2:
            slices = [(slice(None), land_index)]
            dims = (Dims.TIME, Dims.LAND)
        else:
            raise ValueError(f"Unexpected shape '{data.shape}'.")

        for (i, s) in enumerate(slices):
            self._variable_indices.append({"name": name, "slice": s, "dims": dims})
            # Ignore the temporal index when selecting the stdev.
            # Specify (mean, stdev) for normal distribution - sampled values
            # correspond to an offset applied to the data spanning a year.
            self._variable_bounds.append((0, self.data_yearly_stdev[name][s[1:]]))
            self._dists.append("norm")
            self._variable_names.append(f"{name}_{i}")
            # Group all PFTs.
            self._variable_groups.append(name)

        self._check()

    @mark_dependency
    def _check(self):
        # Sanity check.
        assert (
            len(self._variable_indices)
            == len(self._variable_bounds)
            == len(self._variable_names)
            == len(self._variable_groups)
            == len(self._dists)
        )

    @mark_dependency
    def get_problem(self):
        return dict(
            groups=self._variable_groups,
            names=self._variable_names,
            num_vars=len(self._variable_names),
            bounds=self._variable_bounds,
            dists=self._dists,
        )

    @property
    def indices(self):
        return deepcopy(self._variable_indices)


class SobolSA(SensitivityAnalysis):
    @mark_dependency
    def __init__(self, *, exponent=6, **kwargs):
        super().__init__(**kwargs)
        self.exponent = exponent

        self.sa_params = SAParams(
            dryness_method=self.ba_model.dryness_method,
            fuel_build_up_method=self.ba_model.fuel_build_up_method,
            data_stats_params=self.data_stats_params,
        )

    @mark_dependency
    def get_sa_params(self, land_index: int):
        if np.any(self._checks_failed_mask[:, :, land_index]):
            raise LandChecksFailed(f"Some checks failed at location {land_index}.")

        self.sa_params.initialise()

        # For each of the variables to be investigated, define bounds associated with
        # the variable, its name, and the group it belongs to. For variables with
        # multiple PFTs, up to `n` natural PFTs are considered (grouped into the same
        # group).
        # NOTE Currently, bounds for each variable are defined according to its range
        # at the given land point / PFT, although this may change.
        for name in self.data_variables:
            if name in self.source_data:
                self.sa_params.add_vars(
                    name=name,
                    data=self.source_data[name],
                    land_index=land_index,
                )
            elif name in self.proc_params:
                self.sa_params.add_param_vars(
                    name=name,
                    param_data=self.proc_params[name],
                )
            else:
                raise ValueError(f"Unsupported name: '{name}'.")


class BAModelSobolSA(SobolSA):
    @mark_dependency
    def mod_model_data(self, *, name: str, s: tuple, vals):
        if name == "obs_pftcrop_1d":
            # Crop special case.
            self.ba_model.obs_pftcrop_1d[s] = vals
        else:
            self.ba_model.data_dict[name][s] = vals

    def sobol_sis(self, *, land_index, verbose=True):
        # Treat each land point individually.
        self.get_sa_params(land_index)
        problem = self.sa_params.get_problem()

        param_values = saltelli.sample(problem, 2**self.exponent)
        assert len(param_values.shape) == 2

        Y_asinh_nme = np.zeros(param_values.shape[0])
        Y_mpd = np.zeros(param_values.shape[0])

        for i in tqdm(
            range(param_values.shape[0]), desc="samples", disable=not verbose
        ):
            mod_params = deepcopy(self.params)

            # Insert the modified data.
            for j, index_data in enumerate(self.sa_params.indices):
                name, index_dims, index_slice = itemgetter("name", "dims", "slice")(
                    index_data
                )
                param_val = param_values[i, j]

                if name in self.source_data:
                    # For data, the sampled parameter values represent offsets.
                    if self.source_data[name].ndim == 3:
                        pft_i = index_slice[index_dims.index(Dims.PFT)]
                        vals = self.source_data[name][:, pft_i, land_index] + param_val
                    elif self.source_data[name].ndim == 2:
                        vals = self.source_data[name][:, land_index] + param_val
                    else:
                        raise RuntimeError

                    self.mod_model_data(name=name, s=index_slice, vals=vals)
                else:
                    if Dims.PFT in index_dims:
                        pft_i = index_slice[index_dims.index(Dims.PFT)]
                    else:
                        pft_i = None

                    mod_params[f"{name}{pft_i + 1}" if pft_i else name] = param_val

            # Run the model with the new variables.
            model_ba = self.ba_model.run(
                land_point=land_index,
                **mod_params,
            )["model_ba"]

            avg_model_ba = (
                np.einsum(
                    # (orig_time,), (orig_time, new_time) -> (new_time,)
                    "m,mn->n",
                    model_ba[:, land_index],
                    self.weights,
                    optimize=True,
                )
                / self.cum_weights
            ).reshape(12, 1)

            # Mean Error

            # NOTE No denominator is used here, since this would be constant across
            # all land indices anyway.
            Y_asinh_nme[i] = self.get_asinh_nme(
                avg_model_ba=avg_model_ba, land_index=land_index
            ).mean()

            # Phase Difference

            Y_mpd[i] = self.get_mpd(avg_model_ba=avg_model_ba, land_index=land_index)

        out = {}

        for metric_key, Y in tqdm(
            [
                (SAMetric.ARCSINH_NME, Y_asinh_nme),
                (SAMetric.MPD, Y_mpd),
            ],
            desc="Sensitivity metric",
            disable=not verbose,
        ):
            if np.any(np.isnan(Y) | np.isinf(Y)):
                continue

            try:
                sa_indices = sobol.sobol_analyze(
                    problem, Y, print_to_console=False, seed=0
                )
            except Exception as exc:
                logger.exception(exc)
            else:
                out[metric_key] = sa_indices

        return out


class GPUBAModelSobolSA(GPUSAMixin, SobolSA):
    @mark_dependency
    def sensitivity_analysis(self, *, land_index, verbose=True):
        # Treat each land point individually.
        self.get_sa_params(land_index)
        problem = self.sa_params.get_problem()

        param_values = saltelli.sample(problem, 2**self.exponent)
        assert len(param_values.shape) == 2

        if getattr(self, "total_n_samples", None) is None:
            self.total_n_samples = param_values.shape[0]
            self.Y_asinh_nme = np.zeros(self.total_n_samples) + np.nan
            self.Y_mpd = np.zeros(self.total_n_samples) + np.nan
        else:
            assert self.total_n_samples == param_values.shape[0]

        did_set_defaults = defaultdict(lambda: False)

        # Call in batches of `max_n_samples`
        for sl in tqdm(
            range(0, self.total_n_samples, self.gpu_sa.max_n_samples),
            desc="Batched samples",
            disable=not verbose,
        ):
            su = min(sl + self.gpu_sa.max_n_samples, self.total_n_samples)
            n_samples = su - sl

            # Modify data.
            names_to_set = set(self.gpu_sa.sample_data)
            for (index, index_data) in enumerate(self.sa_params.indices):
                self.set_index_data(
                    names_to_set=names_to_set,
                    param_values=param_values,
                    sl=sl,
                    su=su,
                    n_samples=n_samples,
                    index=index,
                    index_data=index_data,
                    land_index=land_index,
                )

            if not did_set_defaults[n_samples]:
                # Set remaining data using defaults.

                # NOTE This only has to be done once, since the default data will not
                # change between iterations, as it is constant (given the land index)!
                for name in names_to_set:
                    self.set_defaults(
                        name=name,
                        land_index=land_index,
                        n_samples=n_samples,
                    )

                did_set_defaults[n_samples] = True

            # Run the model with the new variables.
            model_ba = self.gpu_sa.run(nSamples=n_samples)

            avg_model_ba = self.sa_model_ba_cons_avg(model_ba)

            # Mean Error

            # NOTE No denominator is used here, since this would be constant across
            # all land indices anyway.
            self.Y_asinh_nme[sl:su] = self.get_asinh_nme(
                avg_model_ba=avg_model_ba, land_index=land_index
            )

            # Phase Difference

            self.Y_mpd[sl:su] = self.get_mpd(
                avg_model_ba=avg_model_ba, land_index=land_index
            )

        out = {}

        for metric_key, Y in tqdm(
            [
                (SAMetric.ARCSINH_NME, self.Y_asinh_nme),
                (SAMetric.MPD, self.Y_mpd),
            ],
            desc="Sensitivity metric",
            disable=not verbose,
        ):
            if np.any(np.isnan(Y) | np.isinf(Y)):
                continue

            try:
                sa_indices = sobol.sobol_analyze(
                    problem, Y, print_to_console=False, seed=0
                )
            except Exception as exc:
                logger.exception(exc)
            else:
                out[metric_key] = sa_indices

        return out


@mark_dependency
def sobol_qc(sobol_output):
    """Quality control for Sobol SA output."""

    def raise_error(details: str):
        raise SobolQCError(f"Sobol values failed checks: {details}")

    for key in ("S1", "ST"):
        vals = sobol_output[key]
        conf_vals = sobol_output[f"{key}_conf"]

        # Check effect values.
        if np.any(vals < -1e-1):
            raise_error(f"{key} {vals} below threshold.")
        if np.any(vals > 100):
            raise_error(f"{key} {vals} above threshold.")

        # Check conf.
        ratios = conf_vals / vals

        max_ratio = np.max(ratios)

        if max_ratio > 1:
            if max_ratio > 100:
                raise_error(f"{key} excessive conf ratio {max_ratio}.")
            else:
                logger.warning(f"{key} large conf ratio {max_ratio}.")


analyse_sis = partial(
    analyse_sis,
    method_name="Sobol",
    method_keys=["S1", "ST"],
    sort_key="ST",
    to_df_func=sobol.to_df,
)

# NOTE Due to complications with cpp dependencies, cache should be reset manually when
# needed.
@cache(
    dependencies=[
        BAModel.__init__,
        BAModel.process_kwargs,
        GPUBAModelSobolSA._get_data_shallow,
        GPUBAModelSobolSA.get_asinh_nme,
        GPUBAModelSobolSA.get_mpd,
        GPUBAModelSobolSA.get_sa_params,
        GPUBAModelSobolSA.sa_model_ba_cons_avg,
        GPUBAModelSobolSA.sensitivity_analysis,
        GPUBAModelSobolSA.set_defaults,
        GPUBAModelSobolSA.set_index_data,
        GPUSAMixin.__init__,
        SAParams.__init__,
        SAParams._check,
        SAParams.add_param_vars,
        SAParams.add_vars,
        SAParams.get_problem,
        SensitivityAnalysis.__init__,
        SobolSA.__init__,
        _sis_calc,
        batched_sis_calc,
        calculate_phase,
        get_data_yearly_stdev,
        get_space_template,
        iter_opt_methods,
        sobol.sobol_analyze,
        sobol_qc,
        spotpy_dream,
    ],
    ignore=["n_batches"],
)
def sobol_sis_calc(
    *,
    n_batches=1,
    land_points,
    params,
    exponent=8,
    data_variables=None,
    fuel_build_up_method,
    dryness_method,
):
    return batched_sis_calc(
        n_batches=n_batches,
        params=params,
        exponent=exponent,
        data_variables=data_variables,
        fuel_build_up_method=fuel_build_up_method,
        dryness_method=dryness_method,
        land_points=land_points,
        sa_class=GPUBAModelSobolSA,
    )
