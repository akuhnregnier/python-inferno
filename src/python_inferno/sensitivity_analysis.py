# -*- coding: utf-8 -*-
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
from .configuration import Dims, N_pft_groups, land_pts, npft
from .data import get_data_yearly_stdev
from .hyperopt import get_space_template
from .iter_opt import ALWAYS_OPTIMISED, IGNORED
from .model_params import get_param_uncertainties
from .py_gpu_inferno import GPUSA


class LandChecksFailed(RuntimeError):
    """Raised when checks are failed at a given location."""


def map_keys(source_data, key_mapping):
    out = {}
    for target_key, source_key in key_mapping.items():
        if source_key in source_data:
            out[target_key] = source_data[source_key]
    return out


class SAParams:
    @mark_dependency
    def __init__(
        self, *, dryness_method, fuel_build_up_method, df_sel, data_stats_params
    ):
        self._variable_indices = []
        self._variable_bounds = []
        self._variable_names = []
        self._variable_groups = []
        self._dists = []

        # Param bound setup.
        space_template = get_space_template(
            dryness_method=dryness_method,
            fuel_build_up_method=fuel_build_up_method,
            include_temperature=1,
        )

        param_names = [
            key for key in space_template if key not in ALWAYS_OPTIMISED.union(IGNORED)
        ]

        # Filter columns.
        new_cols = [
            col
            for col in df_sel.columns
            if any(name_root in col for name_root in param_names)
        ] + ["loss"]

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

        self.param_uncertainties = get_param_uncertainties(df_sel=df_sel[new_cols])
        # NOTE Missing keys will be silently ignored. The keys present in
        # `data_yearly_stdev` are therefore determined by `data_stats_params`
        # indirectly.
        self.data_yearly_stdev = map_keys(
            get_data_yearly_stdev(params=data_stats_params), key_mapping
        )

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

            bounds = self.param_uncertainties[i_name]

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


class SensitivityAnalysis:
    @mark_dependency
    def __init__(
        self,
        *,
        params,
        data_variables=None,
        exponent=6,
        df_sel,
        fuel_build_up_method,
        dryness_method,
    ):
        self.params = params
        self.ba_model = BAModel(**self.params)
        self.exponent = exponent
        self.df_sel = df_sel
        self.data_stats_params = {}

        # NOTE Prevent side effects.
        self.ba_model.data_dict = deepcopy(self.ba_model.data_dict)
        self.ba_model.obs_pftcrop_1d = deepcopy(self.ba_model.obs_pftcrop_1d)

        self.proc_params = self.ba_model.process_kwargs(**self.params)
        self._checks_failed_mask = self.ba_model._get_checks_failed_mask()
        assert self._checks_failed_mask.ndim == 3
        assert self._checks_failed_mask.shape[1:] == (npft, land_pts)

        if data_variables is None:
            self.data_variables = [
                "t1p5m_tile",
                "fapar_diag_pft",
                "obs_pftcrop_1d",
            ]
            if fuel_build_up_method == 1:
                self.data_variables.append("fuel_build_up")
            elif fuel_build_up_method == 2:
                self.data_variables.append("litter_pool")

                self.data_stats_params["litter_pool"] = {
                    "litter_tc": self.ba_model.disc_params["litter_tc"],
                    "leaf_f": self.ba_model.disc_params["leaf_f"],
                }
            else:
                raise ValueError

            if dryness_method == 1:
                self.data_variables.append("dry_days")
            elif dryness_method == 2:
                self.data_variables.append("grouped_dry_bal")

                self.data_stats_params["grouped_dry_bal"] = {
                    "rain_f": self.ba_model.disc_params["rain_f"],
                    "vpd_f": self.ba_model.disc_params["vpd_f"],
                }
            else:
                raise ValueError
        else:
            self.data_variables = data_variables

        # Checks.
        if "fuel_build_up" in self.data_variables:
            assert fuel_build_up_method == 1

        if "litter_pool" in self.data_variables:
            assert fuel_build_up_method == 2

            self.data_stats_params["litter_pool"] = {
                "litter_tc": self.ba_model.disc_params["litter_tc"],
                "leaf_f": self.ba_model.disc_params["leaf_f"],
            }

        if "dry_days" in self.data_variables:
            assert dryness_method == 1

        if "grouped_dry_bal" in self.data_variables:
            assert dryness_method == 2

            self.data_stats_params["grouped_dry_bal"] = {
                "rain_f": self.ba_model.disc_params["rain_f"],
                "vpd_f": self.ba_model.disc_params["vpd_f"],
            }

        self.source_data = deepcopy(self._get_data_shallow())

        assert not set(self.source_data).intersection(set(self.proc_params))

        assert all(
            name in set(self.source_data).union(set(self.proc_params))
            for name in self.data_variables
        ), tuple(
            name
            for name in self.data_variables
            if name not in set(self.source_data).union(set(self.proc_params))
        )

    @mark_dependency
    def _get_data_shallow(self):
        return dict(
            **self.ba_model.data_dict,
            # Crop special case due to the way it is processed and stored.
            obs_pftcrop_1d=self.ba_model.obs_pftcrop_1d,
        )

    @mark_dependency
    def mod_model_data(self, *, name: str, s: tuple, vals):
        if name == "obs_pftcrop_1d":
            # Crop special case.
            self.ba_model.obs_pftcrop_1d[s] = vals
        else:
            self.ba_model.data_dict[name][s] = vals

    @mark_dependency
    def get_sa_params(self, land_index: int):
        if np.any(self._checks_failed_mask[:, :, land_index]):
            raise LandChecksFailed(f"Some checks failed at location {land_index}.")

        sa_params = SAParams(
            dryness_method=self.ba_model.dryness_method,
            fuel_build_up_method=self.ba_model.fuel_build_up_method,
            df_sel=self.df_sel,
            data_stats_params=self.data_stats_params,
        )
        # For each of the variables to be investigated, define bounds associated with
        # the variable, its name, and the group it belongs to. For variables with
        # multiple PFTs, up to `n` natural PFTs are considered (grouped into the same
        # group).
        # NOTE Currently, bounds for each variable are defined according to its range
        # at the given land point / PFT, although this may change.
        for name in self.data_variables:
            if name in self.source_data:
                sa_params.add_vars(
                    name=name,
                    data=self.source_data[name],
                    land_index=land_index,
                )
            elif name in self.proc_params:
                sa_params.add_param_vars(
                    name=name,
                    param_data=self.proc_params[name],
                )
            else:
                raise ValueError(f"Unsupported name: '{name}'.")

        return sa_params


class BAModelSensitivityAnalysis(SensitivityAnalysis):
    def sobol_sis(self, *, land_index, verbose=True):
        # Treat each land point individually.
        sa_params = self.get_sa_params(land_index)
        problem = sa_params.get_problem()

        param_values = saltelli.sample(problem, 2**self.exponent)
        assert len(param_values.shape) == 2

        Y = np.zeros(param_values.shape[0])

        for i in tqdm(
            range(param_values.shape[0]), desc="samples", disable=not verbose
        ):
            mod_params = deepcopy(self.params)

            # Insert the modified data.
            for j, index_data in enumerate(sa_params.indices):
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

            # NOTE Mean BA at the location used as a proxy.
            Y[i] = np.mean(model_ba[:, land_index])

        return sobol.analyze(problem, Y, print_to_console=False)


class GPUBAModelSensitivityAnalysis(SensitivityAnalysis):
    @mark_dependency
    def set_index_data(
        self,
        *,
        names_to_set,
        param_values,
        sl,
        su,
        n_samples,
        index,
        index_data,
        gpu_sa,
        land_index,
    ):
        # Broadcasting rules:
        # Collapsed dimensions - Dims.PFT, Dims.LAND.
        # Unlimited dimension - Dims.TIME.
        # In target data (`gpu_sa.sample_data`), Dims.SAMPLE dim is first,
        # resulting in the following possible target dims:
        # - (Dims.SAMPLE, Dims.TIME, Dims.PFT)  - data
        #  -> collapses to (Dims.SAMPLE, Dims.TIME)
        # - (Dims.SAMPLE, Dims.TIME)  - data
        #  -> collapses to (Dims.SAMPLE, Dims.TIME)
        # - (Dims.SAMPLE, Dims.PFT)  - parameters
        #  -> collapses to (Dims.SAMPLE,)
        # Source dims for `values` will be (Dims.SAMPLE,), with shape
        # (`n_samples`,).
        # Therefore, if Dims.TIME in source dims, need to reshape to add axis
        # after original Dims.SAMPLE axis.

        values = param_values[sl:su, index]

        name, index_dims, index_slice = itemgetter("name", "dims", "slice")(index_data)

        if Dims.TIME in index_dims:
            # Ensure it is indeed unlimited as stated above.
            assert index_slice[index_dims.index(Dims.TIME)] == slice(None)
            values = values.reshape(-1, 1)

        if name in self.source_data:
            assert name not in self.proc_params

            arr = self.source_data[name]
            sample_shape = gpu_sa.sample_data[name].shape
            sample_ndim = len(sample_shape)
            assert arr.ndim == sample_ndim

            if sample_ndim == 3:
                assert Dims.PFT in index_dims
                pft_i = index_slice[index_dims.index(Dims.PFT)]
                arr_s = (slice(None), pft_i, land_index)
            elif sample_ndim == 2:
                arr_s = (slice(None), land_index)

            offset_data = self.source_data[name][
                arr_s
                # Double check - the time dimension is `arr.shape[0]`.
            ].reshape(1, arr.shape[0])
        else:
            assert name in self.proc_params
            offset_data = None

        gpu_sa.set_sample_data(
            index_data=index_data,
            values=values,
            n_samples=n_samples,
            offset_data=offset_data,
        )

        if name in names_to_set:
            # Remove the name. Done only once like so, since we are
            # potentially setting data for the same 'name' multiple times for
            # different PFTs.
            names_to_set.remove(name)

    @mark_dependency
    def set_defaults(
        self,
        *,
        name,
        gpu_sa,
        land_index,
        n_samples,
    ):
        def gpu_set_sample_data(*, source_arr, s, dims):
            gpu_sa.set_sample_data(
                index_data={"name": name, "slice": s, "dims": dims},
                values=source_arr[s],
                n_samples=n_samples,
            )

        if name in self.proc_params:
            for pft_i in range(N_pft_groups):
                gpu_set_sample_data(
                    s=(pft_i,), dims=(Dims.PFT,), source_arr=self.proc_params[name]
                )
        else:
            arr = self.source_data[name]
            gpu_set_sample_data = partial(gpu_set_sample_data, source_arr=arr)

            sample_shape = gpu_sa.sample_data[name].shape
            sample_ndim = len(sample_shape)
            assert arr.ndim == sample_ndim

            if sample_ndim == 3:
                for pft_i in range(sample_shape[-1]):
                    gpu_set_sample_data(
                        s=(slice(None), pft_i, land_index),
                        dims=(Dims.TIME, Dims.PFT, Dims.LAND),
                    )
            elif sample_ndim == 2:
                gpu_set_sample_data(
                    s=(slice(None), land_index), dims=(Dims.TIME, Dims.LAND)
                )
            else:
                raise ValueError(f"Unknown sample shape '{sample_shape}'.")

    @mark_dependency
    def sobol_sis(self, *, land_index, verbose=True):
        # Treat each land point individually.
        sa_params = self.get_sa_params(land_index)
        problem = sa_params.get_problem()

        param_values = saltelli.sample(problem, 2**self.exponent)
        assert len(param_values.shape) == 2

        total_n_samples = param_values.shape[0]
        Y = np.zeros(total_n_samples)

        gpu_sa = GPUSA(
            Nt=self.ba_model.Nt,
            drynessMethod=self.ba_model.dryness_method,
            fuelBuildUpMethod=self.ba_model.fuel_build_up_method,
            includeTemperature=self.ba_model.include_temperature,
            overallScale=self.params["overall_scale"],
            crop_f=self.params["crop_f"],
        )

        # Call in batches of `max_n_samples`
        for sl in tqdm(
            range(0, total_n_samples, gpu_sa.max_n_samples),
            desc="Batched samples",
            disable=not verbose,
        ):
            su = min(sl + gpu_sa.max_n_samples, total_n_samples)
            n_samples = su - sl

            # Modify data.
            names_to_set = set(gpu_sa.sample_data)
            for (index, index_data) in enumerate(sa_params.indices):
                self.set_index_data(
                    names_to_set=names_to_set,
                    param_values=param_values,
                    sl=sl,
                    su=su,
                    n_samples=n_samples,
                    index=index,
                    index_data=index_data,
                    gpu_sa=gpu_sa,
                    land_index=land_index,
                )

            # Set remaining data using defaults.
            for name in names_to_set:
                self.set_defaults(
                    name=name,
                    gpu_sa=gpu_sa,
                    land_index=land_index,
                    n_samples=n_samples,
                )

            # Run the model with the new variables.
            model_ba = gpu_sa.run(nSamples=n_samples)

            # NOTE Mean BA at the location is used as a proxy.
            Y[sl:su] = np.mean(model_ba, axis=1)

        gpu_sa.release()

        return sobol.analyze(problem, Y, print_to_console=False)


# NOTE Due to complications with cpp dependencies, cache should be reset manually when
# needed.
@cache(
    dependencies=[
        BAModel.__init__,
        BAModel.process_kwargs,
        GPUBAModelSensitivityAnalysis.__init__,
        GPUBAModelSensitivityAnalysis._get_data_shallow,
        GPUBAModelSensitivityAnalysis.get_sa_params,
        GPUBAModelSensitivityAnalysis.mod_model_data,
        GPUBAModelSensitivityAnalysis.set_defaults,
        GPUBAModelSensitivityAnalysis.set_index_data,
        GPUBAModelSensitivityAnalysis.sobol_sis,
        SAParams.__init__,
        SAParams._check,
        SAParams.add_param_vars,
        SAParams.add_vars,
        SAParams.get_problem,
        get_data_yearly_stdev,
        get_param_uncertainties,
        get_space_template,
    ]
)
def sis_calc(
    *,
    params,
    exponent=8,
    land_points,
    data_variables=None,
    df_sel,
    fuel_build_up_method,
    dryness_method,
):
    sa = GPUBAModelSensitivityAnalysis(
        params=params,
        exponent=8,
        data_variables=data_variables,
        df_sel=df_sel,
        fuel_build_up_method=fuel_build_up_method,
        dryness_method=dryness_method,
    )
    sobol_sis = {}
    for i in tqdm(land_points, desc="land"):
        try:
            sobol_sis[i] = sa.sobol_sis(land_index=i, verbose=False)
        except LandChecksFailed as exc:
            logger.warning(exc)

    return sobol_sis
