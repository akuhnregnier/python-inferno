# -*- coding: utf-8 -*-
from copy import deepcopy
from functools import partial

import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from tqdm import tqdm

from .ba_model import BAModel
from .cache import cache
from .configuration import Dims, N_pft_groups, land_pts, npft
from .py_gpu_inferno import GPUSA


class LandChecksFailed(RuntimeError):
    """Raised when checks are failed at a given location."""


def min_max_bound_func(values):
    min_val = np.min(values)
    max_val = np.max(values)
    if np.isclose(min_val, max_val):
        print("Close bounds:", min_val)
        return [min_val, min_val + 1]

    return [min_val, max_val]


class SAParams:
    def __init__(self):
        self._variable_indices = []
        self._variable_bounds = []
        self._variable_names = []
        self._variable_groups = []

    def add_vars(
        self,
        *,
        name: str,
        data: np.ndarray,
        land_index: int,
        bound_func: callable,
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
            self._variable_bounds.append(bound_func(data[s]))
            self._variable_names.append(f"{name}_{i}")
            # Group all PFTs.
            self._variable_groups.append(name)

        # Sanity check.
        assert (
            len(self._variable_indices)
            == len(self._variable_bounds)
            == len(self._variable_names)
            == len(self._variable_groups)
        )

    def get_problem(self):
        return dict(
            groups=self._variable_groups,
            names=self._variable_names,
            num_vars=len(self._variable_names),
            bounds=self._variable_bounds,
        )

    @property
    def indices(self):
        return deepcopy(self._variable_indices)


class SensitivityAnalysis:
    def __init__(self, *, params, data_variables=None, bound_func=None, exponent=6):
        self.params = params
        self.ba_model = BAModel(**self.params)
        self.exponent = exponent

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
                "fuel_build_up",
                "fapar_diag_pft",
                "litter_pool",
                "dry_days",
                "grouped_dry_bal",
                # Extra param.
                "obs_pftcrop_1d",
            ]
        if bound_func is None:
            self.bound_func = min_max_bound_func

        self.source_data = deepcopy(self._get_data_shallow())

        assert all(name in self.source_data for name in self.data_variables)

    def _get_data_shallow(self):
        return dict(
            **self.ba_model.data_dict,
            # Crop special case due to the way it is processed and stored.
            obs_pftcrop_1d=self.ba_model.obs_pftcrop_1d,
        )

    def mod_model_data(self, *, name: str, s: tuple, vals):
        if name == "obs_pftcrop_1d":
            # Crop special case.
            self.ba_model.obs_pftcrop_1d[s] = vals
        else:
            self.ba_model.data_dict[name][s] = vals

    def get_sa_params(self, land_index: int):
        if np.any(self._checks_failed_mask[:, :, land_index]):
            raise LandChecksFailed(f"Some checks failed at location {land_index}.")

        sa_params = SAParams()
        # For each of the variables to be investigated, define bounds associated with
        # the variable, its name, and the group it belongs to. For variables with
        # multiple PFTs, up to `n` natural PFTs are considered (grouped into the same
        # group).
        # NOTE Currently, bounds for each variable are defined according to its range
        # at the given land point / PFT, although this may change.
        for name in self.data_variables:
            sa_params.add_vars(
                name=name,
                data=self.source_data[name],
                land_index=land_index,
                bound_func=self.bound_func,
            )

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
            # Insert the modified data.
            for j, index_data in enumerate(sa_params.indices):
                self.mod_model_data(
                    name=index_data["name"],
                    s=index_data["slice"],
                    vals=param_values[i, j],
                )

            # Run the model with the new variables.
            model_ba = self.ba_model.run(
                land_point=land_index,
                **self.params,
            )["model_ba"]

            # NOTE Mean BA at the location used as a proxy.
            Y[i] = np.mean(model_ba[:, land_index])

        return sobol.analyze(problem, Y, print_to_console=False)


class GPUBAModelSensitivityAnalysis(SensitivityAnalysis):
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
            land_point_checks_failed=self.ba_model._get_checks_failed_mask()[
                :, :, land_index
            ],
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
                if Dims.TIME in index_data["dims"]:
                    # Ensure it is indeed unlimited as stated above.
                    assert index_data["slice"][
                        index_data["dims"].index(Dims.TIME)
                    ] == slice(None)
                    values = values.reshape(-1, 1)

                gpu_sa.set_sample_data(
                    index_data=index_data,
                    values=values,
                    n_samples=n_samples,
                )
                if index_data["name"] in names_to_set:
                    # Remove the name. Done only once like so, since we are
                    # potentially setting data for the same 'name' multiple times for
                    # different PFTs.
                    names_to_set.remove(index_data["name"])

            # Set remaining data using defaults.
            for name in names_to_set:

                def gpu_set_sample_data(*, source_arr, s, dims):
                    gpu_sa.set_sample_data(
                        index_data={"name": name, "slice": s, "dims": dims},
                        values=source_arr[s],
                        n_samples=n_samples,
                    )

                if name in self.proc_params:
                    gpu_set_sample_data = partial(
                        gpu_set_sample_data, source_arr=self.proc_params[name]
                    )
                    for pft_i in range(N_pft_groups):
                        gpu_set_sample_data(s=(pft_i,), dims=(Dims.PFT,))
                else:
                    arr = self.ba_model.data_dict[name]
                    gpu_set_sample_data = partial(gpu_set_sample_data, source_arr=arr)
                    sample_shape = gpu_sa.sample_data[name].shape
                    assert len(arr.shape) == len(sample_shape)

                    if len(sample_shape) == 3:
                        for pft_i in range(sample_shape[-1]):
                            gpu_set_sample_data(
                                s=(slice(None), pft_i, land_index),
                                dims=(Dims.TIME, Dims.PFT, Dims.LAND),
                            )
                    elif len(sample_shape) == 2:
                        gpu_set_sample_data(
                            s=(slice(None), land_index), dims=(Dims.TIME, Dims.LAND)
                        )
                    else:
                        raise ValueError(f"Unknown sample shape '{sample_shape}'.")

            # Run the model with the new variables.
            model_ba = gpu_sa.run(nSamples=n_samples)

            # NOTE Mean BA at the location is used as a proxy.
            Y[sl:su] = np.mean(model_ba, axis=1)

        gpu_sa.release()

        return sobol.analyze(problem, Y, print_to_console=False)


@cache
def sis_calc(*, params, exponent=8, land_points):
    sa = GPUBAModelSensitivityAnalysis(params=params, exponent=8)
    sobol_sis = {}
    for i in tqdm(land_points, desc="land"):
        try:
            sobol_sis[i] = sa.sobol_sis(land_index=i, verbose=False)
        except LandChecksFailed:
            pass

    return sobol_sis
