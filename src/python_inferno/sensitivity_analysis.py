# -*- coding: utf-8 -*-
from copy import deepcopy

import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from tqdm import tqdm

from .ba_model import BAModel
from .configuration import npft


def bound_func(values):
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
        elif data.ndim == 2:
            slices = [(slice(None), land_index)]
        else:
            raise ValueError(f"Unexpected shape '{data.shape}'.")

        for i, s in enumerate(slices):
            self._variable_indices.append({"name": name, "slice": s})
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
    def __init__(self, *, params, data_variables=None, bound_func=None):
        self.params = params
        self.ba_model = BAModel(**self.params)

        if data_variables is None:
            self.data_variables = [
                "t1p5m_tile",
                "q1p5m_tile",
                "pstar",
                "sthu_soilt_single",
                # "frac",
                "c_soil_dpm_gb",
                "c_soil_rpm_gb",
                "canht",
                "ls_rain",
                "con_rain",
                "fuel_build_up",
                "fapar_diag_pft",
                "litter_pool",
                "dry_days",
                "grouped_dry_bal",
                # Extra param.
                "obs_pftcrop_1d",
            ]
        if bound_func is None:
            self.bound_func = bound_func

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

    def get_problem(self, land_index: int):
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

        return sa_params.get_problem()


class BAModelSensitivityAnalysis(SensitivityAnalysis):
    def sobol_sis(self, *, land_index):
        # Treat each land point individually.
        problem = self.get_problem(land_index)

        param_values = saltelli.sample(problem, 2**6)
        assert len(param_values.shape) == 2

        Y = np.zeros(param_values.shape[0])
        for i in tqdm(range(param_values.shape[0]), desc="samples"):
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

            # TODO NOTE Use first timestep as proxy here
            Y[i] = model_ba[0, land_index]

        return sobol.analyze(problem, Y, print_to_console=False)
