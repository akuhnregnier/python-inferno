# -*- coding: utf-8 -*-
from copy import deepcopy
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
from .model_params import get_param_uncertainties
from .py_gpu_inferno import GPUSA
from .sensitivity_analysis import GPUSAMixin, LandChecksFailed, SensitivityAnalysis
from .utils import map_keys


class SobolQCError(RuntimeError):
    """Raised when Sobol values fail checks."""


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


class SobolSA(SensitivityAnalysis):
    @mark_dependency
    def __init__(self, *, df_sel, exponent=6, **kwargs):
        super().__init__(**kwargs)
        self.exponent = exponent
        self.df_sel = df_sel

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

        return sobol.analyze(problem, Y, print_to_console=False, seed=0)


class GPUBAModelSobolSA(SobolSA, GPUSAMixin):
    @mark_dependency
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.gpu_sa = GPUSA(
            Nt=self.ba_model.Nt,
            drynessMethod=self.ba_model.dryness_method,
            fuelBuildUpMethod=self.ba_model.fuel_build_up_method,
            includeTemperature=self.ba_model.include_temperature,
            overallScale=self.params["overall_scale"],
            crop_f=self.params["crop_f"],
        )

    @mark_dependency
    def sobol_sis(self, *, land_index, verbose=True):
        # Treat each land point individually.
        sa_params = self.get_sa_params(land_index)
        problem = sa_params.get_problem()

        param_values = saltelli.sample(problem, 2**self.exponent)
        assert len(param_values.shape) == 2

        total_n_samples = param_values.shape[0]
        Y = np.zeros(total_n_samples)

        # Call in batches of `max_n_samples`
        for sl in tqdm(
            range(0, total_n_samples, self.gpu_sa.max_n_samples),
            desc="Batched samples",
            disable=not verbose,
        ):
            su = min(sl + self.gpu_sa.max_n_samples, total_n_samples)
            n_samples = su - sl

            # Modify data.
            names_to_set = set(self.gpu_sa.sample_data)
            for (index, index_data) in enumerate(sa_params.indices):
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

            # Set remaining data using defaults.
            for name in names_to_set:
                self.set_defaults(
                    name=name,
                    land_index=land_index,
                    n_samples=n_samples,
                )

            # Run the model with the new variables.
            model_ba = self.gpu_sa.run(nSamples=n_samples)

            # NOTE Mean BA at the location is used as a proxy.
            Y[sl:su] = np.mean(model_ba, axis=1)

        return sobol.analyze(problem, Y, print_to_console=False, seed=0)

    def release(self):
        self.gpu_sa.release()


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


# NOTE Due to complications with cpp dependencies, cache should be reset manually when
# needed.
@cache(
    dependencies=[
        BAModel.__init__,
        BAModel.process_kwargs,
        GPUBAModelSobolSA.__init__,
        GPUBAModelSobolSA._get_data_shallow,
        GPUBAModelSobolSA.get_sa_params,
        GPUBAModelSobolSA.set_defaults,
        GPUBAModelSobolSA.set_index_data,
        GPUBAModelSobolSA.sobol_sis,
        SAParams.__init__,
        SAParams._check,
        SAParams.add_param_vars,
        SAParams.add_vars,
        SAParams.get_problem,
        SensitivityAnalysis.__init__,
        SobolSA.__init__,
        get_data_yearly_stdev,
        get_param_uncertainties,
        get_space_template,
        sobol_qc,
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
    sa = GPUBAModelSobolSA(
        params=params,
        exponent=9,
        data_variables=data_variables,
        df_sel=df_sel,
        fuel_build_up_method=fuel_build_up_method,
        dryness_method=dryness_method,
    )
    sobol_sis = {}
    fail = 0
    success = 0
    for i in (prog := tqdm(land_points, desc="land")):
        try:
            sobol_sis[i] = sa.sobol_sis(land_index=i, verbose=False)
            sobol_qc(sobol_sis[i])
        except (SobolQCError, LandChecksFailed) as exc:
            logger.warning(exc)
            fail += 1
        else:
            success += 1

        prog.set_description(f"land fail - {fail} success - {success}")

    sa.release()

    return sobol_sis
