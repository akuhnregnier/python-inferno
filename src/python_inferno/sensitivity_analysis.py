# -*- coding: utf-8 -*-
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from enum import Enum
from functools import partial
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.signal
import scipy.stats
from jules_output_analysis.data import cube_1d_to_2d, get_1d_data_cube
from loguru import logger
from pylatex.utils import escape_latex
from tqdm import tqdm
from wildfires.analysis import cube_plotting

from python_inferno.hyperopt import get_space_template
from python_inferno.iter_opt import ALWAYS_OPTIMISED, IGNORED

from .ba_model import ARCSINH_FACTOR, BAModel
from .cache import cache, mark_dependency
from .configuration import Dims, N_pft_groups, land_pts, npft
from .data import get_yearly_data, load_jules_lats_lons
from .hyperopt import HyperoptSpace, get_space_template
from .iter_opt import ALWAYS_OPTIMISED, IGNORED
from .mcmc import iter_opt_methods
from .metrics import calculate_phase
from .plotting import use_style
from .py_gpu_inferno import GPUSA
from .space import generate_space_spec
from .spotpy_mcmc import get_cached_mcmc_chains, spotpy_dream
from .stats import gen_correlated_samples, gen_correlated_samples_from_chains
from .utils import DebugExecutor, map_keys

SAMetric = Enum("SAMetric", ["ARCSINH_NME", "MPD"])


class LandChecksFailed(RuntimeError):
    """Raised when checks are failed at a given location."""


def _plot_transform(data):
    out = data - np.min(data, axis=0)
    out /= np.max(out, axis=0)
    return out


class SensitivityAnalysis:
    @mark_dependency
    def __init__(
        self,
        *,
        params,
        data_variables=None,
        fuel_build_up_method,
        dryness_method,
    ):
        self.params = params
        self.ba_model = BAModel(**self.params)

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
            else:
                raise ValueError

            if dryness_method == 1:
                self.data_variables.append("dry_days")
            elif dryness_method == 2:
                self.data_variables.append("grouped_dry_bal")
            else:
                raise ValueError
        else:
            self.data_variables = data_variables

        self.data_stats_params = {}

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

        # Conservative averaging setup.
        self.weights = self.ba_model._cons_monthly_avg.weights
        self.weights[self.weights < 1e-9] = 0
        self.cum_weights = np.sum(self.weights, axis=0)  # mn -> n

        assert not np.any(np.ma.getmask(self.ba_model.mon_avg_gfed_ba_1d))
        self.obs_ba = np.ma.getdata(self.ba_model.mon_avg_gfed_ba_1d)
        self.arcsinh_obs_ba = np.arcsinh(ARCSINH_FACTOR * self.obs_ba)

    @mark_dependency
    def _get_data_shallow(self):
        return dict(
            **self.ba_model.data_dict,
            # Crop special case due to the way it is processed and stored.
            obs_pftcrop_1d=self.ba_model.obs_pftcrop_1d,
        )

    @mark_dependency
    def sa_model_ba_cons_avg(self, model_ba):
        # Conservative averaging - with temporal dimension last instead of first
        # here.
        avg_model_ba = np.einsum(
            # (samples, orig_time), (orig_time, new_time) -> (new_time, samples)
            "sm,mn->ns",
            model_ba,
            self.weights,
            optimize=True,
        ) / self.cum_weights.reshape(-1, 1)

        return avg_model_ba

    @mark_dependency
    def get_asinh_nme(self, *, avg_model_ba, land_index):
        return np.mean(
            np.abs(
                self.arcsinh_obs_ba[:, land_index][:, None]
                - np.arcsinh(ARCSINH_FACTOR * avg_model_ba)
            ),
            axis=0,
        )

    @mark_dependency
    def get_mpd(self, *, avg_model_ba, land_index):
        # NOTE Division by pi omitted (see above).
        if np.all(np.isclose(self.obs_ba[:, land_index], 0, rtol=0, atol=1e-15)):
            return np.nan
        else:
            pred_zero_mask = np.all(
                np.isclose(avg_model_ba, 0, rtol=0, atol=1e-15), axis=0
            )

            assert pred_zero_mask.ndim == 1

            if np.all(pred_zero_mask):
                return np.nan
            else:
                obs_phase = calculate_phase(self.obs_ba[:, land_index][:, None])
                pred_phase = calculate_phase(avg_model_ba[:, ~pred_zero_mask])
                phase_diff = obs_phase - pred_phase

                mpd_vals = np.zeros(pred_zero_mask.size)
                mpd_vals[~pred_zero_mask] = np.arccos(np.cos(phase_diff))
                mpd_vals[pred_zero_mask] = np.nan

                return mpd_vals


class GPUSAMixin:
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
            sample_shape = self.gpu_sa.sample_data[name].shape
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

        self.gpu_sa.set_sample_data(
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
        land_index,
        n_samples,
    ):
        def gpu_set_sample_data(*, source_arr, s, dims):
            self.gpu_sa.set_sample_data(
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

            sample_shape = self.gpu_sa.sample_data[name].shape
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

    def release(self):
        self.gpu_sa.release()


class GPUSampleMetricSA(GPUSAMixin, SensitivityAnalysis):
    @mark_dependency
    def __init__(
        self,
        *,
        N: int = int(1e3),
        chain_data: np.ndarray,
        chain_names,
        **kwargs,
    ):
        """

        Args:
            N (int): Number of parameters to sample.
            chain_data (2D numpy.ndarray): Each column represents a parameter's chain.
            chain_names (iterable of str): Parameter names corresponding to the
                columns of `chain_data`.

        """
        assert kwargs.get("data_variables") is None, "Want all variables."

        super().__init__(**kwargs)

        self.total_n_samples = N
        self.Y_asinh_nme = np.zeros(self.total_n_samples)
        self.Y_mpd = np.zeros(self.total_n_samples)

        self.rng = np.random.default_rng(0)

        space_template = get_space_template(
            dryness_method=self.ba_model.dryness_method,
            fuel_build_up_method=self.ba_model.fuel_build_up_method,
            include_temperature=1,
        )

        space = HyperoptSpace(generate_space_spec(space_template))

        template_param_names = [
            key for key in space_template if key not in ALWAYS_OPTIMISED.union(IGNORED)
        ]

        param_names = []

        for template_param_name in template_param_names:
            assert template_param_name in space.continuous_param_names
            param_names.extend(
                name
                for name in space.continuous_param_names
                if template_param_name in name
            )

        assert set(chain_names).issuperset(set(param_names))

        new_chain_names = []

        to_delete = []
        for i, chain_var in enumerate(chain_names):
            if chain_var not in param_names or chain_var == "crop_f":
                to_delete.append(i)
            else:
                new_chain_names.append(chain_var)

        assert set(new_chain_names).union({"crop_f"}) == set(param_names)

        self.param_names = new_chain_names  # Consistent ordering with chains.

        new_chain_data = np.delete(chain_data, to_delete, 1)
        assert new_chain_data.shape[1] == len(new_chain_names)

        # Sample parameters based on marginals and correlations from the chains.
        _sampled_params_data = gen_correlated_samples_from_chains(
            N=self.total_n_samples,
            chains=new_chain_data,
            icdf_steps=int(1e4),
            rng=np.random.default_rng(0),
        )

        assert _sampled_params_data.shape[0] == self.total_n_samples
        assert _sampled_params_data.shape[1] == len(self.param_names)

        self.sampled_params_data = {
            name: data for name, data in zip(self.param_names, _sampled_params_data.T)
        }

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

        # NOTE Missing keys will be silently ignored. The keys present in
        # `data_yearly_stdev` are therefore determined by `data_stats_params`
        # indirectly.
        self.yearly_data = map_keys(
            get_yearly_data(params=self.data_stats_params), key_mapping
        )
        # Collapse temperatures, since they are all virtually identical (select single
        # PFT).
        self.yearly_data["t1p5m_tile"] = self.yearly_data["t1p5m_tile"][:, 0]

        self.data_names = self.data_variables.copy()

        assert set(self.yearly_data).issuperset(set(self.data_names))

        # # NOTE Combined sampling across all grid points - yields very non-normal
        # # distributions due to e.g. data errors.
        # # Sample data from distribution of yearly mean IAV data differences.

        # flattened_yearly_data = []
        # flattened_names = []

        # for name, data in self.yearly_data.items():
        #     if data.ndim == 2:
        #         # No PFTs.
        #         flattened_names.append(name)
        #         flattened_yearly_data.append(data.ravel())
        #     elif data.ndim == 3:
        #         # With PFTs.
        #         for i in range(data.shape[1]):
        #             flattened_names.append(f"{name}_{i}")
        #             flattened_yearly_data.append(data[:, i].ravel())
        #     else:
        #         raise ValueError()

        # flattened_yearly_data = np.ma.getdata(
        #     np.hstack([data[:, None] for data in flattened_yearly_data])
        # )

        # # We are interested in offsets relative to the baseline, which we want to
        # # compute from the yearly data.

        # # First, de-trend the yearly data.
        # flattened_yearly_data = scipy.signal.detrend(
        #     flattened_yearly_data, axis=0, overwrite_data=True
        # )

        # # Using the data offsets calculated above, sample data for SA.
        # # NOTE This assumes the data is independent of the parameters.
        # _sampled_data = gen_correlated_samples_from_chains(
        #     N=self.total_n_samples,
        #     # Add very small noise to the data in order to ensure correlation matrix
        #     # is positive-definite.
        #     chains=flattened_yearly_data
        #     + (
        #         (1e-6 * np.mean(np.abs(flattened_yearly_data), axis=0))
        #         * np.random.default_rng(0).random(flattened_yearly_data.shape)
        #     ),
        #     icdf_steps=int(1e3),
        #     rng=np.random.default_rng(0),
        # )

        # self.sampled_data = {
        #     name: data for name, data in zip(flattened_names, _sampled_data.T)
        # }

        # # NOTE Plotting.
        # # Original, raw data.
        # plot_pairwise_grid(
        #     chains=_plot_transform(flattened_yearly_data),
        #     names=flattened_names,
        #     save_dir=Path("/tmp"),
        #     filename="orig_pairs",
        # )
        # # Sampled data.
        # plot_pairwise_grid(
        #     chains=_plot_transform(_sampled_data),
        #     names=flattened_names,
        #     save_dir=Path("/tmp"),
        #     filename="sampled_pairs",
        # )

    @mark_dependency
    def set_sa_data(
        self,
        *,
        names_to_set,
        sampled_data,
        su,
        sl,
        n_samples,
        land_index,
    ):
        assert n_samples == su - sl

        # Record names that have been set.
        set_names = set()

        for name in names_to_set:
            sample_shape = self.gpu_sa.sample_data[name].shape
            sample_ndim = len(sample_shape)

            index_data = {"name": name}

            if name in self.data_names:
                source_arr = self.source_data[name]
                assert source_arr.ndim == sample_ndim
                Nt = source_arr.shape[0]

                assert sample_ndim in (2, 3)
                has_pft = sample_ndim == 3

                if has_pft:
                    # PFT.

                    index_data["dims"] = (Dims.TIME, Dims.PFT, Dims.LAND)

                    n_pft = sample_shape[-1]

                    for pft_i in range(n_pft):
                        if name == "t1p5m_tile":
                            # We only sampled 1 version of temperature, since they
                            # are nearly identical across PFTs.
                            sampled_name = name
                        else:
                            sampled_name = f"{name}_{pft_i}"

                        values = sampled_data[sampled_name][sl:su][:, None]

                        arr_s = (slice(None), pft_i, land_index)
                        index_data["slice"] = arr_s

                        offset_data = source_arr[arr_s].reshape(1, Nt)

                        self.gpu_sa.set_sample_data(
                            index_data=index_data,
                            values=values,
                            n_samples=n_samples,
                            offset_data=offset_data,
                        )

                else:
                    # No PFT.
                    index_data["dims"] = (Dims.TIME, Dims.LAND)

                    values = sampled_data[name][sl:su][:, None]

                    arr_s = (slice(None), land_index)
                    index_data["slice"] = arr_s

                    offset_data = source_arr[arr_s].reshape(1, Nt)

                    self.gpu_sa.set_sample_data(
                        index_data=index_data,
                        values=values,
                        n_samples=n_samples,
                        offset_data=offset_data,
                    )

            elif name in self.param_names:
                # Only parameters that vary across PFTs are currently supported.
                assert sample_ndim == 2, "samples, pft"

                index_data["dims"] = (Dims.PFT,)

                n_pft = sample_shape[-1]
                assert n_pft == N_pft_groups

                for pft_i in range(n_pft):

                    if pft_i > 0:
                        sampled_name = f"{name}{pft_i+1}"
                    else:
                        sampled_name = name

                    values = self.sampled_params_data[sampled_name][sl:su]
                    index_data["slice"] = (pft_i,)

                    self.gpu_sa.set_sample_data(
                        index_data=index_data,
                        values=values,
                        n_samples=n_samples,
                    )

            else:
                # Will be set using defaults later.
                continue

            set_names.add(name)

        for name in set_names:
            names_to_set.remove(name)

    @mark_dependency
    def sensitivity_analysis(self, *, land_index, verbose=True):
        if np.any(self._checks_failed_mask[:, :, land_index]):
            raise LandChecksFailed(f"Some checks failed at location {land_index}.")

        # Sample data from distribution of yearly mean IAV data differences.

        # NOTE Land-point specific.
        point_yearly_data = {
            key: data[..., land_index]
            for key, data in self.yearly_data.items()
            if key in self.data_names
        }
        assert set(point_yearly_data) == set(self.data_names)

        flattened_point_yearly_data = []
        flattened_names = []

        for name, data in point_yearly_data.items():
            if data.ndim == 1:
                flattened_names.append(name)
                flattened_point_yearly_data.append(data)
            elif data.ndim == 2:
                for i in range(data.shape[1]):
                    flattened_names.append(f"{name}_{i}")
                    flattened_point_yearly_data.append(data[:, i])
            else:
                raise ValueError()

        flattened_point_yearly_data = np.ma.getdata(
            np.hstack([data[:, None] for data in flattened_point_yearly_data])
        )

        # We are interested in offsets relative to the baseline, which we want to
        # compute from the yearly data.

        # First, de-trend the yearly data.
        flattened_point_yearly_data = scipy.signal.detrend(
            flattened_point_yearly_data,
            axis=0,
            overwrite_data=True,
        )

        # Using the data offsets calculated above, sample data for SA.
        # NOTE This assumes the data is independent of the parameters.

        # Add noisy data repetition to ensure correlation matrix is positive-definite,
        # since otherwise the nr. of samples (years) will be less than the number of
        # variables.
        u_noise = np.random.default_rng(0).random(flattened_point_yearly_data.shape)
        corr_sample_data = np.vstack(
            (
                flattened_point_yearly_data,
                flattened_point_yearly_data * (1 + (0.01 * (u_noise - 0.5))),
            )
        )

        corr, corr_p = scipy.stats.spearmanr(corr_sample_data)
        # Data offset distributions are approximated by normal distributions here due
        # to the low sample size for each location (only 17 years / samples each).
        F_invs = [
            # Ensure there is a minimum standard deviation.
            scipy.stats.norm(loc=0, scale=max(np.std(data), 1e-5)).ppf
            for data in flattened_point_yearly_data.T
        ]

        _sampled_data = gen_correlated_samples(
            N=self.total_n_samples,
            F_invs=F_invs,
            R=corr,
            rng=np.random.default_rng(0),
        )

        sampled_data = {
            name: data for name, data in zip(flattened_names, _sampled_data.T)
        }

        # NOTE Plotting.
        # plot_pairwise_grid(
        #     chains=_plot_transform(flattened_point_yearly_data),
        #     names=flattened_names,
        #     save_dir=Path("/tmp"),
        #     filename="orig_pairs",
        # )
        # plot_pairwise_grid(
        #     chains=_plot_transform(_sampled_data),
        #     names=flattened_names,
        #     save_dir=Path("/tmp"),
        #     filename="sampled_pairs",
        # )

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
            self.set_sa_data(
                names_to_set=names_to_set,
                sampled_data=sampled_data,
                su=su,
                sl=sl,
                n_samples=n_samples,
                land_index=land_index,
            )

            if not did_set_defaults[n_samples]:
                # NOTE Only need to do this once since the defaults will not change
                # between batches.

                # Set remaining data using defaults.
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

        all_names = flattened_names + self.param_names
        assert len(set(all_names)) == len(all_names)

        X = []
        for name in all_names:
            if name in flattened_names:
                X.append(sampled_data[name][:, None])
            elif name in self.param_names:
                X.append(self.sampled_params_data[name][:, None])
            else:
                raise ValueError(name)

        X = np.hstack(X)

        problem = self.gen_problem(all_names=all_names)

        out = {}

        for metric_key, Y in tqdm(
            [
                (SAMetric.ARCSINH_NME, self.Y_asinh_nme),
                (SAMetric.MPD, self.Y_mpd),
            ],
            desc="Sensitivity metric",
            disable=not verbose,
        ):
            invalid_mask = np.isnan(Y) | np.isinf(Y)
            if np.all(invalid_mask):
                continue
            if np.any(invalid_mask):
                valid_mask = ~invalid_mask
                X_sel = X[valid_mask]
                Y_sel = Y[valid_mask]
            else:
                X_sel = X
                Y_sel = Y

            try:
                sa_indices = self.calc_sa_indices(
                    problem=problem,
                    X=X_sel,
                    Y=Y_sel,
                    verbose=verbose,
                )

            except Exception as exc:
                logger.exception(exc)
            else:
                out[metric_key] = sa_indices

        return out


def init_stats():
    success_stats = {}
    for metric in SAMetric:
        success_stats[metric] = dict(
            fail=0,
            success=0,
        )
    return success_stats


def update_stats(success_stats, indices):
    for metric, stats in success_stats.items():
        if metric in indices:
            stats["success"] += 1
        else:
            stats["fail"] += 1


def fail_stats(success_stats):
    for stats in success_stats.values():
        stats["fail"] += 1


def format_stats(success_stats):
    out = ""
    for key, stats in success_stats.items():
        out += f"{key.name} - {stats['success']}/{stats['fail']} | "
    return out.strip()


@cache
def get_mean_df(*, sis, keys, names):
    key_data = {}

    for key in keys:
        all_vals = np.hstack(
            [np.asarray(indices[key]).reshape(-1, 1) for indices in sis.values()]
        )

        # TODO - investigate large magnitudes - observed e.g. for B2, t1p5m and litter
        # pool
        all_vals[np.abs(all_vals) > 10] = np.nan

        # TODO - investigate NaNs here
        avg_vals = np.nanmean(all_vals, axis=1)
        assert avg_vals.size == len(names)

        key_data[key] = avg_vals
        key_data[f"{key}_std"] = np.nanstd(all_vals, axis=1)

    df = pd.DataFrame(key_data, index=names)
    return df


def plot_si(*, data, jules_lats, jules_lons, name, plot_dir):
    use_style()
    cube_2d = cube_1d_to_2d(get_1d_data_cube(data, lats=jules_lats, lons=jules_lons))

    plt.ioff()
    fig = plt.figure(figsize=(6, 4), dpi=150)
    cube_plotting(cube_2d, title=name, fig=fig)
    fig.savefig(plot_dir / name)
    plt.close(fig)


def analyse_sis(
    *,
    sis,
    save_dir,
    exp_name,
    exp_key,
    metric_name,
    method_name,
    method_keys,
    sort_key,
    analysis_type,
    to_df_func=None,
):
    try:
        names = next(iter(sis.values()))["names"]
    except KeyError:
        names = to_df_func(next(iter(sis.values())))[0].index

    df = get_mean_df(sis=sis, keys=list(sorted(method_keys)), names=names)
    df = df.sort_values(sort_key, ascending=False)

    assert len(names) == df.shape[0]

    print_df = df.copy()
    print_df.index = list(map(escape_latex, print_df.index))
    print_df.columns = list(map(escape_latex, print_df.columns))

    caption_lines = (
        f"Global {analysis_type} {method_name} sensitivity analysis of for {exp_name} "
        f"measured by {metric_name}. Computed from individual significance values at "
        f"{len(sis)} land points.",
    )

    print(
        print_df.style.to_latex(
            caption=(
                "\n".join(("%", *map(escape_latex, caption_lines), "")),
                escape_latex(
                    f"Global {analysis_type} {method_name} SA for {exp_name} measured "
                    f"by {metric_name}."
                ),
            ),
            label=(
                f"table:global_sis_{analysis_type.lower()}_{exp_key.lower()}"
                f"_{metric_name.lower()}"
            ),
            hrules=True,
            position_float="centering",
            siunitx=True,
        )
    )

    jules_lats, jules_lons = load_jules_lats_lons()

    # Plotting.

    executor = ProcessPoolExecutor()
    futures = []

    for method_key in method_keys:
        plot_dir = save_dir / method_key
        plot_dir.mkdir(exist_ok=True)

        for i, name in enumerate(tqdm(names, desc="Submitting plots")):
            data = np.ma.MaskedArray(np.zeros(land_pts), mask=True)

            for land_i, val in sis.items():
                st_val = val[method_key][i]
                if not np.isnan(st_val):
                    data[land_i] = st_val

            if np.all(np.ma.getmaskarray(data)):
                logger.warning(f"{name} all masked!")
                continue

            if np.all(np.isclose(data, 0)):
                logger.info(f"{name} all close to 0.")
                continue

            futures.append(
                executor.submit(
                    plot_si,
                    data=data,
                    jules_lats=jules_lats,
                    jules_lons=jules_lons,
                    name=name,
                    plot_dir=plot_dir,
                )
            )

    for f in tqdm(as_completed(futures), total=len(futures), desc="Plotting"):
        f.result()

    executor.shutdown()


@mark_dependency
def _sis_calc(
    *,
    land_points,
    verbose=False,
    tqdm_level=None,
    sa_class,
    **kwargs,
):
    """

    For GPUSAMetric classes, kwargs:
        - params
        - dryness_method
        - fuel_build_up_method
        - N
        - chain_data
        - chain_names

    For GPUBAModelSobolSA, kwargs:
        - params
        - exponent
        - data_variables
        - fuel_build_up_method
        - dryness_method

    """
    sa = sa_class(**kwargs)

    sis = {}

    success_stats = init_stats()

    for land_index in (
        prog := tqdm(
            land_points, desc="land points", disable=verbose < 1, position=tqdm_level
        )
    ):
        try:
            indices = sa.sensitivity_analysis(
                land_index=land_index, verbose=verbose > 1
            )
            sis[land_index] = indices
        except Exception as exc:
            logger.exception(exc)
            fail_stats(success_stats)
        else:
            update_stats(success_stats, indices)

        prog.set_description(format_stats(success_stats))

    sa.release()

    return sis


@mark_dependency
def batched_sis_calc(
    *,
    n_batches=1,
    land_points,
    sa_class,
    **kwargs,
):
    """

    For GPUSAMetric classes, kwargs:
        - params
        - dryness_method
        - fuel_build_up_method
        - N
        - chain_data
        - chain_names

    For GPUBAModelSobolSA, kwargs:
        - params
        - exponent
        - data_variables
        - fuel_build_up_method
        - dryness_method

    """
    batch_bounds = np.unique(
        np.linspace(0, len(land_points), n_batches + 1, dtype=np.int64)
    )
    batch_land_points = [
        land_points[low:upp] for low, upp in zip(batch_bounds[:-1], batch_bounds[1:])
    ]

    executor = (
        ProcessPoolExecutor(max_workers=max(20, n_batches))
        if n_batches > 1
        else DebugExecutor()
    )
    futures = []

    for batch_i, batch_land_points in enumerate(batch_land_points):
        futures.append(
            executor.submit(
                _sis_calc,
                land_points=batch_land_points,
                sa_class=sa_class,
                verbose=1,
                tqdm_level=batch_i,
                **kwargs,
            )
        )

    all_sis = {}
    for f in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Collecting results",
        position=n_batches,
    ):
        all_sis = {**all_sis, **f.result()}

    return all_sis


def get_gpu_sample_metric_sa_class_dependencies(sa_class):
    class_deps = list(
        getattr(sa_class, name)
        for name in [
            "__init__",
            "_get_data_shallow",
            "calc_sa_indices",
            "gen_problem",
            "get_asinh_nme",
            "get_mpd",
            "sa_model_ba_cons_avg",
            "salib_analyze_func",
            "sensitivity_analysis",
            "set_defaults",
            "set_sa_data",
        ]
    )

    return class_deps + [
        BAModel.__init__,
        BAModel.process_kwargs,
        SensitivityAnalysis.__init__,
        _sis_calc,
        batched_sis_calc,
        calculate_phase,
        gen_correlated_samples,
        gen_correlated_samples_from_chains,
        get_cached_mcmc_chains,
        get_space_template,
        get_yearly_data,
        iter_opt_methods,
        spotpy_dream,
    ]
