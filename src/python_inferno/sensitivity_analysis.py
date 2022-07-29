# -*- coding: utf-8 -*-
from copy import deepcopy
from functools import partial
from operator import itemgetter

from .ba_model import BAModel
from .cache import mark_dependency
from .configuration import Dims, N_pft_groups, land_pts, npft


class LandChecksFailed(RuntimeError):
    """Raised when checks are failed at a given location."""


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

    @mark_dependency
    def _get_data_shallow(self):
        return dict(
            **self.ba_model.data_dict,
            # Crop special case due to the way it is processed and stored.
            obs_pftcrop_1d=self.ba_model.obs_pftcrop_1d,
        )


class GPUSAMixin:
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
