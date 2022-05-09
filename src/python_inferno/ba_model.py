# -*- coding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from loguru import logger
from sklearn.metrics import r2_score

from .configuration import N_pft_groups, land_pts, npft
from .data import get_processed_climatological_data, timestep
from .metrics import Metrics, loghist, nmse
from .multi_timestep_inferno import multi_timestep_inferno
from .py_gpu_inferno import GPUCalculateMPD, cpp_nme
from .utils import (
    ConsMonthlyAvgNoMask,
    expand_pft_params,
    transform_dtype,
    unpack_wrapped,
)

ARCSINH_FACTOR = 1e6
MPD_IGNORE_THRES = 5600

dummy_pop_den = np.zeros((land_pts,), dtype=np.float32) - 1
dummy_flash_rate = np.zeros((land_pts,), dtype=np.float32) - 1


class BAModelException(RuntimeError):
    """Raised when inadequate BA model parameters are used."""


def process_param_inplace(*, kwargs, name, n_source, n_target, dtype):
    """Process parameter values.

    For example, a parameter with `name='param'` could be optimised for 3 different PFT groups,
    which would be given as `kwargs=dict(param=1.0, param2=2.0, param3=1.2)`.

    This function will then take these inputs and transform them according to
    `n_source`, `n_target`, and `dtype`.

    `kwargs` will be consumed in the process. Pass a copy (`kwargs.copy()`) to prevent
    side-effects.

    """
    assert n_target >= n_source
    assert n_target in (npft, N_pft_groups)

    if n_source == 1:
        return np.array([kwargs.pop(name)] * n_target).astype(dtype)
    elif n_source == 3:
        values = [kwargs.pop(name), kwargs.pop(name + "2"), kwargs.pop(name + "3")]
        if n_target == npft:
            return np.asarray(expand_pft_params(values)).astype(dtype)
        return np.asarray(values).astype(dtype)
    else:
        raise ValueError(f"n_source {n_source} not supported.")


def run_model(
    *,
    dryness_method,
    fuel_build_up_method,
    include_temperature,
    overall_scale,
    data_dict,
    **kwargs,
):
    model_ba = unpack_wrapped(multi_timestep_inferno)(
        ignition_method=1,
        timestep=timestep,
        flammability_method=2,
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=include_temperature,
        overall_scale=overall_scale,
        # These are not used for ignition mode 1, nor do they contain a temporal
        # coordinate.
        pop_den=dummy_pop_den,
        flash_rate=dummy_flash_rate,
        **data_dict,
        **kwargs,
    )
    return model_ba


def calculate_mpd(avg_ba, mon_avg_gfed_ba_1d, mpd=GPUCalculateMPD(land_pts).run):
    if mon_avg_gfed_ba_1d.shape[0] == 12:
        pad_func = lambda x: x
    else:
        pad_func = partial(
            np.pad,
            pad_width=((0, 12 - mon_avg_gfed_ba_1d.shape[0]), (0, 0)),
            constant_values=0.0,
        )
    mpd_val, ignored = mpd(
        obs=pad_func(mon_avg_gfed_ba_1d), pred=pad_func(avg_ba), return_ignored=True
    )
    if ignored > MPD_IGNORE_THRES:
        # Ensure that not too many samples are ignored.
        raise BAModelException()

    return mpd_val, ignored


def calculate_scores(
    *, model_ba, cons_monthly_avg, mon_avg_gfed_ba_1d, requested=Metrics
):
    assert requested

    if np.all(np.abs(model_ba) < 1e-15):
        raise BAModelException()

    # Calculate monthly averages.
    avg_ba = cons_monthly_avg.cons_monthly_average_data(model_ba)
    assert avg_ba.shape == mon_avg_gfed_ba_1d.shape

    if any(
        metric in requested
        for metric in (
            Metrics.R2,
            Metrics.NME,
            Metrics.ARCSINH_NME,
            Metrics.NMSE,
            Metrics.LOGHIST,
        )
    ):
        sel = ~np.ma.getmaskarray(mon_avg_gfed_ba_1d)
        y_pred = np.ma.getdata(avg_ba)[sel]
        y_true = np.ma.getdata(mon_avg_gfed_ba_1d)[sel]
        assert y_pred.shape == y_true.shape

    scores = {}

    with ThreadPoolExecutor(
        max_workers=max(
            1, int(Metrics.MPD in requested) + int(Metrics.NME in requested)
        )
    ) as executor:
        if Metrics.MPD in requested:
            mpd_future = executor.submit(calculate_mpd, avg_ba, mon_avg_gfed_ba_1d)

        if Metrics.NME in requested:
            nme_future = executor.submit(cpp_nme, obs=y_true, pred=y_pred)

        if Metrics.R2 in requested:
            scores["r2"] = r2_score(y_true=y_true, y_pred=y_pred)

        if Metrics.ARCSINH_NME in requested:
            scores["arcsinh_nme"] = cpp_nme(
                obs=np.arcsinh(ARCSINH_FACTOR * y_true),
                pred=np.arcsinh(ARCSINH_FACTOR * y_pred),
            )

        if Metrics.NMSE in requested:
            scores["nmse"] = nmse(obs=y_true, pred=y_pred)

        if Metrics.LOGHIST in requested:
            scores["loghist"] = loghist(
                obs=y_true, pred=y_pred, edges=np.linspace(0, 0.4, 20)
            )

        if Metrics.MPD in requested:
            scores["mpd"], scores["mpd_ignored"] = mpd_future.result()

        if Metrics.NME in requested:
            scores["nme"] = nme_future.result()

    if any(np.ma.is_masked(val) for val in scores.values()):
        raise BAModelException()

    return scores, avg_ba


class BAModel:
    def __init__(
        self,
        *,
        dryness_method,
        fuel_build_up_method,
        include_temperature,
        average_samples,
        **kwargs,
    ):
        self.dryness_method = int(dryness_method)
        self.fuel_build_up_method = int(fuel_build_up_method)
        self.include_temperature = int(include_temperature)
        self.average_samples = int(average_samples)

        logger.info(
            "Init: "
            + ", ".join(
                map(
                    str,
                    (
                        self.dryness_method,
                        self.fuel_build_up_method,
                        self.include_temperature,
                        self.average_samples,
                    ),
                )
            )
        )

        self.rain_f = (
            process_param_inplace(
                kwargs=kwargs,
                name="rain_f",
                n_source=3 if "rain_f2" in kwargs else 1,
                n_target=N_pft_groups,
                dtype=np.float64,
            )
            if dryness_method == 2
            else None
        )
        self.vpd_f = (
            process_param_inplace(
                kwargs=kwargs,
                name="vpd_f",
                n_source=3 if "vpd_f2" in kwargs else 1,
                n_target=N_pft_groups,
                dtype=np.float64,
            )
            if dryness_method == 2
            else None
        )

        self.n_samples_pft = (
            process_param_inplace(
                kwargs=kwargs,
                name="fuel_build_up_n_samples",
                n_source=3 if "fuel_build_up_n_samples2" in kwargs else 1,
                n_target=N_pft_groups,
                dtype=np.int64,
            )
            if fuel_build_up_method == 1
            else None
        )

        self.litter_tc = (
            process_param_inplace(
                kwargs=kwargs,
                name="litter_tc",
                n_source=3 if "litter_tc2" in kwargs else 1,
                n_target=npft,
                dtype=np.float64,
            )
            if fuel_build_up_method == 2
            else None
        )
        self.leaf_f = (
            process_param_inplace(
                kwargs=kwargs,
                name="leaf_f",
                n_source=3 if "leaf_f2" in kwargs else 1,
                n_target=npft,
                dtype=np.float64,
            )
            if fuel_build_up_method == 2
            else None
        )

        (
            data_dict,
            self.mon_avg_gfed_ba_1d,
            self.jules_time_coord,
        ) = get_processed_climatological_data(
            litter_tc=self.litter_tc,
            leaf_f=self.leaf_f,
            n_samples_pft=self.n_samples_pft,
            average_samples=self.average_samples,
            rain_f=self.rain_f,
            vpd_f=self.vpd_f,
        )

        # Shallow copy to allow popping of the dictionary without affecting the
        # memoized copy.
        self.data_dict = data_dict.copy()
        # Extract variables not used further below.
        self.obs_pftcrop_1d = self.data_dict.pop("obs_pftcrop_1d")

        # Set up conservative averaging.
        self.cons_monthly_avg = ConsMonthlyAvgNoMask(self.jules_time_coord, L=land_pts)

    def process_kwargs(self, **kwargs):
        n_params = N_pft_groups
        dtype_params = np.float64
        dummy_params = np.ones(n_params, dtype=dtype_params)

        processed_kwargs = dict(overall_scale=kwargs.pop("overall_scale", 1.0))

        def process_key_from_kwargs(key):
            return process_param_inplace(
                kwargs=kwargs,
                name=key,
                n_source=3 if f"{key}2" in kwargs else 1,
                n_target=n_params,
                dtype=dtype_params,
            )

        # The below may conditionally be present. If not, they need to be provided
        # by 'dummy' variables.
        for keys, condition in (
            (
                # Always process these keys.
                [
                    "fapar_factor",
                    "fapar_centre",
                    "fapar_shape",
                    "fapar_weight",
                    "dryness_weight",
                    "fuel_weight",
                ],
                True,
            ),
            (
                [
                    "temperature_factor",
                    "temperature_centre",
                    "temperature_shape",
                    "temperature_weight",
                ],
                self.include_temperature == 1,
            ),
            (
                ["fuel_build_up_factor", "fuel_build_up_centre", "fuel_build_up_shape"],
                self.fuel_build_up_method == 1,
            ),
            (
                ["litter_pool_factor", "litter_pool_centre", "litter_pool_shape"],
                self.fuel_build_up_method == 2,
            ),
            (
                ["dry_day_factor", "dry_day_centre", "dry_day_shape"],
                self.dryness_method == 1,
            ),
            (
                ["dry_bal_factor", "dry_bal_centre", "dry_bal_shape"],
                self.dryness_method == 2,
            ),
        ):
            for key in keys:
                processed_kwargs[key] = (
                    process_key_from_kwargs(key) if condition else dummy_params
                )

        return processed_kwargs

    def get_model_ba(
        self,
        dryness_method,
        fuel_build_up_method,
        include_temperature,
        data_dict,
        **processed_kwargs,
    ):
        return run_model(
            dryness_method=self.dryness_method,
            fuel_build_up_method=self.fuel_build_up_method,
            include_temperature=self.include_temperature,
            data_dict=self.data_dict,
            **processed_kwargs,
        )

    def run(
        self,
        *,
        crop_f,
        **kwargs,
    ):
        processed_kwargs = self.process_kwargs(**kwargs)

        model_ba = self.get_model_ba(
            dryness_method=self.dryness_method,
            fuel_build_up_method=self.fuel_build_up_method,
            include_temperature=self.include_temperature,
            data_dict=self.data_dict,
            **processed_kwargs,
        )

        # Modify the predicted BA using the crop fraction (i.e. assume a certain
        # proportion of cropland never burns, even though this may be the case in
        # given the weather conditions).
        model_ba *= 1 - crop_f * self.obs_pftcrop_1d

        return dict(
            model_ba=model_ba,
            data_params=dict(
                litter_tc=self.litter_tc,
                leaf_f=self.leaf_f,
                n_samples_pft=self.n_samples_pft,
                average_samples=self.average_samples,
                rain_f=self.rain_f,
                vpd_f=self.vpd_f,
            ),
            obs_pftcrop_1d=self.obs_pftcrop_1d,
            jules_time_coord=self.jules_time_coord,
            mon_avg_gfed_ba_1d=self.mon_avg_gfed_ba_1d,
            data_dict=self.data_dict,
        )

    def calc_scores(self, *, model_ba, requested):
        scores, avg_ba = calculate_scores(
            model_ba=model_ba,
            cons_monthly_avg=self.cons_monthly_avg,
            mon_avg_gfed_ba_1d=self.mon_avg_gfed_ba_1d,
            requested=requested,
        )

        return dict(
            avg_ba=avg_ba,
            scores=scores,
        )


class GPUBAModel(BAModel):
    def __init__(self, **kwargs):
        from .py_gpu_inferno import GPUInferno

        logger.info("GPUBAModel init.")

        super().__init__(**kwargs)

        self.Nt = self.data_dict["pstar"].shape[0]

        self._gpu_inferno = transform_dtype(GPUInferno)(
            ignition_method=1,
            flammability_method=2,
            dryness_method=self.dryness_method,
            fuel_build_up_method=self.fuel_build_up_method,
            include_temperature=self.include_temperature,
            Nt=self.Nt,
            t1p5m_tile=self.data_dict["t1p5m_tile"].ravel(),
            q1p5m_tile=self.data_dict["q1p5m_tile"].ravel(),
            pstar=self.data_dict["pstar"].ravel(),
            sthu_soilt_single=self.data_dict["sthu_soilt"][:, 0, 0].ravel(),
            frac=self.data_dict["frac"].ravel(),
            c_soil_dpm_gb=self.data_dict["c_soil_dpm_gb"].ravel(),
            c_soil_rpm_gb=self.data_dict["c_soil_rpm_gb"].ravel(),
            canht=self.data_dict["canht"].ravel(),
            ls_rain=self.data_dict["ls_rain"].ravel(),
            con_rain=self.data_dict["con_rain"].ravel(),
            pop_den=dummy_pop_den,
            flash_rate=dummy_flash_rate,
            fuel_build_up=self.data_dict["fuel_build_up"].ravel(),
            fapar_diag_pft=self.data_dict["fapar_diag_pft"].ravel(),
            grouped_dry_bal=self.data_dict["grouped_dry_bal"].ravel(),
            litter_pool=self.data_dict["litter_pool"].ravel(),
            dry_days=self.data_dict["dry_days"].ravel(),
        )

    def get_model_ba(
        self,
        *,
        overall_scale,
        fapar_factor,
        fapar_centre,
        fapar_shape,
        fuel_build_up_factor,
        fuel_build_up_centre,
        fuel_build_up_shape,
        temperature_factor,
        temperature_centre,
        temperature_shape,
        dry_day_factor,
        dry_day_centre,
        dry_day_shape,
        dry_bal_factor,
        dry_bal_centre,
        dry_bal_shape,
        litter_pool_factor,
        litter_pool_centre,
        litter_pool_shape,
        fapar_weight,
        dryness_weight,
        temperature_weight,
        fuel_weight,
        **kwargs,
    ):
        # TODO Eliminate need for dtype transform.
        return transform_dtype(self._gpu_inferno.run)(
            overall_scale=overall_scale,
            fapar_factor=fapar_factor,
            fapar_centre=fapar_centre,
            fapar_shape=fapar_shape,
            fuel_build_up_factor=fuel_build_up_factor,
            fuel_build_up_centre=fuel_build_up_centre,
            fuel_build_up_shape=fuel_build_up_shape,
            temperature_factor=temperature_factor,
            temperature_centre=temperature_centre,
            temperature_shape=temperature_shape,
            dry_day_factor=dry_day_factor,
            dry_day_centre=dry_day_centre,
            dry_day_shape=dry_day_shape,
            dry_bal_factor=dry_bal_factor,
            dry_bal_centre=dry_bal_centre,
            dry_bal_shape=dry_bal_shape,
            litter_pool_factor=litter_pool_factor,
            litter_pool_centre=litter_pool_centre,
            litter_pool_shape=litter_pool_shape,
            fapar_weight=fapar_weight,
            dryness_weight=dryness_weight,
            temperature_weight=temperature_weight,
            fuel_weight=fuel_weight,
        )

    def release(self):
        self._gpu_inferno.release()


def gen_to_optimise(
    *,
    fail_func,
    success_func,
    **model_ba_init_kwargs,
):
    try:
        ba_model = GPUBAModel(**model_ba_init_kwargs)
    except ModuleNotFoundError:
        logger.warning("GPU INFERNO module not found.")
        ba_model = BAModel(**model_ba_init_kwargs)

    def to_optimise(**run_kwargs):
        # NOTE: It is assumed here that the data being operated on will not change
        # between runs of the `to_optimise` function. Data-relevant parameters in
        # `run_kwargs` will be silently ignored!
        try:
            model_ba = ba_model.run(**run_kwargs)["model_ba"]
            scores = ba_model.calc_scores(
                model_ba=model_ba, requested=(Metrics.MPD, Metrics.ARCSINH_NME)
            )["scores"]
        except BAModelException:
            return fail_func()

        # Aim to minimise the combined score.
        # loss = scores["nme"] + scores["nmse"] + scores["mpd"] + 2 * scores["loghist"]
        loss = scores["arcsinh_nme"] + scores["mpd"]
        logger.debug(f"loss: {loss:0.6f}")
        return success_func(loss)

    # TODO Call release on GPU class when done?

    return to_optimise
