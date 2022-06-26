# -*- coding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from loguru import logger
from sklearn.metrics import r2_score

from .cache import mark_dependency
from .configuration import N_pft_groups, land_pts, npft
from .data import get_processed_climatological_data
from .metrics import Metrics, loghist, nmse
from .multi_timestep_inferno import (
    _get_checks_failed_mask,
    _get_diagnostics,
    multi_timestep_inferno,
)
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
    land_point=-1,
    **kwargs,
):
    return unpack_wrapped(multi_timestep_inferno)(
        ignition_method=1,
        flammability_method=2,
        dryness_method=dryness_method,
        fuel_build_up_method=fuel_build_up_method,
        include_temperature=include_temperature,
        overall_scale=overall_scale,
        land_point=land_point,
        # These are not used for ignition mode 1, nor do they contain a temporal
        # coordinate.
        pop_den=dummy_pop_den,
        flash_rate=dummy_flash_rate,
        **data_dict,
        **kwargs,
    )


def mpd_check(mpd_ignored):
    if mpd_ignored > MPD_IGNORE_THRES:
        # Ensure that not too many samples are ignored.
        raise BAModelException()


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
    mpd_check(ignored)

    return mpd_val, ignored


def _calculate_scores_from_avg_ba(*, avg_ba, mon_avg_gfed_ba_1d, requested):
    assert requested

    if np.all(np.abs(avg_ba) < 1e-15):
        raise BAModelException()

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
        assert not np.any(np.ma.getmaskarray(mon_avg_gfed_ba_1d))
        y_pred = np.ma.getdata(avg_ba).ravel()
        y_true = np.ma.getdata(mon_avg_gfed_ba_1d).ravel()
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

    return scores


def calculate_scores(
    *, model_ba, cons_monthly_avg, mon_avg_gfed_ba_1d, requested=Metrics
):
    # Calculate monthly averages.
    avg_ba = cons_monthly_avg.cons_monthly_average_data(model_ba)
    assert avg_ba.shape == mon_avg_gfed_ba_1d.shape

    return (
        _calculate_scores_from_avg_ba(
            avg_ba=avg_ba,
            mon_avg_gfed_ba_1d=mon_avg_gfed_ba_1d,
            requested=requested,
        ),
        avg_ba,
    )


class ModelParams:
    def __init__(
        self,
        *,
        dryness_method,
        fuel_build_up_method,
        include_temperature,
        disc_params,
    ):
        self.dryness_method = int(dryness_method)
        self.fuel_build_up_method = int(fuel_build_up_method)
        self.include_temperature = int(include_temperature)
        self.init_disc_params(**disc_params)

    @staticmethod
    def get_proc_func(kwargs):
        def process_key_from_kwargs(key, n_target=N_pft_groups, dtype=np.float64):
            n_source = None

            if f"{key}2" in kwargs:
                if f"{key}3" in kwargs:
                    n_source = 3
            else:
                n_source = 1

            if n_source is None:
                # Prevent cases like key, key2, (and not key3 too).
                raise ValueError

            return process_param_inplace(
                kwargs=kwargs,
                name=key,
                n_source=n_source,
                n_target=n_target,
                dtype=dtype,
            )

        return process_key_from_kwargs

    @mark_dependency
    def process_kwargs(self, **kwargs):
        dummy_params = np.ones(N_pft_groups, dtype=np.float64)

        processed_kwargs = dict(overall_scale=kwargs.pop("overall_scale", 1.0))

        if "land_point" in kwargs:
            processed_kwargs["land_point"] = int(kwargs["land_point"])

        process_key_from_kwargs = self.get_proc_func(kwargs)

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

    def init_disc_params(self, **kwargs):
        process_key_from_kwargs = self.get_proc_func(kwargs)

        self.rain_f = (
            process_key_from_kwargs("rain_f") if self.dryness_method == 2 else None
        )
        self.vpd_f = (
            process_key_from_kwargs("vpd_f") if self.dryness_method == 2 else None
        )

        self.n_samples_pft = (
            process_key_from_kwargs("fuel_build_up_n_samples", dtype=np.int64)
            if self.fuel_build_up_method == 1
            else None
        )

        self.litter_tc = (
            process_key_from_kwargs("litter_tc", n_target=npft)
            if self.fuel_build_up_method == 2
            else None
        )
        self.leaf_f = (
            process_key_from_kwargs("leaf_f", n_target=npft)
            if self.fuel_build_up_method == 2
            else None
        )

        self.average_samples = int(kwargs["average_samples"])

    @property
    def disc_params(self):
        return dict(
            rain_f=self.rain_f,
            vpd_f=self.vpd_f,
            n_samples_pft=self.n_samples_pft,
            litter_tc=self.litter_tc,
            leaf_f=self.leaf_f,
            average_samples=self.average_samples,
        )


class BAModel(ModelParams):
    @mark_dependency
    def __init__(
        self,
        *,
        dryness_method,
        fuel_build_up_method,
        include_temperature,
        _uncached_data=False,
        **kwargs,
    ):
        super().__init__(
            dryness_method=dryness_method,
            fuel_build_up_method=fuel_build_up_method,
            include_temperature=include_temperature,
            disc_params=kwargs,
        )

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

        data_func = (
            get_processed_climatological_data
            if not _uncached_data
            else get_processed_climatological_data._wrapped_func._orig_func
        )

        (data_dict, self.mon_avg_gfed_ba_1d, self.jules_time_coord) = data_func(
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
        self._cons_monthly_avg = ConsMonthlyAvgNoMask(self.jules_time_coord, L=land_pts)

        self.Nt = self.data_dict["pstar"].shape[0]

        # NOTE This is not recalculated every time the BA model is run. Changes to
        # relevant quantities (e.g. temperature) will therefore not affect the
        # times/pfts/points at which the model is run!
        self.checks_failed = self._get_checks_failed_mask()

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
            checks_failed=self.checks_failed,
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
            cons_monthly_avg=self._cons_monthly_avg,
            mon_avg_gfed_ba_1d=self.mon_avg_gfed_ba_1d,
            requested=requested,
        )

        return dict(
            avg_ba=avg_ba,
            scores=scores,
        )

    def _get_checks_failed_mask(self):
        return transform_dtype(_get_checks_failed_mask)(
            t1p5m_tile=self.data_dict["t1p5m_tile"],
            q1p5m_tile=self.data_dict["q1p5m_tile"],
            pstar=self.data_dict["pstar"],
            sthu_soilt_single=self.data_dict["sthu_soilt_single"],
            ls_rain=self.data_dict["ls_rain"],
            con_rain=self.data_dict["con_rain"],
        )

    def _get_diagnostics(self):
        return _get_diagnostics(
            t1p5m_tile=self.data_dict["t1p5m_tile"],
            q1p5m_tile=self.data_dict["q1p5m_tile"],
            pstar=self.data_dict["pstar"],
            ls_rain=self.data_dict["ls_rain"],
            con_rain=self.data_dict["con_rain"],
        )


class GPUBAModel(BAModel):
    def __init__(self, **kwargs):

        logger.info("GPUBAModel init.")

        super().__init__(**kwargs)

        self._gpu_inferno = transform_dtype(self._gpu_class)(
            ignition_method=1,
            flammability_method=2,
            dryness_method=self.dryness_method,
            fuel_build_up_method=self.fuel_build_up_method,
            include_temperature=self.include_temperature,
            Nt=self.Nt,
            t1p5m_tile=self.data_dict["t1p5m_tile"].ravel(),
            q1p5m_tile=self.data_dict["q1p5m_tile"].ravel(),
            pstar=self.data_dict["pstar"].ravel(),
            sthu_soilt_single=self.data_dict["sthu_soilt_single"].ravel(),
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
            checks_failed=BAModel._get_checks_failed_mask(self),
        )

    @property
    def _gpu_class(self):
        from .py_gpu_inferno import GPUInferno

        return GPUInferno

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

    def _get_diagnostics(self):
        return self._gpu_inferno.get_diagnostics()

    def release(self):
        self._gpu_inferno.release()


class ModAvgCropMixin:
    def _mod_avg_crop(self):
        # NOTE This is a workaround based on the assumption that cropland fraction is
        # slowly changing, meaning that there should be little difference between
        # averaging from original timesteps to 12 months, vs. aggregated timesteps to
        # 12 months (as is being done below). This is done for computational
        # convenience at this point.
        assert not np.any(np.ma.getmaskarray(self.obs_pftcrop_1d.mask))
        self.obs_pftcrop_1d = self._cons_monthly_avg.cons_monthly_average_data(
            np.ma.getdata(self.obs_pftcrop_1d)
        )
        assert self.obs_pftcrop_1d.shape == (12, land_pts)
        return self.obs_pftcrop_1d


class GPUConsAvgBAModel(GPUBAModel, ModAvgCropMixin):
    def __init__(self, **kwargs):
        logger.info("GPUConsAvgBAModel init.")
        super().__init__(**kwargs)
        # Modify crop variable using temporal averaging.
        self._mod_avg_crop()

    @property
    def _gpu_class(self):
        from .py_gpu_inferno import GPUInfernoAvg

        return partial(GPUInfernoAvg, weights=self._cons_monthly_avg.weights)

    def calc_scores(self, *, model_ba, requested):
        # NOTE - `model_ba` ~ `avg_ba` in this case since conservative averaging is
        # done as part of the kernel.
        scores = _calculate_scores_from_avg_ba(
            avg_ba=model_ba,
            mon_avg_gfed_ba_1d=self.mon_avg_gfed_ba_1d,
            requested=requested,
        )

        return dict(
            avg_ba=model_ba,
            scores=scores,
        )


class GPUConsAvgScoreBAModel(GPUBAModel, ModAvgCropMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def _gpu_class(self):
        from .py_gpu_inferno import GPUInfernoAvgScore

        return partial(
            GPUInfernoAvgScore,
            weights=self._cons_monthly_avg.weights,
            obs_data=self.mon_avg_gfed_ba_1d,
            # NOTE This also changes the stored instance variable.
            obs_pftcrop=self._mod_avg_crop(),
        )

    def get_model_ba(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def calc_scores(self, *args, **kwargs):
        raise NotImplementedError

    def get_scores(
        self,
        *,
        requested,
        crop_f=1.0,
        **kwargs,
    ):
        if set(requested) != set((Metrics.MPD, Metrics.ARCSINH_NME)):
            raise NotImplementedError

        processed_kwargs = self.process_kwargs(**kwargs)

        return transform_dtype(self._gpu_inferno.score)(
            **dict(crop_f=crop_f, **processed_kwargs)
        )


def gen_to_optimise(
    *,
    fail_func,
    success_func,
    _uncached_data=True,
    **model_ba_init_kwargs,
):
    requested = (Metrics.MPD, Metrics.ARCSINH_NME)
    try:
        score_model = GPUConsAvgScoreBAModel(
            _uncached_data=_uncached_data, **model_ba_init_kwargs
        )

        def to_optimise(**run_kwargs):
            scores = score_model.get_scores(requested=requested, **run_kwargs)

            try:
                mpd_check(scores["mpd_ignored"])
            except BAModelException:
                return fail_func()

            # Aim to minimise the combined score.
            loss = scores["arcsinh_nme"] + scores["mpd"]
            if np.isnan(loss):
                return fail_func()

            return success_func(loss)

    except ModuleNotFoundError:
        logger.warning("GPU INFERNO module not found.")
        ba_model = BAModel(_uncached_data=_uncached_data, **model_ba_init_kwargs)

        def to_optimise(**run_kwargs):
            try:
                model_ba = ba_model.run(**run_kwargs)["model_ba"]
                scores = ba_model.calc_scores(model_ba=model_ba, requested=requested)[
                    "scores"
                ]
            except BAModelException:
                return fail_func()

            # Aim to minimise the combined score.
            loss = scores["arcsinh_nme"] + scores["mpd"]
            return success_func(loss)

    # NOTE: It is assumed here that the data being operated on will not change
    # between runs of the `to_optimise` function. Data-relevant parameters in
    # `run_kwargs` will be silently ignored!
    return to_optimise
