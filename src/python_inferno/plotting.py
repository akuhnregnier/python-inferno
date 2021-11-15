# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from wildfires.analysis import cube_plotting

from python_inferno.metrics import calculate_phase, calculate_phase_2d


def lin_cube_plotting(*, data, title):
    cube_plotting(
        data,
        title=title,
        nbins=9,
        vmin_vmax_percentiles=(5, 95),
        fig=plt.figure(figsize=(12, 5)),
        colorbar_kwargs=dict(format="%.1e"),
    )


def log_cube_plotting(*, data, title, raw_data):
    cube_plotting(
        data,
        title=title,
        boundaries=np.geomspace(*np.quantile(raw_data[raw_data > 0], [0.05, 0.95]), 8),
        fig=plt.figure(figsize=(12, 5)),
        colorbar_kwargs=dict(format="%.1e"),
    )


def plot_phase_map(*, data, title):
    assert len(data.shape) == 3, "Need time, x, y"
    phase = np.ma.MaskedArray(
        calculate_phase_2d(np.ascontiguousarray(np.ma.getdata(data))),
        mask=np.all(np.ma.getmaskarray(data), axis=0),
    )
    phase += np.pi
    # Phase is now in [0, 2pi].
    phase -= calculate_phase(
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(12, 1)
    )[0]
    phase %= 2 * np.pi
    # Phase is now in [0, 2pi].
    #
    # Transform to [0.5, 12.5].
    phase *= 12 / (2 * np.pi)
    phase += 0.5

    cube_plotting(
        phase,
        title=title,
        boundaries=np.linspace(0.5, 12.5, 13),
        cmap="twilight",
        fig=plt.figure(figsize=(12, 5)),
        colorbar_kwargs=dict(format="%.1e"),
    )


def plotting(
    *,
    exp_name,
    exp_key=None,
    raw_data,
    model_ba_2d_data,
    hist_bins,
    arcsinh_adj_factor,
    arcsinh_factor,
    scores=None,
    save_dir=None,
):
    if scores is not None:
        arcsinh_nme = scores["arcsinh_nme"]
        mpd = scores["mpd"]
        total = arcsinh_nme + mpd
        title = (
            f"{exp_name}, arcsinh NME: {arcsinh_nme:0.2f}, MPD: {mpd:0.2f}, "
            f"Total: {total:0.2f}"
        )
    else:
        title = exp_name

    if exp_key is None:
        exp_key = exp_name.replace(" ", "_").lower()

    logger.info("Plotting hist")
    plt.figure()
    plt.hist(raw_data, bins=hist_bins)
    plt.yscale("log")
    plt.title(title)
    if save_dir is not None:
        plt.savefig(save_dir / f"hist_{exp_key}.png")
    plt.close()

    log_cube_plotting(data=model_ba_2d_data, title=title, raw_data=raw_data)
    if save_dir is not None:
        plt.savefig(save_dir / f"BA_map_log_{exp_key}.png")
    plt.close()

    lin_cube_plotting(data=model_ba_2d_data, title=title)
    if save_dir is not None:
        plt.savefig(save_dir / f"BA_map_lin_{exp_key}.png")
    plt.close()

    lin_cube_plotting(
        data=arcsinh_adj_factor * np.arcsinh(arcsinh_factor * model_ba_2d_data),
        title=title,
    )
    if save_dir is not None:
        plt.savefig(save_dir / f"BA_map_arcsinh_{exp_key}.png")
    plt.close()

    plot_phase_map(
        data=model_ba_2d_data,
        title=title,
    )
    if save_dir is not None:
        plt.savefig(save_dir / f"Phase_map_{exp_key}.png")
    plt.close()
