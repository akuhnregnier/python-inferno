#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import partial

import numpy as np

from python_inferno.metrics import mpd


def arcsinh_transform(data, factor):
    return np.arcsinh(factor * data)


# True
obs = np.array(
    [1e-19, 1e-19, 1e-19, 1e-10, 1e-9, 1e-3, 1e-4, 1e-9, 1e-10, 1e-15, 1e-16, 1e-19]
).reshape(12, 1)

# orig-pred
a0 = np.array(
    [1e-10, 1e-10, 1e-10, 1e-7, 1e-8, 1e-4, 1e-5, 1e-8, 1e-9, 1e-12, 1e-12, 1e-11]
).reshape(12, 1)
# improved preds
# better low magnitude values
a1 = np.array(
    [1e-13, 1e-13, 1e-13, 1e-9, 1e-8, 1e-4, 1e-5, 3e-9, 1e-9, 1e-14, 1e-15, 1e-14]
).reshape(12, 1)
# better high magnitude values
a2 = np.array(
    [1e-10, 1e-10, 1e-10, 1e-7, 1e-8, 5e-4, 5e-5, 1e-8, 1e-9, 1e-12, 1e-12, 1e-11]
).reshape(12, 1)


def get_data():
    data = (obs, a0, a1, a2)
    for mode in ["normal", "arcsinh", "log"]:
        print(f"- {mode}")
        if mode == "normal":
            yield data
        elif mode == "arcsinh":
            for factor in [1e4, 1e5, 1e6, 1e7, 1e8]:
                print(f"-- {factor:0.0e}")
                yield tuple(map(partial(arcsinh_transform, factor=factor), data))
        elif mode == "log":
            yield tuple(map(np.log, data))
        else:
            raise ValueError()


for obs, a0, a1, a2 in get_data():
    ref = mpd(obs=obs, pred=a0)
    a1_nme = mpd(obs=obs, pred=a1) / ref
    a2_nme = mpd(obs=obs, pred=a2) / ref
    print(f"better-low : {a1_nme:0.2f}")
    print(f"better-high: {a2_nme:0.2f}")
    print(f"relative   : {a1_nme / a2_nme:0.2f}")
    print()
