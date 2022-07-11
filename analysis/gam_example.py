#!/Users/alexander/separate_miniconda/envs/python-inferno/bin/python
# -*- coding: utf-8 -*-

# Use the correct R installation.

# isort: off
import os

os.environ["R_HOME"] = "/Users/alexander/separate_miniconda/envs/python-inferno/lib/R"
# isort: on

import os
from contextlib import contextmanager
from time import time

import numpy as np
import rpy2.robjects as ro
import scipy.stats
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter


@contextmanager
def numpy2ri_context():
    try:
        numpy2ri.activate()
        yield
    finally:
        # Deactivate to prep for repeated call of `activate()` elsewhere.
        numpy2ri.deactivate()


if __name__ == "__main__":
    # Measure wall time.
    start = time()
    ro.r("library(mgcv)")

    # formula_str = "y~a+b+c+d"
    formula_str = "y~a"

    N = 1000000

    rng = np.random.default_rng(0)

    y = rng.random(N)
    a = rng.random(N)
    a2 = rng.random(N)
    b = rng.random(N)
    c = rng.random(N)
    d = rng.random(N)

    # with localconverter(ro.default_converter + pandas2ri.converter), numpy2ri_context():
    with localconverter(ro.default_converter), numpy2ri_context():
        # Transfer data to R.
        ro.globalenv["y"] = y

        ro.globalenv["a"] = a
        ro.globalenv["a2"] = a2
        ro.globalenv["b"] = b
        ro.globalenv["c"] = c
        ro.globalenv["d"] = d

        # Evaluate the formula string.
        ro.r(f"gam_formula = as.formula({formula_str})")

        data_list_str = "list(a=a, b=b, c=c, d=d)"
        new_data_list_str = "list(a=a2, b=b, c=c, d=d)"

        ro.r(
            f"""
            fitted_gam = gam(
                gam_formula,
                method="REML",
                family=quasibinomial(link="logit"),
                data={data_list_str},
            )
        """
        )

        gam_pred = ro.r('predict(fitted_gam, type="response")')
        print("Gam pred:")
        print(scipy.stats.describe(gam_pred))

        # Used to get from link to response scale.
        inv_link = ro.r("family(fitted_gam)$linkinv")

        ro.globalenv["grid_pred"] = ro.r(
            f"""predict(
                fitted_gam,
                newdata={new_data_list_str},
                type="link",
                se.fit=TRUE
            )"""
        )

        print(scipy.stats.describe(ro.r("grid_pred$fit")))
        print(scipy.stats.describe(ro.r("grid_pred$se.fit")))

        print(scipy.stats.describe(inv_link(np.random.random(N))))

    print("time taken:", time() - start)
