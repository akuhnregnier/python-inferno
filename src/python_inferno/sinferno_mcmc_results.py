# -*- coding: utf-8 -*-
import joblib


def get_sinferno_mcmc_results(*, method_index, r_hat):
    r_hat_hash = joblib.hashing.hash(r_hat)

    if method_index == 0 and r_hat_hash == "64b6b959cc356ad97ff96fd04ac04431":
        return {
            "Total Duration": "571.68 seconds",
            "Total Repetitions": 500000,
            "Maximal objective value": -0.825924,
            "overall_scale": 0.756661,
            "fapar_factor": 0.773065,
            "fapar_factor2": 0.647868,
            "fapar_factor3": 0.505033,
            "fapar_centre": 0.132776,
            "fapar_centre2": 0.711452,
            "fapar_centre3": 0.839562,
            "fapar_shape": 0.798137,
            "fapar_shape2": 0.729013,
            "fapar_shape3": 0.254109,
            "fapar_weight": 0.999218,
            "fapar_weight2": 0.355863,
            "fapar_weight3": 0.286084,
            "dryness_weight": 0.14085,
            "dryness_weight2": 0.998948,
            "dryness_weight3": 0.976527,
            "fuel_weight": 0.659687,
            "fuel_weight2": 0.997677,
            "fuel_weight3": 0.339484,
            "crop_f": 0.271608,
            "dry_day_factor": 0.305584,
            "dry_day_factor2": 0.0295176,
            "dry_day_factor3": 0.787272,
            "dry_day_centre": 0.937701,
            "dry_day_centre2": 0.931264,
            "dry_day_centre3": 0.790028,
            "dry_day_shape": 0.581005,
            "dry_day_shape2": 0.871778,
            "dry_day_shape3": 0.175816,
            "fuel_build_up_factor": 0.754474,
            "fuel_build_up_factor2": 0.57887,
            "fuel_build_up_factor3": 0.487006,
            "fuel_build_up_centre": 0.832668,
            "fuel_build_up_centre2": 0.475441,
            "fuel_build_up_centre3": 0.734438,
            "fuel_build_up_shape": 0.99233,
            "fuel_build_up_shape2": 0.811927,
            "fuel_build_up_shape3": 0.766936,
            "temperature_factor": 0.897744,
            "temperature_factor2": 0.479224,
            "temperature_factor3": 0.824314,
            "temperature_centre": 0.95663,
            "temperature_centre2": 0.309139,
            "temperature_centre3": 0.133024,
            "temperature_shape": 0.945011,
            "temperature_shape2": 0.00656756,
            "temperature_shape3": 0.767039,
            "temperature_weight": 0.962093,
            "temperature_weight2": 0.999953,
            "temperature_weight3": 0.271545,
        }

    if method_index == 1 and r_hat_hash == "29bc947f4ac05904975bf9960fe030d7":
        return {
            "Total Duration": "11347.52 seconds",
            "Total Repetitions": 10000000,
            "Maximal objective value": -0.790425,
            "overall_scale": 0.720834,
            "fapar_factor": 0.511116,
            "fapar_factor2": 0.577206,
            "fapar_factor3": 0.637802,
            "fapar_centre": 0.87563,
            "fapar_centre2": 0.768641,
            "fapar_centre3": 0.0266548,
            "fapar_shape": 0.16654,
            "fapar_shape2": 0.300605,
            "fapar_shape3": 0.634744,
            "fapar_weight": 0.665337,
            "fapar_weight2": 0.610508,
            "fapar_weight3": 0.932275,
            "dryness_weight": 0.809122,
            "dryness_weight2": 0.999386,
            "dryness_weight3": 0.467414,
            "fuel_weight": 0.965497,
            "fuel_weight2": 0.998969,
            "fuel_weight3": 0.262581,
            "crop_f": 0.146281,
            "dry_day_factor": 0.381777,
            "dry_day_factor2": 0.0458098,
            "dry_day_factor3": 0.31178,
            "dry_day_centre": 0.867588,
            "dry_day_centre2": 0.0976704,
            "dry_day_centre3": 0.845314,
            "dry_day_shape": 0.0720506,
            "dry_day_shape2": 0.996051,
            "dry_day_shape3": 0.416673,
            "litter_pool_factor": 0.428134,
            "litter_pool_factor2": 0.726189,
            "litter_pool_factor3": 0.878052,
            "litter_pool_centre": 0.792662,
            "litter_pool_centre2": 0.0881105,
            "litter_pool_centre3": 0.39505,
            "litter_pool_shape": 0.31479,
            "litter_pool_shape2": 0.364353,
            "litter_pool_shape3": 0.286226,
            "temperature_factor": 0.643515,
            "temperature_factor2": 0.854991,
            "temperature_factor3": 0.431428,
            "temperature_centre": 0.695461,
            "temperature_centre2": 0.190696,
            "temperature_centre3": 0.626433,
            "temperature_shape": 0.0771464,
            "temperature_shape2": 0.00330108,
            "temperature_shape3": 0.66237,
            "temperature_weight": 0.999357,
            "temperature_weight2": 0.999941,
            "temperature_weight3": 0.822052,
        }

    if method_index == 2 and r_hat_hash == "0786daeb9d961eb85b52055c70b0fa30":
        return {
            "Total Duration": "13049.75 seconds",
            "Total Repetitions": 10000000,
            "Maximal objective value": -0.791012,
            "overall_scale": 0.475949,
            "fapar_factor": 0.317584,
            "fapar_factor2": 0.82669,
            "fapar_factor3": 0.0534359,
            "fapar_centre": 0.420553,
            "fapar_centre2": 0.862683,
            "fapar_centre3": 0.168037,
            "fapar_shape": 0.250759,
            "fapar_shape2": 0.268546,
            "fapar_shape3": 0.787715,
            "fapar_weight": 0.999459,
            "fapar_weight2": 0.778146,
            "fapar_weight3": 0.928959,
            "dryness_weight": 0.915004,
            "dryness_weight2": 0.999917,
            "dryness_weight3": 0.492747,
            "fuel_weight": 0.98597,
            "fuel_weight2": 0.934137,
            "fuel_weight3": 0.960588,
            "crop_f": 0.0263834,
            "dry_bal_factor": 0.962605,
            "dry_bal_factor2": 0.93899,
            "dry_bal_factor3": 0.433047,
            "dry_bal_centre": 0.138033,
            "dry_bal_centre2": 0.365715,
            "dry_bal_centre3": 0.0307901,
            "dry_bal_shape": 0.52517,
            "dry_bal_shape2": 0.0430308,
            "dry_bal_shape3": 0.843301,
            "fuel_build_up_factor": 0.636958,
            "fuel_build_up_factor2": 0.251106,
            "fuel_build_up_factor3": 0.822577,
            "fuel_build_up_centre": 0.952151,
            "fuel_build_up_centre2": 0.600703,
            "fuel_build_up_centre3": 0.828532,
            "fuel_build_up_shape": 0.178402,
            "fuel_build_up_shape2": 0.569221,
            "fuel_build_up_shape3": 0.243468,
            "temperature_factor": 0.998245,
            "temperature_factor2": 0.335185,
            "temperature_factor3": 0.728091,
            "temperature_centre": 0.184416,
            "temperature_centre2": 0.393123,
            "temperature_centre3": 0.993337,
            "temperature_shape": 0.582922,
            "temperature_shape2": 0.25177,
            "temperature_shape3": 0.405431,
            "temperature_weight": 0.529092,
            "temperature_weight2": 0.999999,
            "temperature_weight3": 0.957872,
        }

    if method_index == 3 and r_hat_hash == "244692fa2be97a680168f217cc4f272b":
        return {
            "Total Duration": "11466.13 seconds",
            "Total Repetitions": 10000000,
            "Maximal objective value": -0.759895,
            "overall_scale": 0.91606,
            "fapar_factor": 0.348339,
            "fapar_factor2": 0.415668,
            "fapar_factor3": 0.837358,
            "fapar_centre": 0.829139,
            "fapar_centre2": 8.32294e-05,
            "fapar_centre3": 0.789299,
            "fapar_shape": 0.161696,
            "fapar_shape2": 0.549607,
            "fapar_shape3": 1,
            "fapar_weight": 0.543652,
            "fapar_weight2": 0.585628,
            "fapar_weight3": 0.276864,
            "dryness_weight": 1,
            "dryness_weight2": 0.99465,
            "dryness_weight3": 0.987901,
            "fuel_weight": 0.826918,
            "fuel_weight2": 0.999576,
            "fuel_weight3": 0.372534,
            "crop_f": 0.611529,
            "dry_bal_factor": 0.878597,
            "dry_bal_factor2": 0.975843,
            "dry_bal_factor3": 0.12248,
            "dry_bal_centre": 0.39114,
            "dry_bal_centre2": 0.346427,
            "dry_bal_centre3": 0.173164,
            "dry_bal_shape": 0.400989,
            "dry_bal_shape2": 0.683557,
            "dry_bal_shape3": 0,
            "litter_pool_factor": 0.202924,
            "litter_pool_factor2": 0.578406,
            "litter_pool_factor3": 0.016949,
            "litter_pool_centre": 0.229778,
            "litter_pool_centre2": 0.5212,
            "litter_pool_centre3": 0.325145,
            "litter_pool_shape": 0.393834,
            "litter_pool_shape2": 0.37194,
            "litter_pool_shape3": 1,
            "temperature_factor": 0.645752,
            "temperature_factor2": 0.15497,
            "temperature_factor3": 0.0275504,
            "temperature_centre": 0.417497,
            "temperature_centre2": 0.692484,
            "temperature_centre3": 0.41348,
            "temperature_shape": 0.188169,
            "temperature_shape2": 0.276339,
            "temperature_shape3": 0.463464,
            "temperature_weight": 0.999927,
            "temperature_weight2": 0.999988,
            "temperature_weight3": 0.0531827,
        }

    raise ValueError
