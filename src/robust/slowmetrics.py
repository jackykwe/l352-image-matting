import numpy as np

import utils


def slow_distance_ratio_squared(c, f, b):
    """
    `c`: 1D length-C float array
    `f`: 1D length-C float array
    `b`: 1D length-C float array
    """
    divisor = (np.dot(f - b, f - b) + utils.DIVISION_EPSILON)
    alpha = np.dot(c - b, f - b) / divisor
    numerator = c - (alpha * f + (1 - alpha) * b)
    return np.dot(numerator, numerator) / divisor


def slow_penalty_foreground(c, f, all_fs):
    """
    `c`: 1D length-C float array
    `f`: 1D length-C float array
    `all_fs`: (foreground_samples_count) x C float array
    """
    fi_minus_c = all_fs - c
    DF_squared = np.min(np.sum(fi_minus_c * fi_minus_c, axis=1)) + utils.DIVISION_EPSILON
    return np.exp(-np.dot(f - c, f - c) / DF_squared)

def slow_penalty_background(c, b, all_bs):
    """
    `c`: 1D length-C float array
    `b`: 1D length-C float array
    `all_bs`: (background_samples_count) x C float array
    """
    bj_minus_c = all_bs - c
    DB_squared = np.min(np.sum(bj_minus_c * bj_minus_c, axis=1)) + utils.DIVISION_EPSILON
    return np.exp(-np.dot(b - c, b - c) / DB_squared)

def slow_confidence_exparg(c, f, b, all_fs, all_bs, sigma_squared):
    return -slow_distance_ratio_squared(c, f, b) * slow_penalty_foreground(c, f, all_fs) * slow_penalty_background(c, b, all_bs) / sigma_squared
