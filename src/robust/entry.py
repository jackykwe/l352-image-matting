import itertools
import logging
import math
import time

import numpy as np
import scipy.ndimage as spndimage
import skimage
from tqdm.auto import tqdm

DIVISION_EPSILON = 1e-100

def euclidean_distances(
    foreground_boundary_is,
    foreground_boundary_js,
    background_boundary_is,
    background_boundary_js,
    unknown_i,
    unknown_j
):
    """
    `foreground_boundary_is`: length-N 1D array of x-coordinates of foreground boundary pixels
    `foreground_boundary_ys`: length-N 1D array of y-coordinates of foreground boundary pixels
    `background_boundary_is`: length-N 1D array of x-coordinates of background boundary pixels
    `background_boundary_ys`: length-N 1D array of y-coordinates of background boundary pixels
    `unknown_i` and `unknown_j`: coordinates of unknown pixel for which to obtain foreground and background samples for

    Returns:
    ```
    (
        length-N 1D array of Euclidean distances of the foreground boundary pixels relative to unknown pixel,
        length-N 1D array of Euclidean distances of the background boundary pixels relative to unknown pixel
    )
    """
    foreground_boundary_distances = np.sqrt((foreground_boundary_is - unknown_i) ** 2 + (foreground_boundary_js - unknown_j) ** 2)  # 1D float array
    background_boundary_distances = np.sqrt((background_boundary_is - unknown_i) ** 2 + (background_boundary_js - unknown_j) ** 2)  # 1D float array
    return foreground_boundary_distances, background_boundary_distances


def manhattan_distances(
    foreground_boundary_is,
    foreground_boundary_js,
    background_boundary_is,
    background_boundary_js,
    unknown_i,
    unknown_j
):
    """
    `foreground_boundary_is`: length-N 1D array of x-coordinates of foreground boundary pixels
    `foreground_boundary_ys`: length-N 1D array of y-coordinates of foreground boundary pixels
    `background_boundary_is`: length-N 1D array of x-coordinates of background boundary pixels
    `background_boundary_ys`: length-N 1D array of y-coordinates of background boundary pixels
    `unknown_i` and `unknown_j`: coordinates of unknown pixel for which to obtain foreground and background samples for

    Returns:
    ```
    (
        length-N 1D array of Manhattan distances of the foreground boundary pixels relative to unknown pixel,
        length-N 1D array of Manhattan distances of the background boundary pixels relative to unknown pixel
    )
    """
    foreground_boundary_distances = np.absolute(foreground_boundary_is - unknown_i) + np.absolute(foreground_boundary_js - unknown_j)  # 1D float array
    background_boundary_distances = np.absolute(background_boundary_is - unknown_i) + np.absolute(background_boundary_js - unknown_j)  # 1D float array
    return foreground_boundary_distances, background_boundary_distances


def chebyshev_distances(
    foreground_boundary_is,
    foreground_boundary_js,
    background_boundary_is,
    background_boundary_js,
    unknown_i,
    unknown_j
):
    """
    `foreground_boundary_is`: length-N 1D array of x-coordinates of foreground boundary pixels
    `foreground_boundary_ys`: length-N 1D array of y-coordinates of foreground boundary pixels
    `background_boundary_is`: length-N 1D array of x-coordinates of background boundary pixels
    `background_boundary_ys`: length-N 1D array of y-coordinates of background boundary pixels
    `unknown_i` and `unknown_j`: coordinates of unknown pixel for which to obtain foreground and background samples for

    Returns:
    ```
    (
        length-N 1D array of Chebyshev distances of the foreground boundary pixels relative to unknown pixel,
        length-N 1D array of Chebyshev distances of the background boundary pixels relative to unknown pixel
    )
    """
    foreground_boundary_distances = np.max(
        np.vstack((
            np.absolute(foreground_boundary_is - unknown_i),
            np.absolute(foreground_boundary_js - unknown_j)
        )),
        axis=0
    )  # 1D float array
    background_boundary_distances = np.max(
        np.vstack((
            np.absolute(background_boundary_is - unknown_i),
            np.absolute(background_boundary_js - unknown_j)
        )),
        axis=0
    )  # 1D float array
    return foreground_boundary_distances, background_boundary_distances


def get_samples(
    I,
    foreground_boundary_is,
    foreground_boundary_js,
    global_foreground_boundary_pixels,
    background_boundary_is,
    background_boundary_js,
    global_background_boundary_pixels,
    unknown_i,  # this function is called in a loop, with all other arguments constant except these two
    unknown_j,  # this function is called in a loop, with all other arguments constant except these two
    foreground_samples_count,
    background_samples_count,
    scheme_config
):
    """
    `I`: H x W x C float array
    `foreground_boundary_is`: 1D array of x-coordinates of foreground boundary pixels
    `foreground_boundary_ys`: 1D array of y-coordinates of foreground boundary pixels
    `global_foreground_pixels`: len(foreground_boundary_is) x 3 float array of foreground boundary pixels
    `background_boundary_is`: 1D array of x-coordinates of background boundary pixels
    `background_boundary_ys`: 1D array of y-coordinates of background boundary pixels
    `global_background_pixels`: len(background_boundary_is) x 3 float array of background boundary pixels
    `unknown_i` and `unknown_j`: coordinates of unknown pixel for which to obtain foreground and background samples for
    `foreground_samples_count`: number of foreground pixel samples to return
    `background_samples_count`: number of background pixel samples to return
    `scheme`: takes the shape of one of the following:
    - `{"name": "global_random"}`
    - `{"name": "local_random", "nearest_candidates_count": int}`
    - `{"name": "deterministic"}`

    For an unknown pixel at coordinates `unknown_ij, the schemes behave differently:
    - `global_random`: sample `foreground_samples_count` pixels randomly from all foreground boundary pixels; likewise for background
    - `local_random`: sample `foreground_samples_count` pixels randomly from the nearest `nearest_candidates_count` foreground boundary pixels; likewise for background
    - `deterministic`: sample the nearest `foreground_samples_count` foreground boundary pixels; likewise for background

    Returns:
    ```
    (
        foreground_samples_count x C float array,  (foreground pixel values)
        background_samples_count x C float array,  (background pixel values)
    )
    ```

    Useful discussion on distance metrics at https://chris3606.github.io/GoRogue/articles/grid_components/measuring-distance.html
    """
    if scheme_config["name"] == "global_random":
        foreground_choices = np.random.choice(len(global_foreground_boundary_pixels), foreground_samples_count, replace=False)
        background_choices = np.random.choice(len(global_background_boundary_pixels), background_samples_count, replace=False)
    elif scheme_config["name"] == "local_random":
        nearest_candidates_count = scheme_config["nearest_candidates_count"]
        # choice between euclidean_distances/manhattan_distances/chebyshev_distances
        # chose Manhattan distances, gives same ranking as Euclidean but cheaper
        foreground_boundary_distances, background_boundary_distances = manhattan_distances(
            foreground_boundary_is,
            foreground_boundary_js,
            background_boundary_is,
            background_boundary_js,
            unknown_i,
            unknown_j
        )
        foreground_choices = np.random.choice(
            np.argsort(foreground_boundary_distances)[:nearest_candidates_count],
            foreground_samples_count,
            replace=False
        )
        background_choices = np.random.choice(
            np.argsort(background_boundary_distances)[:nearest_candidates_count],
            background_samples_count,
            replace=False
        )
    elif scheme_config["name"] == "deterministic":
        # choice between euclidean_distances/manhattan_distances/chebyshev_distances
        # chose Manhattan distances, gives same ranking as Euclidean but cheaper
        foreground_boundary_distances, background_boundary_distances = manhattan_distances(
            foreground_boundary_is,
            foreground_boundary_js,
            background_boundary_is,
            background_boundary_js,
            unknown_i,
            unknown_j
        )
        foreground_choices = np.argsort(foreground_boundary_distances)[:foreground_samples_count]
        background_choices = np.argsort(background_boundary_distances)[:background_samples_count]
    elif scheme_config["name"] == "deterministic_spread":
        # choice between euclidean_distances/manhattan_distances/chebyshev_distances
        # chose Manhattan distances, gives same ranking as Euclidean but cheaper
        foreground_boundary_distances, background_boundary_distances = manhattan_distances(
            foreground_boundary_is,
            foreground_boundary_js,
            background_boundary_is,
            background_boundary_js,
            unknown_i,
            unknown_j
        )
        foreground_interval = len(foreground_boundary_distances) // foreground_samples_count
        background_interval = len(background_boundary_distances) // background_samples_count
        foreground_choices = np.argsort(foreground_boundary_distances)[::foreground_interval][:foreground_samples_count]
        background_choices = np.argsort(background_boundary_distances)[::background_interval][:background_samples_count]
    else:
        raise NotImplementedError

    foreground_samples = global_foreground_boundary_pixels[foreground_choices]
    background_samples = global_background_boundary_pixels[background_choices]
    return foreground_samples, background_samples


def penalty_distance_ratio_squared(c, f, b):
    """
    `c`: 1D length-C array (C=3 for RGB image). Unknown pixel.
    `f`: 1D length-C array (C=3 for RGB image). Foreground sample.
    `b`: 1D length-C array (C=3 for RGB image). Background sample.
    `estimated_alpha`: equals `estimate_alpha(c, f, b)`

    Returns: Scalar float.

    NB. this returns the square of equation (3), because this expression (distance ratio) is used
    squared in equation (6) in the paper.
    """
    f_minus_b = f - b
    f_minus_b_squared = np.dot(f_minus_b, f_minus_b) + DIVISION_EPSILON
    c_minus_b = c - b
    c_minus_projected_c = c_minus_b - f_minus_b * np.dot(c_minus_b, f_minus_b) / f_minus_b_squared  # project c onto line between f and b
    return np.dot(c_minus_projected_c, c_minus_projected_c) / f_minus_b_squared  # NB. no square root is done for both numerator and denominator


def estimate_alpha(c, f, b):
    """
    `c`: 1D length-C array (C=3 for RGB image). Unknown pixel.
    `f`: 1D length-C array (C=3 for RGB image). Foreground sample.
    `b`: 1D length-C array (C=3 for RGB image). Background sample.

    Return: Scalar float estimate of alpha by projecting c onto the line between f and b, then calculating
    where on the line (0 if at b, 1 if at f) the projected point sits.
    """
    f_minus_b = f - b
    return np.dot(c - b, f_minus_b) / (np.dot(f_minus_b, f_minus_b) + DIVISION_EPSILON)


def solve_alpha(
        I,
        foreground_map,
        background_map,
        unknown_map,
        foreground_samples_count=20,
        background_samples_count=20,
        highest_confidence_pairs_to_select=3
    ):
    """
    `I`: H x W x C float array
    `foreground_map`: H x W bool array
    `background_map`: H x W bool array
    `unknown_map`: H x W bool array

    Returns: H x W float array
    """
    unknown_map_dilated = spndimage.binary_dilation(unknown_map, np.ones((3, 3)))
    foreground_boundary_map = foreground_map & unknown_map_dilated  # H x W bool array
    background_boundary_map = background_map & unknown_map_dilated  # H x W bool array
    foreground_boundary_is, foreground_boundary_js = foreground_boundary_map.nonzero()  # slow! do not do in a loop
    background_boundary_is, background_boundary_js = background_boundary_map.nonzero()  # slow! do not do in a loop
    # Using boolean indexing as in https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing
    foreground_boundary_pixels = I[foreground_boundary_is, foreground_boundary_js]  # slow! do not do in a loop
    background_boundary_pixels = I[background_boundary_is, background_boundary_js]  # slow! do not do in a loop

    result = np.zeros(I.shape[:2], dtype=float)
    result[foreground_map] = 1

    # try parallelise the below.

    for (unknown_i, unknown_j) in tqdm(
        zip(*unknown_map.nonzero()), total=np.count_nonzero(unknown_map),
        desc="Obtaining pixel samples and confidences for each unknown pixel",
        disable=not logging.root.isEnabledFor(logging.INFO)
    ):
        cT = I[unknown_i, unknown_j]
        FiT, BjT = get_samples(
            I,
            foreground_boundary_is,
            foreground_boundary_js,
            foreground_boundary_pixels,
            background_boundary_is,
            background_boundary_js,
            background_boundary_pixels,
            unknown_i,
            unknown_j,
            foreground_samples_count,
            background_samples_count,
            scheme_config={"name": "deterministic_spread"}
        )
        # Names now in 2D world (what we are familiar with)
        cT_minus_FiT = cT - FiT  # foreground_samples_count x C float array
        cT_minus_FiT_squared = np.sum(cT_minus_FiT * cT_minus_FiT, axis=1)  # 1D length-foreground_samples_count float array
        penalty_foreground_exparg = -cT_minus_FiT_squared / (np.min(cT_minus_FiT_squared) + DIVISION_EPSILON)  # this is dividing by D_F^2;  # 1D length-foreground_samples_count float array
        cT_minus_BjT = cT - BjT   # background_samples_count x C float array
        cT_minus_BjT_squared = np.sum(cT_minus_BjT * cT_minus_BjT, axis=1)  # 1D length-background_samples_count float array
        penalty_background_exparg = -cT_minus_BjT_squared / (np.min(cT_minus_BjT_squared) + DIVISION_EPSILON)  # this is dividing by D_B^2;  # 1D length-background_samples_count float array
        penalty_foreground_background = np.exp(np.repeat(penalty_foreground_exparg, background_samples_count) + np.tile(penalty_background_exparg, foreground_samples_count))  # 1D length-(foreground_samples_count * background_samples_count) float array

        # Names now in 3D
        Fi_minus_Bj_3D = (np.repeat(FiT, background_samples_count, axis=0) - np.tile(BjT, (foreground_samples_count, 1)))[:, :, np.newaxis]  # (foreground_samples_count * background_samples_count) x C x 1 float array
        FiT_minus_BjT_3D = np.transpose(Fi_minus_Bj_3D, (0, 2, 1))  # (foreground_samples_count * background_samples_count) x 1 x C float array
        Fi_minus_Bj_squared_3D = FiT_minus_BjT_3D @ Fi_minus_Bj_3D + DIVISION_EPSILON # (foreground_samples_count * background_samples_count) x 1 x 1 float array
        alpha_premultiplier_3D = FiT_minus_BjT_3D / Fi_minus_Bj_squared_3D  # this subexpression is named, as it's useful again later when estimating alphas (see estimated_alphas variable)
        Aij_3D = \
            (
                np.tile(np.eye(3).reshape(1, 3, 3), (foreground_samples_count * background_samples_count, 1, 1)) \
                - (Fi_minus_Bj_3D @ alpha_premultiplier_3D)
            ) / Fi_minus_Bj_squared_3D  # (foreground_samples_count * background_samples_count) x C x C float array
        c_minus_Bj_3D = np.tile(cT_minus_BjT, (foreground_samples_count, 1))[:, :, np.newaxis] # (foreground_samples_count * background_samples_count) x C x 1 float array
        cT_minus_BjT_3D = np.transpose(c_minus_Bj_3D, (0, 2, 1))  # (foreground_samples_count * background_samples_count) x 1 x C float array

        confidences = -np.squeeze(cT_minus_BjT_3D @ Aij_3D @ c_minus_Bj_3D) * penalty_foreground_background  # 1D length-(foreground_samples_count * background_samples_count) float array

        highest_confidence_argsort = np.argsort(confidences)[-highest_confidence_pairs_to_select:]
        estimated_alphas = (alpha_premultiplier_3D[highest_confidence_argsort, :, :] @ c_minus_Bj_3D[highest_confidence_argsort, :, :]).squeeze()  # 1D length-(highest_confidence_pairs_to_select) float array
        result[unknown_i, unknown_j] = np.average(estimated_alphas, weights=np.exp(confidences[highest_confidence_argsort]))

    result = np.clip(result, 0, 1)  # there tends to be some alphas that lie outside of [0, 1]...
    # np.save("helpme.npy", result)
    # result = np.load("helpme.npy")
    skimage.io.imshow(result)
    skimage.io.show()



    # print("entry")
    pass
