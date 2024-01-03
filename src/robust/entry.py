import itertools
import logging
import time

import numpy as np
import scipy as sp
import scipy.ndimage as spndimage
import skimage
from tqdm.auto import tqdm


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
        1D foreground_samples_count int array,    (foreground pixel x-coordinates)
        1D foreground_samples_count int array,    (foreground pixel y-coordinates)
        background_samples_count x C float array,  (background pixel values)
        1D background_samples_count int array,    (background pixel x-coordinates)
        1D background_samples_count int array,    (background pixel y-coordinates)
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
        foreground_boundary_distances, background_boundary_distances = euclidean_distances(
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
        foreground_boundary_distances, background_boundary_distances = euclidean_distances(
            foreground_boundary_is,
            foreground_boundary_js,
            background_boundary_is,
            background_boundary_js,
            unknown_i,
            unknown_j
        )
        foreground_choices = np.argsort(foreground_boundary_distances)[:foreground_samples_count]
        background_choices = np.argsort(background_boundary_distances)[:background_samples_count]
    else:
        raise NotImplementedError

    foreground_samples = global_foreground_boundary_pixels[foreground_choices]
    foreground_sample_is = foreground_boundary_is[foreground_choices]
    foreground_sample_js = foreground_boundary_js[foreground_choices]
    background_samples = global_background_boundary_pixels[background_choices]
    background_sample_is = background_boundary_is[background_choices]
    background_sample_js = background_boundary_js[background_choices]
    return (
        foreground_samples,
        foreground_sample_is,
        foreground_sample_js,
        background_samples,
        background_sample_is,
        background_sample_js
    )


# TODO: unused?
def estimate_alpha(c, f, b):
    """
    `c`: 1D length-C array (C=3 for RGB image). Unknown pixel.
    `f`: 1D length-C array (C=3 for RGB image). Foreground sample.
    `b`: 1D length-C array (C=3 for RGB image). Background sample.

    Return: Scalar float estimate of alpha by projecting c onto the line between f and b, then calculating
    where on the line (0 if at b, 1 if at f) the projected point sits.
    """
    f_minus_b = f - b
    return np.dot(c - b, f_minus_b) / np.dot(f_minus_b, f_minus_b)


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
    projected_c = b + np.dot(c - b, f_minus_b) * (f_minus_b) / np.dot(f_minus_b, f_minus_b)  # project c onto line between f and b
    numerator = c - projected_c
    numerator = np.dot(numerator, numerator)  # NB. no square root is done
    denominator = f - b
    denominator = np.dot(denominator, denominator)  # NB. no square root is done
    return numerator / denominator  # TODO: division by zero


def solve_alpha(
        I,
        foreground_map,
        background_map,
        unknown_map,
        foreground_samples_count=20,
        background_samples_count=20,
        sigma_squared=0.01,
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

    for (unknown_i, unknown_j) in tqdm(
        zip(*unknown_map.nonzero()), total=np.count_nonzero(unknown_map),
        desc="Obtaining pixel samples for each unknown pixel",
        disable=not logging.root.isEnabledFor(logging.INFO)
    ):
        c = I[unknown_i, unknown_j]
        foreground_samples, foreground_sample_is, foreground_sample_js, \
        background_samples, background_sample_is, background_sample_js = get_samples(
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
            scheme_config={"name": "global_random"}  # TODO: change to deterministic when done (is abt 3x slower)
        )

        minimum_foreground_distance = foreground_samples - c  # foreground_samples_count x C float array
        minimum_foreground_distance = np.min(np.sum(minimum_foreground_distance * minimum_foreground_distance, axis=1))  # this is D_F^2
        penalty_foreground_weights = foreground_samples - c
        penalty_foreground_weights = np.sum(penalty_foreground_weights * penalty_foreground_weights, axis=1) / minimum_foreground_distance
        penalty_foreground_weights = np.exp(-penalty_foreground_weights)  # 1D length-foreground_samples_count float array

        minimum_background_distance = background_samples - c  # background_samples_count x C float array
        minimum_background_distance = np.min(np.sum(minimum_background_distance * minimum_background_distance, axis=1))  # this is D_B^2
        penalty_background_weights = background_samples - c
        penalty_background_weights = np.sum(penalty_background_weights * penalty_background_weights, axis=1) / minimum_background_distance
        penalty_background_weights = np.exp(-penalty_background_weights)  # 1D length-foreground_samples_count float array

        foreground_sample_i_background_sample_j_pairs = np.array(list(
            itertools.product(range(foreground_samples_count), range(background_samples_count))
        ))
        confidences = []
        for foreground_sample_i, background_sample_j in foreground_sample_i_background_sample_j_pairs:
            f = foreground_samples[foreground_sample_i]
            b = background_samples[background_sample_j]
            print(f">>> {penalty_distance_ratio_squared(c, f, b)} {penalty_foreground_weights[foreground_sample_i]} {penalty_background_weights[background_sample_j]}")
            confidence = penalty_distance_ratio_squared(c, f, b) * penalty_foreground_weights[foreground_sample_i] * penalty_background_weights[background_sample_j] / sigma_squared  # scalar
            confidence = np.exp(-confidence)  # scalar
            print(f"    | {confidence}")
            confidences.append(confidence)
        confidences = np.array(confidences)
        highest_confidence_argsort = np.argsort(confidences)[-highest_confidence_pairs_to_select:]
        estimated_alphas = []  # only the highest_confidence_pairs_to_select most confident ones
        for foreground_sample_i, background_sample_j in foreground_sample_i_background_sample_j_pairs[highest_confidence_argsort]:
            estimated_alphas.append(estimate_alpha(c, foreground_samples[foreground_sample_i], background_samples[background_sample_j]))
        print(estimated_alphas)
        print(confidences)
        print(confidences[highest_confidence_argsort])

        print(np.average(estimated_alphas, weights=confidences[highest_confidence_argsort]))
        raise

        print(foreground_samples)

        raise

    raise

    # skimage.io.imshow(foreground_map)
    # skimage.io.show()
    # skimage.io.imshow(foreground_boundary_map)
    # skimage.io.show()

    # skimage.io.imshow(background_map)
    # skimage.io.show()
    # skimage.io.imshow(background_boundary_map)
    # skimage.io.show()

    # print("entry")
    pass
