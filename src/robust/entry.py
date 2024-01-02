import logging
import time

import numpy as np
import scipy as sp
import scipy.ndimage as spndimage
import skimage
from tqdm.auto import tqdm


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

    Returns: [foreground_samples_count x C float array, background_samples_count x C float array]

    Useful discussion on distance metrics at https://chris3606.github.io/GoRogue/articles/grid_components/measuring-distance.html
    """
    if scheme_config["name"] == "global_random":
        foreground_pixel_samples = global_foreground_boundary_pixels[
            np.random.choice(len(global_foreground_boundary_pixels), foreground_samples_count, replace=False)
        ]
        background_pixel_samples = global_background_boundary_pixels[
            np.random.choice(len(global_background_boundary_pixels), background_samples_count, replace=False)
        ]
    elif scheme_config["name"] == "local_random":
        nearest_candidates_count = scheme_config["nearest_candidates_count"]

                # # Euclidean distances (slow)
        # foreground_boundary_distances = np.sqrt((foreground_boundary_is - unknown_i) ** 2 + (foreground_boundary_js - unknown_j) ** 2)
        # background_boundary_distances = np.sqrt((background_boundary_is - unknown_i) ** 2 + (background_boundary_js - unknown_j) ** 2)

        # Manhattan distances (same ranking as Euclidean but cheaper)
        foreground_boundary_distances = np.absolute(foreground_boundary_is - unknown_i) + np.absolute(foreground_boundary_js - unknown_j)
        background_boundary_distances = np.absolute(background_boundary_is - unknown_i) + np.absolute(background_boundary_js - unknown_j)

        # # Chebyshev distances
        # foreground_boundary_distances = np.max(
        #     np.vstack((
        #         np.absolute(foreground_boundary_is - unknown_i),
        #         np.absolute(foreground_boundary_js - unknown_j)
        #     )),
        #     axis=0
        # )
        # background_boundary_distances = np.max(
        #     np.vstack((
        #         np.absolute(background_boundary_is - unknown_i),
        #         np.absolute(background_boundary_js - unknown_j)
        #     )),
        #     axis=0
        # )

        foreground_pixel_samples = global_foreground_boundary_pixels[
            np.random.choice(
                np.argsort(foreground_boundary_distances)[:nearest_candidates_count],
                foreground_samples_count,
                replace=False
            )
        ]
        background_pixel_samples = global_background_boundary_pixels[
            np.random.choice(
                np.argsort(background_boundary_distances)[:nearest_candidates_count],
                background_samples_count,
                replace=False
            )
        ]
    elif scheme_config["name"] == "deterministic":
        # # Euclidean distances (slow)
        # foreground_boundary_distances = np.sqrt((foreground_boundary_is - unknown_i) ** 2 + (foreground_boundary_js - unknown_j) ** 2)
        # background_boundary_distances = np.sqrt((background_boundary_is - unknown_i) ** 2 + (background_boundary_js - unknown_j) ** 2)

        # Manhattan distances (same ranking as Euclidean but cheaper)
        foreground_boundary_distances = np.absolute(foreground_boundary_is - unknown_i) + np.absolute(foreground_boundary_js - unknown_j)
        background_boundary_distances = np.absolute(background_boundary_is - unknown_i) + np.absolute(background_boundary_js - unknown_j)

        # # Chebyshev distances
        # foreground_boundary_distances = np.max(
        #     np.vstack((
        #         np.absolute(foreground_boundary_is - unknown_i),
        #         np.absolute(foreground_boundary_js - unknown_j)
        #     )),
        #     axis=0
        # )
        # background_boundary_distances = np.max(
        #     np.vstack((
        #         np.absolute(background_boundary_is - unknown_i),
        #         np.absolute(background_boundary_js - unknown_j)
        #     )),
        #     axis=0
        # )

        foreground_pixel_samples = global_foreground_boundary_pixels[
            np.argsort(foreground_boundary_distances)[:foreground_samples_count]
        ]
        background_pixel_samples = global_background_boundary_pixels[
            np.argsort(background_boundary_distances)[:background_samples_count]
        ]
    else:
        raise NotImplementedError

    return foreground_pixel_samples, background_pixel_samples

def solve_alpha(
        I,
        foreground_map,
        background_map,
        unknown_map
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
        foreground_samples, background_samples = get_samples(
            I,
            foreground_boundary_is,
            foreground_boundary_js,
            foreground_boundary_pixels,
            background_boundary_is,
            background_boundary_js,
            background_boundary_pixels,
            unknown_i,
            unknown_j,
            foreground_samples_count=20,
            background_samples_count=20,
            scheme_config={"name": "global_random"}  # TODO: change to deterministic when done (is abt 3x slower)
        )

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
