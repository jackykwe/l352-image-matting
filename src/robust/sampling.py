
import numpy as np

# def euclidean_distances(
#     foreground_boundary_is,
#     foreground_boundary_js,
#     background_boundary_is,
#     background_boundary_js,
#     unknown_i,
#     unknown_j
# ):
#     """
#     `foreground_boundary_is`: length-N 1D array of x-coordinates of foreground boundary pixels
#     `foreground_boundary_ys`: length-N 1D array of y-coordinates of foreground boundary pixels
#     `background_boundary_is`: length-N 1D array of x-coordinates of background boundary pixels
#     `background_boundary_ys`: length-N 1D array of y-coordinates of background boundary pixels
#     `unknown_i` and `unknown_j`: coordinates of unknown pixel for which to obtain foreground and background samples for

#     Returns:
#     ```
#     (
#         length-N 1D array of Euclidean distances of the foreground boundary pixels relative to unknown pixel,
#         length-N 1D array of Euclidean distances of the background boundary pixels relative to unknown pixel
#     )
#     """
#     foreground_boundary_distances = np.sqrt((foreground_boundary_is - unknown_i) ** 2 + (foreground_boundary_js - unknown_j) ** 2)  # 1D float array
#     background_boundary_distances = np.sqrt((background_boundary_is - unknown_i) ** 2 + (background_boundary_js - unknown_j) ** 2)  # 1D float array
#     return foreground_boundary_distances, background_boundary_distances


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


# def chebyshev_distances(
#     foreground_boundary_is,
#     foreground_boundary_js,
#     background_boundary_is,
#     background_boundary_js,
#     unknown_i,
#     unknown_j
# ):
#     """
#     `foreground_boundary_is`: length-N 1D array of x-coordinates of foreground boundary pixels
#     `foreground_boundary_ys`: length-N 1D array of y-coordinates of foreground boundary pixels
#     `background_boundary_is`: length-N 1D array of x-coordinates of background boundary pixels
#     `background_boundary_ys`: length-N 1D array of y-coordinates of background boundary pixels
#     `unknown_i` and `unknown_j`: coordinates of unknown pixel for which to obtain foreground and background samples for

#     Returns:
#     ```
#     (
#         length-N 1D array of Chebyshev distances of the foreground boundary pixels relative to unknown pixel,
#         length-N 1D array of Chebyshev distances of the background boundary pixels relative to unknown pixel
#     )
#     """
#     foreground_boundary_distances = np.max(
#         np.vstack((
#             np.absolute(foreground_boundary_is - unknown_i),
#             np.absolute(foreground_boundary_js - unknown_j)
#         )),
#         axis=0
#     )  # 1D float array
#     background_boundary_distances = np.max(
#         np.vstack((
#             np.absolute(background_boundary_is - unknown_i),
#             np.absolute(background_boundary_js - unknown_j)
#         )),
#         axis=0
#     )  # 1D float array
#     return foreground_boundary_distances, background_boundary_distances


def get_samples(
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
    - `{"name": "global_random_same"}`
    - `{"name": "global_random"}`
    - `{"name": "local_random", "nearest_candidates_count": int}`
    - `{"name": "deterministic"}`
    - `{"name": "deterministic_spread_global"}`
    - `{"name": "deterministic_spread_local", "nearest_candidates_count": int}`

    For an unknown pixel at coordinates `unknown_ij, the schemes behave differently:
    - `global_random_same`: sample `foreground_samples_count` pixels randomly from all foreground boundary pixels; likewise for background. all unknown pixels use the same samples.
    - `global_random`: same as `global_random_same` but resampling is performed for each unknown pixel.
    - `local_random`: sample `foreground_samples_count` pixels randomly from the nearest `nearest_candidates_count` foreground boundary pixels; likewise for background. resampling is performed for each unknown pixel.
    - `deterministic`: sample the nearest `foreground_samples_count` foreground boundary pixels; likewise for background. resampling is performed for each unknown pixel.
    - `deterministic_spread_global`: same as `deterministic`, but the samples are evenly spread across image, like the circles in Figure 5(b). resampling is performed for each unknown pixel.
    - `deterministic_spread_local`: same as `deterministic_spread_global`, but the samples are evenly spread across the nearest `nearest_candidates_count` fore/background pixels, like the circles in Figure 5(b). resampling is performed for each unknown pixel. This is the default scheme.

    Returns:
    ```
    (
        foreground_samples_count x C float array,  (foreground pixel values)
        background_samples_count x C float array,  (background pixel values)
        1D length-(foreground_samples_count) int array,  (foreground pixel x-coordinates)
        1D length-(foreground_samples_count) int array,  (foreground pixel y-coordinates)
        1D length-(background_samples_count) int array,  (background pixel x-coordinates)
        1D length-(background_samples_count) int array,  (background pixel y-coordinates)
    )
    ```

    Useful discussion on distance metrics at https://chris3606.github.io/GoRogue/articles/grid_components/measuring-distance.html
    """
    if scheme_config["name"] == "global_random_same":
        rng = np.random.default_rng(seed=42)
        foreground_choices = rng.choice(len(global_foreground_boundary_pixels), foreground_samples_count, replace=False)
        background_choices = rng.choice(len(global_background_boundary_pixels), background_samples_count, replace=False)
    elif scheme_config["name"] == "global_random":
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
    elif scheme_config["name"] == "deterministic_spread_global":
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
    elif scheme_config["name"] == "deterministic_spread_local":
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
        foreground_interval = nearest_candidates_count // foreground_samples_count
        background_interval = nearest_candidates_count // background_samples_count
        foreground_choices = np.argsort(foreground_boundary_distances)[:nearest_candidates_count:foreground_interval][:foreground_samples_count]
        background_choices = np.argsort(background_boundary_distances)[:nearest_candidates_count:background_interval][:background_samples_count]
    else:
        raise NotImplementedError

    foreground_samples = global_foreground_boundary_pixels[foreground_choices]
    background_samples = global_background_boundary_pixels[background_choices]
    assert len(foreground_samples) == foreground_samples_count, f"len(foreground_samples) ({len(foreground_samples)}) != foreground_samples_count {foreground_samples_count}"
    assert len(background_samples) == background_samples_count, f"len(background_samples) ({len(background_samples)}) != background_samples_count {background_samples_count}"
    return (
        foreground_samples,
        background_samples,
        None, # foreground_boundary_is[foreground_choices],  # returned for debugging purposes; set to None to avoid unnecessary slow comptuation (indexing)
        None, # foreground_boundary_js[foreground_choices],  # returned for debugging purposes; set to None to avoid unnecessary slow comptuation (indexing)
        None, # background_boundary_is[background_choices],  # returned for debugging purposes; set to None to avoid unnecessary slow comptuation (indexing)
        None, # background_boundary_js[background_choices]  # returned for debugging purposes; set to None to avoid unnecessary slow comptuation (indexing)
    )
