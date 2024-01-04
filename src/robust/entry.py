import logging

import numpy as np
import numpy.matlib as npmatlib
import scipy.linalg as splinalg
import scipy.ndimage as spndimage
import scipy.sparse as spsparse
import scipy.sparse.linalg as spsparselinalg
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


def get_laplacian(I, epsilon, window_size, index_displacement_map):
    """
    `I`: H x W x C array

    Returns: HW x HW sparse float array

    Code is very similar to the one used in closed form matting, small modifications are made.
    """
    neighbourhood_size = (1 + 2 * window_size) ** 2
    neighbourhood_size_squared = neighbourhood_size ** 2
    H, W, C = I.shape

    indices = np.arange(H * W).reshape((H, W))  # row-major
    # temp_length is the number of window centres (excluding a strip of width window_size around image) that we need to sum
    # i.e. those whose neighbourhoods contain at least one unconstrained pixel
    constructor_length = (H - 2 * window_size) * (W - 2 * window_size) * neighbourhood_size_squared  # this is tlen in MATLAB code; this is the length of the arguments to sp.sparse.csr_matrix(). Here in Python we also exploit the feature of csr_matrix() that accumulates values of duplicated indices

    # These three are arguments to the sp.sparse.csr_matrix() sparse matrix constructor
    constructor_row_indices = np.zeros(constructor_length)
    constructor_col_indices = np.zeros(constructor_length)
    constructor_vals = np.zeros(constructor_length)

    len = 0
    for i in tqdm(
        range(window_size, H - window_size),
        desc=f"Generating Laplacian",
        disable=not logging.root.isEnabledFor(logging.INFO)
    ):
        for j in range(window_size, W - window_size):
            # (i, j) represents a particular window centre k

            window_indices = indices[i - window_size:i + window_size + 1, j - window_size:j + window_size + 1].flatten()  # row-major
            # NB. The following (window_I) is the matrix A_{(k)} in my proof
            window_I = I[i - window_size:i + window_size + 1, j - window_size:j + window_size + 1, :].reshape((neighbourhood_size, C))  # row-major
            # NB. The following (window_mu) is the vector \mu_{(k)} in my proof
            window_mu = np.mean(window_I, axis=0).reshape(-1, 1)  # sum columns; C x 1 vector
            # NB. The following (window_var) is the matrix (\Sigma_{(k)} + \frac{\epsilon}{M^2} I_{CxC})^{-1} in my proof
            window_var = splinalg.inv((window_I.T @ window_I) / neighbourhood_size - window_mu @ window_mu.T + epsilon / neighbourhood_size * np.eye(C))
            window_I = window_I - npmatlib.repmat(window_mu.T, neighbourhood_size, 1)
            # NB. The following (temp_vals) is the 1/M^2 * (1 + (...)^T (...)^{-1} (...)) scalar term for (k)
            # to be used in the computation of Laplacian matrix elements for all M^2 x M^2 possible
            # (i, j) pairs represented by window_indices
            temp_vals = (1 + window_I @ window_var @ window_I.T) / neighbourhood_size  # neighbourhood_size x neighbourhood_size matrix

            window_indices_displaced = index_displacement_map[window_indices]
            constructor_vals[len:len + neighbourhood_size_squared] = temp_vals.flatten()  # row-major
            # corresponding row_indices[len:len + neighbourhood_size_squared] should look like [0, 0, ..., 0                     , 1, 1, ..., 1                     , ..., neighbourhood_size_squared - 1]
            # corresponding col_indices[len:len + neighbourhood_size_squared] should look like [0, 1, ..., neighbourhood_size - 1, 0, 1, ..., neighbourhood_size - 1, ..., neighbourhood_size_squared - 1]
            constructor_row_indices[len:len + neighbourhood_size_squared] = npmatlib.repmat(window_indices_displaced, neighbourhood_size, 1).flatten(order="F")
            constructor_col_indices[len:len + neighbourhood_size_squared] = npmatlib.repmat(window_indices_displaced, neighbourhood_size, 1).flatten()
            len += neighbourhood_size_squared
    result = spsparse.csr_array((constructor_vals, (constructor_row_indices, constructor_col_indices)), shape=(H * W, H * W)) # this is A in MATLAB code
    return result

def solve_alpha(
        I,
        foreground_map,
        background_map,
        unknown_map,
        foreground_samples_count=1,
        background_samples_count=1,
        sigma_squared=0.01,
        highest_confidence_pairs_to_select=3,
        epsilon=1e-5,
        window_size=1
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

    H, W, C = I.shape
    unknown_count = np.count_nonzero(unknown_map)
    known_count = np.count_nonzero(~unknown_map)

    estimated_alphas = np.zeros((unknown_count, highest_confidence_pairs_to_select), dtype=float)  # row-major traversal of unknown pixels;  unknown_count x highest_confidence_pairs_to_select float array
    estimated_confidences = np.zeros((unknown_count, highest_confidence_pairs_to_select), dtype=float)  # row-major traversal of unknown pixels;  unknown_count x highest_confidence_pairs_to_select float array

    for i, (unknown_i, unknown_j) in enumerate(tqdm(
        zip(*unknown_map.nonzero()), total=np.count_nonzero(unknown_map),
        desc="Obtaining pixel samples and confidences for each unknown pixel",
        disable=not logging.root.isEnabledFor(logging.INFO)
    )):  # row-major traversal of unknown pixels
        cT = I[unknown_i, unknown_j]
        FiT, BjT = get_samples(
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

        confidences = -np.squeeze(cT_minus_BjT_3D @ Aij_3D @ c_minus_Bj_3D) * penalty_foreground_background / sigma_squared # 1D length-(foreground_samples_count * background_samples_count) float array

        highest_confidence_argsort = np.argsort(confidences)[-highest_confidence_pairs_to_select:]
        # means and exponentiations are taken later at top level for efficiency
        estimated_alphas[i, :] = (alpha_premultiplier_3D[highest_confidence_argsort, :, :] @ c_minus_Bj_3D[highest_confidence_argsort, :, :]).squeeze()  # 1D length-(highest_confidence_pairs_to_select) float array
        estimated_confidences[i, :] = confidences[highest_confidence_argsort]  # 1D length-(highest_confidence_pairs_to_select) float array

    estimated_alphas = np.mean(estimated_alphas, axis=1)  # row-major traversal of unknown pixels;  1D length-(unknown_count) float array
    estimated_confidences = np.mean(np.exp(estimated_confidences), axis=1)  # row-major traversal of unknown pixels;  1D length-(unknown_count) float array

    indices = np.arange(H * W).reshape((H, W))  # row-major
    unknown_indices_rowmaj = indices[unknown_map]  # row-major traversal of unknown pixels;  1D length-(unknown_count) int array
    known_indices_rowmaj = indices[~unknown_map]  # row-major traversal of known pixels;  1D length-(known_count) int array

    index_displacement_map = np.zeros(H * W, dtype=int)
    index_displacement_map[known_indices_rowmaj] = np.arange(known_count)
    index_displacement_map[unknown_indices_rowmaj] = known_count + np.arange(unknown_count)

    WiF = estimated_confidences * estimated_alphas + (1 - estimated_confidences) * (estimated_alphas > 0.5).astype(int)  # row-major traversal of unknown pixels;  1D length-(unknown_count) float array
    WiB = estimated_confidences * (1 - estimated_alphas) + (1 - estimated_confidences) * (estimated_alphas < 0.5).astype(int)  # row-major traversal of unknown pixels;  1D length-(unknown_count) float array
    L = get_laplacian(I, epsilon, window_size, index_displacement_map).tolil()  # H x W sparse-LIL float matrix. LIL format required for fancy indexing and still retaining sparse format

    RT_fragment = L[known_count:, :known_count]  # a little slow
    # print(RT_fragment.shape)

    # IMPOSSIBLE TO EXECUTE DUE TO OOM CONSTRUCTING THE INDEXING INDICES
    # RT_fragment = L[np.repeat(unknown_indices_rowmaj, known_count), np.tile(known_indices_rowmaj, unknown_count)].reshape(unknown_count, known_count)

    # TOO SLOW
    # sparse_arrays_to_concatenate = []
    # for unknown_index_rowmaj in tqdm(unknown_indices_rowmaj, desc="holy"):
    #     sparse_arrays_to_concatenate.append(L[[unknown_index_rowmaj] * known_count, known_indices_rowmaj])
    # RT_fragment = spsparse.hstack(sparse_arrays_to_concatenate)

    RT = spsparse.hstack((WiF.reshape(-1, 1), WiB.reshape(-1, 1), RT_fragment), format="csr")
    # print(RT.shape)
    # print(type(RT))

    Lu = L[known_count:, known_count:].reshape(unknown_count, unknown_count).tocsr()
    # print(Lu.shape)
    # print(type(Lu))

    result = foreground_map.astype(float)  # H x W float array
    Ak = np.concatenate(([1, 0], result[~unknown_map]))
    # print(Ak.shape)
    # print(type(Ak))

    negative_RT_Ak = -RT @ Ak
    # print(negative_RT_Ak.shape)
    # print(type(negative_RT_Ak))

    solved_alphas, result_code = spsparselinalg.cg(Lu, negative_RT_Ak)
    assert result_code == 0, f"Conjugate Gradient did not successfully exit, got result code {result_code} (expected 0)"

    result[unknown_map] = solved_alphas  # there tends to be some alphas that lie outside of [0, 1]...
    return result
