import logging

import numpy as np
import numpy.matlib as npmatlib
import scipy.linalg as splinalg
import scipy.ndimage as spndimage
import scipy.sparse as spsparse
import scipy.sparse.linalg as spsparselinalg
import skimage
from tqdm.auto import tqdm

from . import sampling
from .utils import DIVISION_EPSILON

# from . import slowmetrics  # for debugging only


def get_laplacian(I, constrained_map, epsilon, window_size, index_displacement_map):
    """
    `I`: H x W x C array
    `constrained_map`: H x W bool array

    Returns: HW x HW sparse float array

    Code is very similar to the one used in closed form matting, small modifications are made.
    """
    neighbourhood_size = (1 + 2 * window_size) ** 2
    neighbourhood_size_squared = neighbourhood_size ** 2
    H, W, C = I.shape
    # constrain a pixel iff all pixels in its neighbourhood are also constrained. For these centres, no need to enforce linear colour model,
    # so we omit summing them. Remember the alphaT L alpha term is there to enforce the smoothness/linear colour models.
    constrained_map = spndimage.binary_erosion(constrained_map, np.ones((1 + 2 * window_size, 1 + 2 * window_size))).astype(int)

    indices = np.arange(H * W).reshape((H, W))  # row-major
    # temp_length is the number of window centres (excluding a strip of width window_size around image) that we need to sum
    # i.e. those whose neighbourhoods contain at least one unconstrained pixel
    constructor_length = np.sum(1 - constrained_map[window_size:-window_size, window_size:-window_size]) * neighbourhood_size_squared  # this is tlen in MATLAB code; this is the length of the arguments to sp.sparse.csr_matrix(). Here in Python we also exploit the feature of csr_matrix() that accumulates values of duplicated indices

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
            if constrained_map[i, j]:
                continue

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
    W_mat = spsparse.csr_array((constructor_vals, (constructor_row_indices, constructor_col_indices)), shape=(H * W, H * W)) # this is A in MATLAB code
    W_mat_row_sum = W_mat.sum(axis=1)  # this is sumA in MATLAB code
    D = spsparse.diags(W_mat_row_sum, 0, shape=(H * W, H * W), format="csr")
    # W_mat.setdiag(0)  # Zero diagonals? NO: do NOT set to zero; that's mathematically wrong. The MATLAB code of closed form is correct, and is also used here in Robust. The delta_{ij} terms exist.
    result = D - W_mat
    # Technically this function can be further optimised by choosing to return RT_fragment and Lu directly, so we completely ignore constructing the Lk and R parts. That is done in the CPP code at https://github.com/wangchuan/RobustMatting/blob/f0d6144a800128a489e66cd2b5c5fb669c7a133c/src/robust_matting/robust_matting.cpp#L113. This is an avenue for further optimisation.
    return result



def solve_alpha(
        I,
        foreground_map,
        background_map,
        unknown_map,
        foreground_samples_count=20,
        background_samples_count=20,
        sampling_method="deterministic_spread",
        nearest_candidates_count=200,
        sigma_squared=0.01,
        highest_confidence_pairs_to_select=3,
        epsilon=1e-5,
        gamma=0.1,
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
    estimated_confidences_exparg = np.zeros((unknown_count, highest_confidence_pairs_to_select), dtype=float)  # row-major traversal of unknown pixels;  unknown_count x highest_confidence_pairs_to_select float array

    if sampling_method in ("local_random", "deterministic_spread_local"):
        scheme_config = {"name": sampling_method, "nearest_candidates_count": nearest_candidates_count}
    else:
        scheme_config = {"name": sampling_method}

    for i, (unknown_i, unknown_j) in enumerate(tqdm(
        zip(*unknown_map.nonzero()), total=np.count_nonzero(unknown_map),
        desc="Obtaining pixel samples and confidences for each unknown pixel",
        disable=not logging.root.isEnabledFor(logging.INFO)
    )):  # row-major traversal of unknown pixels
        cT = I[unknown_i, unknown_j]
        FiT, BjT, debug_Fx, debug_Fy, debug_Bx, debug_By = sampling.get_samples(
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
            scheme_config
        )

        # DEBUG: display which pixel samples are chosen
        # debug_chosen_pixels_plot = foreground_map.astype(float)
        # debug_chosen_pixels_plot[unknown_map] = 0.5
        # debug_chosen_pixels_plot[debug_Fx, debug_Fy] = 4
        # debug_chosen_pixels_plot[debug_Bx, debug_By] = -3
        # debug_chosen_pixels_plot[unknown_i, unknown_j] = 5
        # skimage.io.imshow(debug_chosen_pixels_plot)
        # skimage.io.show()

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
        alpha_premultiplier_3D = FiT_minus_BjT_3D / Fi_minus_Bj_squared_3D  # this subexpression is named, as it's useful again later when estimating alphas (see estimated_alphas variable);  (foreground_samples_count * background_samples_count) x 1 x C float array
        Aij_3D = \
            (
                np.tile(np.eye(3).reshape(1, 3, 3), (foreground_samples_count * background_samples_count, 1, 1)) \
                - (Fi_minus_Bj_3D @ alpha_premultiplier_3D)
            ) / Fi_minus_Bj_squared_3D  # (foreground_samples_count * background_samples_count) x C x C float array
        c_minus_Bj_3D = np.tile(cT_minus_BjT, (foreground_samples_count, 1))[:, :, np.newaxis] # (foreground_samples_count * background_samples_count) x C x 1 float array
        cT_minus_BjT_3D = np.transpose(c_minus_Bj_3D, (0, 2, 1))  # (foreground_samples_count * background_samples_count) x 1 x C float array

        confidences_exparg = -np.squeeze(cT_minus_BjT_3D @ Aij_3D @ c_minus_Bj_3D) * penalty_foreground_background / sigma_squared # 1D length-(foreground_samples_count * background_samples_count) float array

        # DEBUG: correctness of above computation of confidences (before zeroing those of nonsensical alphas) (WARNING, this code slows down computation by a TON)
        # debug_slow_confidences_exparg = []
        # for i in range(foreground_samples_count):
        #     for j in range(background_samples_count):
        #         debug_slow_confidences_exparg.append(
        #             slowmetrics.slow_confidence_exparg(cT, FiT[i], BjT[j], FiT, BjT, sigma_squared)
        #         )
        # debug_slow_confidences_exparg = np.array(debug_slow_confidences_exparg)
        # assert np.allclose(confidences_exparg, debug_slow_confidences_exparg)

        # Courtesy of https://github.com/wangchuan/RobustMatting/blob/f0d6144a800128a489e66cd2b5c5fb669c7a133c/src/robust_matting/robust_matting.cpp#L266
        alphas = np.squeeze(alpha_premultiplier_3D @ c_minus_Bj_3D) # 1D length-(foreground_samples_count * background_samples_count) float array; need to compute alphas now to zero out confidence values for nonsensical alphas outside of tolerance range (-0.05, 1.05)
        erroneous_alpha_map = (alphas < 0.05) | (alphas > 1.05)
        confidences_exparg[erroneous_alpha_map] = -np.inf

        # DEBUG: Visualise values
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.plot(np.arange(len(alphas)), alphas)
        # ax.set_title("alphas")
        # plt.show()
        #
        # fig, ax = plt.subplots()
        # ax.plot(np.arange(len(confidences_exparg)), confidences_exparg)
        # ax.set_title("confidences_exparg")
        # plt.show()
        #
        # debug_confidence_exparg_descending = confidences_exparg[np.argsort(confidences_exparg)[::-1]]
        # fig, ax = plt.subplots()
        # ax.plot(np.arange(len(debug_confidence_exparg_descending)), debug_confidence_exparg_descending)
        # ax.set_title("debug_confidence_exparg_descending")
        # plt.show()
        #
        # debug_confidence_descending = np.exp(debug_confidence_exparg_descending)
        # fig, ax = plt.subplots()
        # ax.plot(np.arange(len(debug_confidence_descending)), debug_confidence_descending)
        # ax.set_ylim((0, 1))
        # ax.set_title("debug_confidence_descending")
        # plt.show()

        highest_confidence_argsort = np.argsort(confidences_exparg)[-highest_confidence_pairs_to_select:]
        # means and exponentiations are taken later at top level for efficiency. Also, exponentiation causes most values to become 1, which makes the procedure of sorting then taking the highest confidence pairs less meaningful: many pairs will have the maximal confidence of 1.
        estimated_alphas[i, :] = (alpha_premultiplier_3D[highest_confidence_argsort, :, :] @ c_minus_Bj_3D[highest_confidence_argsort, :, :]).squeeze()  # 1D length-(highest_confidence_pairs_to_select) float array
        estimated_confidences_exparg[i, :] = confidences_exparg[highest_confidence_argsort]  # 1D length-(highest_confidence_pairs_to_select) float array

    estimated_alphas = np.mean(estimated_alphas, axis=1)  # row-major traversal of unknown pixels;  1D length-(unknown_count) float array
    estimated_confidences = np.mean(np.exp(estimated_confidences_exparg), axis=1)  # row-major traversal of unknown pixels;  1D length-(unknown_count) float array

    # DEBUG: Show confidences
    # debug_confidence_map = np.ones((H, W)) * -0.5
    # debug_confidence_map[unknown_map] = estimated_confidences
    # print("Showing debug_confidence_map")
    # skimage.io.imshow(debug_confidence_map)
    # skimage.io.show()

    estimated_result = foreground_map.astype(float)  # H x W float array
    estimated_result[unknown_map] = estimated_alphas
    logging.debug(f"np.min(estimated_result) = {np.min(estimated_result)}")
    logging.debug(f"np.min(estimated_result) = {np.max(estimated_result)}")
    estimated_result = np.clip(estimated_result, -0.05, 1.05)  # TODO: Evaluate effectiveness of treating nonsensical alphas at this stage
    logging.debug(f"Clipping estimated_result to [-0.05, 1.05] range")
    logging.info("Showing estimated result (estimated alphas)")
    skimage.io.imshow(np.clip(estimated_result, 0, 1))
    skimage.io.show()

    indices = np.arange(H * W).reshape((H, W))  # row-major
    unknown_indices_rowmaj = indices[unknown_map]  # row-major traversal of unknown pixels;  1D length-(unknown_count) int array
    known_indices_rowmaj = indices[~unknown_map]  # row-major traversal of known pixels;  1D length-(known_count) int array

    index_displacement_map = np.zeros(H * W, dtype=int)
    index_displacement_map[known_indices_rowmaj] = np.arange(known_count)
    index_displacement_map[unknown_indices_rowmaj] = known_count + np.arange(unknown_count)

    WiF = -gamma * (estimated_confidences * estimated_alphas + (1 - estimated_confidences) * (estimated_alphas > 0.5).astype(int))  # row-major traversal of unknown pixels;  1D length-(unknown_count) float array
    WiB = -gamma * (estimated_confidences * (1 - estimated_alphas) + (1 - estimated_confidences) * (estimated_alphas <= 0.5).astype(int))  # row-major traversal of unknown pixels;  1D length-(unknown_count) float array
    L = get_laplacian(I, ~unknown_map, epsilon, window_size, index_displacement_map).tolil()  # H x W sparse-LIL float matrix. LIL format required for fancy indexing and still retaining sparse format

    RT_fragment = L[known_count:, :known_count]  # a little slow
    RT = spsparse.hstack((WiF.reshape(-1, 1), WiB.reshape(-1, 1), RT_fragment), format="csr")
    Lu = L[known_count:, known_count:].reshape(unknown_count, unknown_count).tocsr()
    Lu.setdiag(Lu.diagonal() + gamma)
    result = foreground_map.astype(float)  # H x W float array
    Ak = np.concatenate(([1, 0], result[~unknown_map]))
    negative_RT_Ak = -RT @ Ak

    solved_alphas, result_code = spsparselinalg.cg(Lu, negative_RT_Ak)
    assert result_code == 0, f"Conjugate Gradient did not successfully exit, got result code {result_code} (expected 0)"

    result[unknown_map] = solved_alphas  # there tends to be some alphas that lie outside of [0, 1]...
    return result
