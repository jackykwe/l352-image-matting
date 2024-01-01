import numpy as np
import numpy.matlib as npmatlib
import scipy.ndimage as spndimage
import skimage

import explicit
import sampling
import utils


def get_linear_coefficients(alpha, I, epsilon, window_size):
    r"""
    `alpha`: H x W array
    `I`: H x W x C array

    Returns: H x W x (C + 1) array

    Given a fixed alpha over image I, compute the a_k's and b_k's as defined below.

    For greyscale images, we assume foreground image F and background image B are locally smooth, so
    $$ \alpha_i \approx a_k I_i + b_k $$
    for all pixels i in the neighbourhood of pixel k. Here a_k, I_i and b_k are all scalar.

    Likewise for colour images, we assume the colour line model, so
    $$ \alpha_i \approx a_k^T I_i + b_k $$
    for all pixels i in the neighbourhood of pixel k. Here a_k and I_i are length-3 vectors, b_k is
    scalar.

    This function returns, for each k, the concatenation [a_k, b_k].
    """
    neighbourhood_size = (1 + 2 * window_size) ** 2  # this is neb_size in MATLAB
    H, W, C = I.shape
    result = np.zeros((H, W, C + 1))  # a_k is a vector of length C. b_k is the +1.
    for i in range(window_size, H - window_size):
        for j in range(window_size, W - window_size):
            # NB. The following (window_I) is the matrix A_{(k)} in my proof
            window_I = I[i - window_size:i + window_size + 1, j - window_size:j + window_size + 1, :].reshape((neighbourhood_size, C))  # row-major
            # NB. The following (G) is the matrix G_{(k)} in my proof
            G = np.block([
                [window_I, np.ones((neighbourhood_size, 1))],
                [epsilon ** 0.5 * np.eye(C), np.zeros((C, 1))]
            ])
            # NB. The following (bar_alpha_k) is the vector \bar{\alpha}_{(k)} in my proof
            bar_alpha_k = np.block([
                [alpha[i - window_size:i + window_size + 1, j - window_size:j + window_size + 1].reshape(-1, 1)],  # row-major
                [np.zeros((C, 1))]
            ])
            # NB. The following RHS expression (before reshaping) is the block vector $[ \hat{a}_k & \hat{b}_k ]^T$ in my proof
            result[i, j, :] = (np.linalg.inv(G.T @ G) @ G.T @ bar_alpha_k).reshape(1, 1, -1)
    # The following broadcasting nuance is courtesy of https://stackoverflow.com/questions/3551242/numpy-index-slice-without-losing-dimension-information#comment90059776_18183182
    result[:window_size, :, :] = result[[window_size], :, :]  # TODO why not calculate normally with "zero-padding" idea?
    result[-window_size:, :, :] = result[[-window_size - 1], :, :]  # TODO why not calculate normally with "zero-padding" idea?
    result[:, :window_size, :] = result[:, [window_size], :]  # TODO why not calculate normally with "zero-padding" idea?
    result[:, -window_size:, :] = result[:, [-window_size - 1], :]  # TODO why not calculate normally with "zero-padding" idea?
    return result


def upsample_alpha_using_image(downsampled_alpha, downsampled_I, I, epsilon, window_size):
    """
    `downsampled_alpha`: ceil((H - 2 * antialiasing_filter_size) / 2) x ceil((W - 2 * antialiasing_filter_size) / 2) array
    `downsampled_I`: ceil((H - 2 * antialiasing_filter_size) / 2) x ceil((W - 2 * antialiasing_filter_size) / 2) x C array
    `I`: H x W x C array

    Returns: H x W array
    """
    print("upsample_alpha_using_image invoked")
    downsampled_linear_coefficients = get_linear_coefficients(downsampled_alpha, downsampled_I, epsilon, window_size)  # ceil((H - 2 * antialiasing_filter_size) / 2) x ceil((W - 2 * antialiasing_filter_size) / 2) x (C + 1) array
    linear_coefficients = sampling.upsample_image(downsampled_linear_coefficients, I.shape[0], I.shape[1])  # this is bcoeff in MATLAB code; H x W x (C + 1) array\

    # Using equation $\forall i. \alpha_i = a_i^T I_i + b_i$ where a_i, I_i are vectors and b_i is scalar
    return np.sum(linear_coefficients[:, :, :-1] * I, axis=2) + linear_coefficients[:, :, -1]


def solve_alpha_coarse_to_fine(
        I,
        constrained_map,
        constrained_vals,
        levels_count,
        explicit_alpha_levels_count,
        alpha_threshold,
        epsilon,
        window_size
    ):
    """
    `I`: H x W x C array
    `constrained_map`: H x W array
    `constrained_vals`: H x W array

    Returns: H x W array
    """
    print(f"[fineness {levels_count}] solve_alpha_coarse_to_fine invoked")

    erode_mask_size = 1  # TODO shouldn't this be equal to window_size?
    if levels_count >= 2:
        downsampled_I = sampling.downsample_image(I, antialiasing_filter_size=2)  # ceil((H - 2 * antialiasing_filter_size) / 2) x ceil((W - 2 * antialiasing_filter_size) / 2) x C array
        downsampled_constrained_map = np.round(
            sampling.downsample_image(utils.ensure_3d_image(constrained_map), antialiasing_filter_size=2).squeeze()
        )  # ceil((H - 2 * antialiasing_filter_size) / 2) x ceil((W - 2 * antialiasing_filter_size) / 2) array
        downsampled_constrained_vals = np.round(
            sampling.downsample_image(utils.ensure_3d_image(constrained_vals), antialiasing_filter_size=2).squeeze()
        )  # ceil((H - 2 * antialiasing_filter_size) / 2) x ceil((W - 2 * antialiasing_filter_size) / 2) array

        print(f"[fineness {levels_count}] displaying downsampled_constrained_map")
        skimage.io.imshow(downsampled_constrained_map)  # TODO: remove
        skimage.io.show()  # TODO: remove

        downsampled_alpha = solve_alpha_coarse_to_fine(
            downsampled_I,  # ceil((H - 2 * antialiasing_filter_size) / 2) x ceil((W - 2 * antialiasing_filter_size) / 2) x C array
            downsampled_constrained_map,  # ceil((H - 2 * antialiasing_filter_size) / 2) x ceil((W - 2 * antialiasing_filter_size) / 2) array
            downsampled_constrained_vals,  # ceil((H - 2 * antialiasing_filter_size) / 2) x ceil((W - 2 * antialiasing_filter_size) / 2) array
            levels_count - 1,
            min(levels_count - 1, explicit_alpha_levels_count),
            alpha_threshold,
            epsilon,
            window_size
        )  # ceil((H - 2 * antialiasing_filter_size) / 2) x ceil((W - 2 * antialiasing_filter_size) / 2) array

        print(f"[fineness {levels_count}] displaying downsampled_alpha")
        skimage.io.imshow(downsampled_alpha)  # TODO: remove
        skimage.io.show()  # TODO: remove

        alpha = upsample_alpha_using_image(
            downsampled_alpha,  # ceil((H - 2 * antialiasing_filter_size) / 2) x ceil((W - 2 * antialiasing_filter_size) / 2) array
            downsampled_I,  # ceil((H - 2 * antialiasing_filter_size) / 2) x ceil((W - 2 * antialiasing_filter_size) / 2) x C array
            I,  # H x W x C array
            epsilon,
            window_size
        )  # H x W array

        print(f"[fineness {levels_count}] displaying (upsampled) alpha")
        skimage.io.imshow(alpha)  # TODO: remove
        skimage.io.show()  # TODO: remove

        # The following are only used if levels_count <= explicit_alpha_levels_count later
        temp_alpha = alpha * (1 - constrained_map) + constrained_vals  # enforce scribbled alphas
        # alpha values within alpha_threshold of 0 or 1 are clamped to 0 or 1 respectively.
        # Then, those pixels whose neighbourhoods contain no non-zero or non-one values are considered
        # constrained, and we will not sum over these windows when computing the Laplacian in
        # solve_alpha_explicit() later.
        constrained_map = np.minimum(
            1,
            constrained_map \
                + spndimage.binary_erosion(alpha >= 1 - alpha_threshold, np.ones((1 + 2 * erode_mask_size, 1 + 2 * erode_mask_size))) \
                + spndimage.binary_erosion(alpha <= alpha_threshold, np.ones((1 + 2 * erode_mask_size, 1 + 2 * erode_mask_size)))
        )
        constrained_vals = constrained_map * temp_alpha

    if levels_count <= explicit_alpha_levels_count:
        print(f"[fineness {levels_count}] EXPLICIT RUNNING")
        alpha = explicit.solve_alpha_explicit(
            I,  # H x W x C array
            constrained_map,  # H x W array
            constrained_vals,  # H x W array
            epsilon,
            window_size
        )
    print(f"    [fineness {levels_count}] solve_alpha_coarse_to_fine returning... (displaying)")
    skimage.io.imshow(alpha)
    skimage.io.show()
    print(f"    [fineness {levels_count}] solve_alpha_coarse_to_fine returned")
    return alpha
