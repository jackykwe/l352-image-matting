import math
import os

# import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# import skimage
# import skimage.filters as filters
import skimage

# from scipy import ndimage, signal
# from scipy.interpolate import interp1d

# from skimage import color, util
# from skimage.util import img_as_uint
# from sksparse.cholmod import cholesky
# from tqdm.auto import tqdm
# from tqdm.contrib.concurrent import process_map  # internally uses concurrent.futures

def ensure_3d_image(I):
    I = I[..., np.newaxis] if len(I.shape) == 2 else I
    assert len(I.shape) == 3  # don't want images with alpha channels
    return I


def matlab_compatible_rgb2gray(I):
    """
    Turns a 3D image into a 2D image
    """
    return 0.2989 * I[:, :, 0] + 0.5870 * I[:, :, 1] + 0.1140 * I[:, :, 2]


def downsample_image(I, antialiasing_filter_size):
    if antialiasing_filter_size not in (1, 2):
        raise NotImplementedError(f"Got filter_size={antialiasing_filter_size}, expected 1 or 2")
    filter = np.array([1, 2, 1]) / 4 if antialiasing_filter_size == 1 else np.array([1, 4, 6, 4, 1]) / 16
    I = sp.ndimage.convolve1d(I, filter, axis=0, mode="constant")
    I = sp.ndimage.convolve1d(I, filter, axis=1, mode="constant")
    return I[antialiasing_filter_size:2:-antialiasing_filter_size, antialiasing_filter_size:2:-antialiasing_filter_size, :]


def upsample_image(I, new_H, new_W, reconstruction_filter_size=1):
    """
    This function's size manipulations are the inverse to downsample_image().
    """
    if reconstruction_filter_size == 0:
        reconstruction_filter = np.array([1])
    elif reconstruction_filter_size == 1:
        reconstruction_filter = np.array([1, 2, 1]) / 2  # TODO: Why /2 instead of /4?
    elif reconstruction_filter_size == 2:
        reconstruction_filter = np.array([1, 4, 6, 4, 1]) / 8  # TODO: Why /8 instead of /16?
    else:
        raise NotImplementedError

    # In downsample_image(), the final part of the function removed a strip of antialiasing_filter_size
    # from each of the four edges of the image. The following four quantities calculate the exact
    # width of the four strips.
    top_strip_height = math.floor((new_H - 2 * I.shape[0] + 1) / 2)  # this is id in MATLAB code. Mathematically, this is antialiasing_filter_size, always.
    bottom_strip_height = math.ceil((new_H - 2 * I.shape[0] + 1) / 2)  # this is iu in MATLAB code. Mathematically, this is antialiasing_filter_size if new_H (i.e. H in downsample_image()) is odd, and antialiasing_filter_size + 1 if new_H is even.
    left_strip_width = math.floor((new_W - 2 * I.shape[1] + 1) / 2)  # this is jd in MATLAB code. Mathematically, this is antialiasing_filter_size, always.
    right_strip_width = math.ceil((new_W - 2 * I.shape[1] + 1) / 2)  # this is ju in MATLAB code. Mathematically, this is antialiasing_filter_size if new_H (i.e. H in downsample_image()) is odd, and antialiasing_filter_size + 1 if new_H is even.

    # This is not the final size yet. NB. The reconstruction_filter_size here may not necessarily be the same as antialiasing_filter_size.
    result = np.zeros((new_H + 2 * reconstruction_filter_size, new_W + 2 * reconstruction_filter_size, I.shape[2]))

    # Insert into the correct places in the original pre-downsampled images, where downsampled values come from
    result[
        reconstruction_filter_size + top_strip_height:-reconstruction_filter_size - bottom_strip_height:2,
        reconstruction_filter_size + left_strip_width:-reconstruction_filter_size - right_strip_width:2,
    ] = I

    # Perform copy-padding (instead of zero-padding) for upsampling (interpolation) purposes
    result[reconstruction_filter_size + top_strip_height - 2::-2, :, :] = np.matlib.repmat(
        result[reconstruction_filter_size + top_strip_height, :, :],
        math.ceil((reconstruction_filter_size + top_strip_height - 1) / 2),
        1
    )  # repeat first row of downsampled pixels upwards for interpolation purposes
    result[-reconstruction_filter_size - bottom_strip_height + 1::2, :, :] = np.matlib.repmat(
        result[-reconstruction_filter_size - bottom_strip_height - 1, :, :],
        math.ceil((reconstruction_filter_size + bottom_strip_height - 1) / 2),
        1
    )  # repeat last row of downsampled pixels downwards for interpolation purposes
    result[:, reconstruction_filter_size + left_strip_width - 2::-2, :] = np.matlib.repmat(
        result[:, reconstruction_filter_size + left_strip_width, :],
        1,
        math.ceil((reconstruction_filter_size + left_strip_width - 1) / 2),
    )  # repeat first column of downsampled pixels leftwards for interpolation purposes
    result[:, -reconstruction_filter_size - right_strip_width + 1::-2, :] = np.matlib.repmat(
        result[:, -reconstruction_filter_size - right_strip_width - 1, :],
        1,
        math.ceil((reconstruction_filter_size + right_strip_width - 1) / 2),
    )  # repeat last column of downsampled pixels right for interpolation purposes

    # The actual upsampling via a reconstruction filter
    result = sp.ndimage.convolve1d(result, reconstruction_filter, axis=0, mode="constant")
    result = sp.ndimage.convolve1d(result, reconstruction_filter, axis=1, mode="constant")
    return result[reconstruction_filter_size:-reconstruction_filter_size, reconstruction_filter_size:-reconstruction_filter_size, :]


def get_linear_coefficients(alpha, I, epsilon, window_size):
    r"""
    If I is a HxWxC array, then alpha is a HWx1 vector (with row-major traversal through image I).

    Given a fixed alpha over image I, compute the a_k's and b_k's as defined below. The output is of
    shape `(I.shape[0], I.shape[1], 4)` for coloured images and `(I.shape[0], I.shape[1], 2)` for
    greyscale images.

    For greyscale images, we assume foreground image F and background image B are locally smooth, so
    $$ \alpha_i \approx a_k I_i + b_k $$
    for all pixels i in the neighbourhood of pixel k. Here a_k, I_i and b_k are all scalar.

    Likewise for colour images, we assume the colour line model, so
    $$ \alpha_i \approx a_k^T I_i + b_k $$
    for all pixels i in the neighbourhood of pixel k. Here a_k and I_i are length-3 vectors, b_k is
    scalar.

    This function returns, for each k, the concatenation [a_k, b_k]. For coloured images this is a
    length-4 vector and so this function returns an image of shape
    """
    neighbourhood_size = (1 + 2 * window_size) ** 2  # this is neb_size in MATLAB
    H, W, C = I.shape
    indices = np.arange(H * W).reshape((H, W))  # row-major
    result = np.zeros((H, W, C + 1))  # a_k is a vector of length C. b_k is the +1.
    for i in range(window_size, H - window_size):
        for j in range(window_size, W - window_size):
            window_indices = indices[i - window_size:i + window_size + 1, j - window_size:j + window_size + 1].flatten()  # row-major
            # NB. The following (window_I) is the matrix A_{(k)} in my proof
            window_I = I[i - window_size:i + window_size + 1, j - window_size:j + window_size + 1, :].reshape((neighbourhood_size, C))  # row-major
            # NB. The following (G) is the matrix G_{(k)} in my proof
            G = np.block([
                [window_I, np.ones((neighbourhood_size, 1))],
                [epsilon ** 0.5 * np.eye(C), np.zeros((C, 1))]
            ])
            # NB. The following (bar_alpha_k) is the vector \bar{\alpha}_{(k)} in my proof
            bar_alpha_k = np.block([
                [alpha[window_indices]],  # row-major
                [np.zeros((C, 1))]
            ])
            # NB. The following RHS expression (before reshaping) is the block vector $[ \hat{a}_k & \hat{b}_k ]^T$ in my proof
            result[i, j, :] = (np.linalg.inv(G.T @ G) @ G.T @ bar_alpha_k).reshape(1, 1, -1)
    result[:window_size, :, :] = np.matlib.repmat(result[window_size, :, :], window_size, 1)  # TODO why not calculate normally with "zero-padding" idea?
    result[-window_size:, :, :] = np.matlib.repmat(result[-window_size - 1, :, :], window_size, 1)  # TODO why not calculate normally with "zero-padding" idea?
    result[:, :window_size, :] = np.matlib.repmat(result[:, window_size, :], 1, window_size)  # TODO why not calculate normally with "zero-padding" idea?
    result[:, -window_size:, :] = np.matlib.repmat(result[:, -window_size - 1, :], 1, window_size)  # TODO why not calculate normally with "zero-padding" idea?
    return result


def upsample_alpha_using_image(downsampled_alpha, downsampled_I, I, epsilon, window_size):
    downsampled_linear_coefficients = get_linear_coefficients(downsampled_alpha, downsampled_I, epsilon, window_size)
    linear_coefficients = upsample_image(downsampled_linear_coefficients, I.shape[0], I.shape[1])  # this is bcoeff in MATLAB code

    # Using equation $\forall i. \alpha_i = a_i^T I_i + b_i$ where a_i, I_i are vectors and b_i is scalar



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
    erode_mask_size = 1  # TODO shouldn't this be equal to window_size?
    if levels_count >= 2:
        downsampled_I = downsample_image(I, antialiasing_filter_size=2)
        downsampled_constrained_map = np.round(downsample_image(constrained_map, antialiasing_filter_size=2))
        downsampled_constrained_vals = np.round(downsample_image(constrained_vals, antialiasing_filter_size=2))
        downsampled_alpha = solve_alpha_coarse_to_fine(
            downsampled_I,
            downsampled_constrained_map,
            downsampled_constrained_vals,
            levels_count - 1,
            min(levels_count - 1, explicit_alpha_levels_count),
            alpha_threshold,
            epsilon,
            window_size
        )
        alpha = upsample_alpha_using_image(downsampled_alpha, downsampled_I, I, epsilon, window_size)

    pass


if __name__ == "__main__":
    alpha_threshold = 0.02
    epsilon = 1e-7
    window_size = 3  # or 5. This is M
    assert window_size % 2 != 0, f"Window size M should be even, got {window_size}"

    levels_count = 4
    explicit_alpha_levels_count = 2
    # use active_levels_num<=levels_num.
    # If active_levels_num<levels_num then in the finer resolutions alpha is not computed explicitly.
    # Instead the linear coefficients of the coarser resolution are interpolated.



    # image_name = "GT01g.png"
    # image_path = os.path.join("datasets", "input_training_lowres", image_name)
    # I = skimage.io.imread(image_path)
    # I = ensure_3d_image(I)
    # print(I.shape)
    # skimage.io.imshow(sp.ndimage.convolve1d(I, np.array([1,2,1])/4, axis=0, mode="constant"))
    # skimage.io.show()
    # raise



    image_name = "GT01.png"
    image_path = os.path.join("datasets", "input_training_lowres", image_name)
    I = skimage.io.imread(image_path)
    I = ensure_3d_image(I)

    scribble_image_name = "GT01-scribble.png"
    scribble_image_path = os.path.join("datasets", "input_training_lowres", scribble_image_name)
    I_scribble = skimage.io.imread(scribble_image_path)  # this is mI in MATLAB code
    assert I_scribble.shape == I.shape, f"Shapes of scribble image ({I_scribble.shape}) and image ({I.shape}) don't match"
    I_scribble_gray2D = matlab_compatible_rgb2gray(I_scribble) if I.shape[2] == 3 else I_scribble
    I_scribble = ensure_3d_image(I_scribble)

    # constrained_X are always 2D arrays, not 3D
    constrained_map = (np.sum(np.absolute(I - I_scribble), axis=2) > 1e-3).astype(int)  # this is consts_map in MATLAB
    constrained_vals = I_scribble_gray2D * constrained_map  # NB. * is the element-wise multiply operator
    print(constrained_map.shape)
    print(constrained_vals.shape)

    alpha = solve_alpha_coarse_to_fine(
        I,
        constrained_map,
        constrained_vals,
        levels_count,
        explicit_alpha_levels_count,
        alpha_threshold,
        epsilon,
        window_size
    )


    skimage.io.imshow(sp.ndimage.convolve1d(I, np.array([1,2,1])/4, axis=0, mode="constant"))
    skimage.io.show()






