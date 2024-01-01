import os

import numpy as np
import skimage

import utils
from coarse_to_fine import solve_alpha_coarse_to_fine

if __name__ == "__main__":
    alpha_threshold = 0.05
    epsilon = 1e-7
    window_size = 1  # or 2
    assert (1 + 2 * window_size) % 2 != 0, f"M = (1 + 2 * window_size) should be even, got M = {1 + 2 * window_size}"

    levels_count = 4
    explicit_alpha_levels_count = 2
    # use active_levels_num<=levels_num.
    # If active_levels_num<levels_num then in the finer resolutions alpha is not computed explicitly.
    # Instead the linear coefficients of the coarser resolution are interpolated.

    image_name = "GT01.png"
    image_path = os.path.join("datasets", "input_training_lowres", image_name)
    # image_name = "peacock.bmp"
    # image_path = os.path.join("matlab", image_name)
    I = skimage.io.imread(image_path) / 255
    I = utils.ensure_3d_image(I)  # H x W x C array

    scribble_image_name = "GT01-scribble.png"
    scribble_image_path = os.path.join("datasets", "input_training_lowres", scribble_image_name)
    # scribble_image_name = "peacock_m.bmp"
    # scribble_image_path = os.path.join("matlab", scribble_image_name)
    I_scribble = skimage.io.imread(scribble_image_path) / 255 # this is mI in MATLAB code
    assert I_scribble.shape == I.shape, f"Shapes of scribble image ({I_scribble.shape}) and image ({I.shape}) don't match"
    I_scribble_gray2D = utils.matlab_compatible_rgb2gray(I_scribble) if I.shape[2] == 3 else I_scribble  # HxW array
    I_scribble = utils.ensure_3d_image(I_scribble)  # H x W x C array

    # constrained_X are always 2D arrays, not 3D
    constrained_map = (np.sum(np.absolute(I - I_scribble), axis=2) > 1e-3).astype(int)  # this is consts_map in MATLAB; HxW array
    constrained_vals = I_scribble_gray2D * constrained_map  # NB. * is the element-wise multiply operator; HxW array

    alpha = solve_alpha_coarse_to_fine(
        I,
        constrained_map,
        constrained_vals,
        levels_count,
        explicit_alpha_levels_count,
        alpha_threshold,
        epsilon,
        window_size
    )  # H x W array

    skimage.io.imshow(alpha)
    skimage.io.show()
