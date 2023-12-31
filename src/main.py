import math
import os

# import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as npmatlib

# import scipy as sp
import scipy.ndimage as spndimage

# import skimage
# import skimage.filters as filters
import skimage

from . import sampling, utils
from .solvers import solve_alpha_coarse_to_fine

# from scipy import ndimage, signal
# from scipy.interpolate import interp1d

# from skimage import color, util
# from skimage.util import img_as_uint
# from sksparse.cholmod import cholesky
# from tqdm.auto import tqdm
# from tqdm.contrib.concurrent import process_map  # internally uses concurrent.futures



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

    image_name = "GT01.png"
    image_path = os.path.join("datasets", "input_training_lowres", image_name)
    I = skimage.io.imread(image_path)
    I = utils.ensure_3d_image(I)  # H x W x C array

    scribble_image_name = "GT01-scribble.png"
    scribble_image_path = os.path.join("datasets", "input_training_lowres", scribble_image_name)
    I_scribble = skimage.io.imread(scribble_image_path)  # this is mI in MATLAB code
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


    skimage.io.imshow(spndimage.convolve1d(I, np.array([1,2,1])/4, axis=0, mode="constant"))
    skimage.io.show()






