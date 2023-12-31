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

# from scipy import ndimage, signal
# from scipy.interpolate import interp1d

# from skimage import color, util
# from skimage.util import img_as_uint
# from sksparse.cholmod import cholesky
# from tqdm.auto import tqdm
# from tqdm.contrib.concurrent import process_map  # internally uses concurrent.futures

def get_laplacian(I, constrained_map, epsilon, window_size):
    """
    `I`: H x W x C array
    `constrained_map`: H x W array

    Returns: HW x HW array???????????????????
    """
    neighbourhood_size = (1 + 2 * window_size) ** 2
    H, W, C = I.shape
    # constrain a pixel iff all pixels in its neighbourhood are also constrained
    constrained_map = spndimage.binary_erosion(constrained_map.astype(bool), np.ones(1 + 2 * window_size)).astype(int)



    pass

def solve_alpha_explicit(I, constrained_map, constrained_vals, epsilon, window_size):
    """
    `I`: H x W x C array
    `constrained_map`: H x W array
    `constrained_vals`: H x W array
    """
    H, W, C = I.shape
    L = get_laplacian(I, constrained_map, epsilon, window_size)

    pass
