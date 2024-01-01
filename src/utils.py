import numpy as np


def ensure_3d_image(I):
    """
    If `I` is a H x W x C image, returns `I` with dimensions unchanged

    If `I` is a H x W image, adds a trailing dimension and returns shape H x W x 1.
    """
    I = I[..., np.newaxis] if len(I.shape) == 2 else I
    assert len(I.shape) == 3  # don't want images with alpha channels
    return I


def matlab_compatible_rgb2gray(I):
    """
    Turns a 3D (coloured) image into a 2D (greyscale) float image
    """
    return 0.2989 * I[:, :, 0] + 0.5870 * I[:, :, 1] + 0.1140 * I[:, :, 2]

