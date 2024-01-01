import math

import numpy as np
import scipy.ndimage as spndimage


def downsample_image(I, antialiasing_filter_size):
    """
    `I`: H x W x C float array.

    WARNING: It is crucial that `I` is a float array. If an int array is passed, application of
    anti-aliasing filter in this function will misbehave.

    Returns: ceil((H - 2 * antialiasing_filter_size) / 2) x ceil((W - 2 * antialiasing_filter_size) / 2) x C float array
    """
    assert I.dtype == float

    if antialiasing_filter_size not in (1, 2):
        raise NotImplementedError(f"Got filter_size={antialiasing_filter_size}, expected 1 or 2")

    filter = np.array([1, 2, 1]) / 4 if antialiasing_filter_size == 1 else np.array([1, 4, 6, 4, 1]) / 16
    I = spndimage.convolve1d(I, filter, axis=0, mode="constant")
    I = spndimage.convolve1d(I, filter, axis=1, mode="constant")
    I = I[antialiasing_filter_size:-antialiasing_filter_size:2, antialiasing_filter_size:-antialiasing_filter_size:2, :]

    assert I.dtype == float
    return I


def upsample_image(I, new_H, new_W, reconstruction_filter_size=1):
    """
    This function's size manipulations are the inverse to downsample_image().

    `I`: ceil((H - 2 * antialiasing_filter_size) / 2) x ceil((W - 2 * antialiasing_filter_size) / 2) x C float array

    Returns: H x W x C float array

    NB. `new_H` must be H, and `new_W` must be W, where H & W are the original image height and width
    before downsampling via downsample_image().
    """
    assert I.dtype == float

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
    result = np.zeros((new_H + 2 * reconstruction_filter_size, new_W + 2 * reconstruction_filter_size, I.shape[2]), dtype=float)

    # Insert into the correct places in the original pre-downsampled images, where downsampled values come from
    result[
        reconstruction_filter_size + top_strip_height:-reconstruction_filter_size - bottom_strip_height:2,
        reconstruction_filter_size + left_strip_width:-reconstruction_filter_size - right_strip_width:2,
    ] = I

    # Perform copy-padding (instead of zero-padding) for upsampling (interpolation) purposes
    # The following broadcasting nuance is courtesy of https://stackoverflow.com/questions/3551242/numpy-index-slice-without-losing-dimension-information#comment90059776_18183182
    result[reconstruction_filter_size + top_strip_height - 2::-2, :, :] = result[[reconstruction_filter_size + top_strip_height], :, :]  # repeat first row of downsampled pixels upwards for interpolation purposes
    result[-reconstruction_filter_size - bottom_strip_height + 1::2, :, :] = result[[-reconstruction_filter_size - bottom_strip_height - 1], :, :]  # repeat last row of downsampled pixels downwards for interpolation purposes
    result[:, reconstruction_filter_size + left_strip_width - 2::-2, :] = result[:, [reconstruction_filter_size + left_strip_width], :]  # repeat first column of downsampled pixels leftwards for interpolation purposes
    result[:, -reconstruction_filter_size - right_strip_width + 1::2, :] = result[:, [-reconstruction_filter_size - right_strip_width - 1], :]  # repeat last column of downsampled pixels right for interpolation purposes

    # The actual upsampling via a reconstruction filter
    result = spndimage.convolve1d(result, reconstruction_filter, axis=0, mode="constant")
    result = spndimage.convolve1d(result, reconstruction_filter, axis=1, mode="constant")
    result = result[reconstruction_filter_size:-reconstruction_filter_size, reconstruction_filter_size:-reconstruction_filter_size, :]

    assert result.dtype == float
    return result

