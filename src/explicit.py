import numpy as np
import numpy.matlib as npmatlib
import scipy.ndimage as spndimage
import scipy.sparse as spsparse
from scipy.sparse.linalg import spsolve
from sksparse.cholmod import cholesky


def get_laplacian(I, constrained_map, epsilon, window_size):
    """
    `I`: H x W x C array
    `constrained_map`: H x W array

    Returns: HW x HW sparse array
    """
    neighbourhood_size = (1 + 2 * window_size) ** 2
    neighbourhood_size_squared = neighbourhood_size ** 2
    H, W, C = I.shape
    # constrain a pixel iff all pixels in its neighbourhood are also constrained
    constrained_map = spndimage.binary_erosion(constrained_map.astype(bool), np.ones((1 + 2 * window_size, 1 + 2 * window_size))).astype(int)

    indices = np.arange(H * W).reshape((H, W))  # row-major
    # temp_length is the number of window centres (excluding a strip of width window_size around image) that we need to sum
    # i.e. those whose neighbourhoods contain at least one unconstrained pixel
    constructor_length = np.sum(1 - constrained_map[window_size:-window_size, window_size:-window_size]) * neighbourhood_size_squared  # this is tlen in MATLAB code; this is the length of the arguments to sp.sparse.csr_matrix(). Here in Python we also exploit the feature of csr_matrix() that accumulates values of duplicated indices

    # These three are arguments to the sp.sparse.csr_matrix() sparse matrix constructor
    constructor_row_indices = np.zeros(constructor_length)
    constructor_col_indices = np.zeros(constructor_length)
    constructor_vals = np.zeros(constructor_length)

    len = 0
    for i in range(window_size, H - window_size):
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
            window_var = np.linalg.inv((window_I.T @ window_I) / neighbourhood_size - window_mu @ window_mu.T + epsilon / neighbourhood_size * np.eye(C))
            window_I = window_I - npmatlib.repmat(window_mu.T, neighbourhood_size, 1)
            # NB. The following (temp_vals) is the 1/M^2 * (1 + (...)^T (...)^{-1} scalar term for (k)
            # to be used in the computation of Laplacian matrix elements for all M^2 x M^2 possible
            # (i, j) pairs represented by window_indices
            temp_vals = (1 + window_I @ window_var @ window_I.T) / neighbourhood_size  # neighbourhood_size x neighbourhood_size matrix

            constructor_vals[len:len + neighbourhood_size_squared] = temp_vals.flatten()  # row-major
            # corresponding row_indices should look like [0, 0, ..., 0                     , 1, 1, ..., 1                     , ..., neighbourhood_size_squared - 1]
            # corresponding col_indices should look like [0, 1, ..., neighbourhood_size - 1, 0, 1, ..., neighbourhood_size - 1, ..., neighbourhood_size_squared - 1]
            constructor_row_indices[len:len + neighbourhood_size_squared] = npmatlib.repmat(window_indices, neighbourhood_size, 1).flatten(order="F")
            constructor_col_indices[len:len + neighbourhood_size_squared] = npmatlib.repmat(window_indices, neighbourhood_size, 1).flatten()
            len += neighbourhood_size_squared
    assert constructor_length == len
    constructor_vals = constructor_vals[:len]  # TODO: redundant?
    constructor_row_indices = constructor_row_indices[:len]  # TODO: redundant?
    constructor_col_indices = constructor_col_indices[:len]  # TODO: redundant?
    W_mat = spsparse.csr_array((constructor_vals, (constructor_row_indices, constructor_col_indices)), shape=(H * W, H * W)) # this is A in MATLAB code
    W_mat_row_sum = W_mat.sum(axis=1)  # this is sumA in MATLAB code
    D = spsparse.diags(W_mat_row_sum, 0, shape=(H * W, H * W), format="csr")
    return D - W_mat


def solve_alpha_explicit(I, constrained_map, constrained_vals, epsilon, window_size):
    """
    `I`: H x W x C array
    `constrained_map`: H x W array
    `constrained_vals`: H x W array

    Returns: H x W array
    """
    H, W, C = I.shape
    L = get_laplacian(I, constrained_map, epsilon, window_size)
    D = spsparse.diags(constrained_map.flatten(), 0, shape=(H * W, H * W), format="csr")
    lagrangian_lambda = 100
    A = L + lagrangian_lambda * D
    b = lagrangian_lambda * (constrained_map * constrained_vals).flatten()
    # alpha = cholesky(A)(b)  # HW x 1 vector
    alpha = spsolve(A, b)  # HW x 1 vector
    return np.clip(alpha, 0, 1).reshape((H, W))  # TODO: why is clipping necessary?
