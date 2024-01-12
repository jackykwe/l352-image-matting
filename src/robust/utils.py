import numpy as np
import scipy.sparse as spsparse

DIVISION_EPSILON = 1e-100

# Courtesy of https://stackoverflow.com/a/47771340/7254995
def sparse_allclose(sparse_A, sparse_B, sparse_A_var_name, sparse_B_var_name):
    # If you want to check matrix shapes as well
    if np.array_equal(sparse_A.shape, sparse_B.shape)==0:
        return False

    A_i, A_j, A_v = spsparse.find(sparse_A)
    B_i, B_j, B_v = spsparse.find(sparse_B)
    if np.array_equal(A_i,B_i) and np.array_equal(A_j,B_j):
        # print(f"Passed ij check for pair {sparse_A_var_name} {sparse_B_var_name}")
        return np.allclose(A_v, B_v)
    return False
