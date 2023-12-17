import itertools
import math
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# import skimage
# import skimage.filters as filters
import skimage
from scipy import ndimage, signal
from scipy.interpolate import interp1d

# from skimage import color, util
# from skimage.util import img_as_uint
from sksparse.cholmod import cholesky
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map  # internally uses concurrent.futures


def parallelcompute_mu_k(neighbourhood):
    # pixels is a HW x W^2 array
    return np.mean(neighbourhood, axis=0)
    # output is a length-3 vector

def parallelcompute_Sigma_k(neighbourhood):
    # pixels is a HW x W^2 array
    return np.cov(neighbourhood, rowvar=False, bias=True)  # bias True to divide by M^2 instead of (M^2 - 1)
    # output is a 3 x 3 matrix



# given coordinates in padded image (paddedx, paddedy) outputs i \in {0, ..., HW - 1} (or None if is a zero-padding pixel)
def padded_to_unpadded_indexmap(unpadded_to_padded_indexmap, paddedx, paddedy):
    optional = np.where((unpadded_to_padded_indexmap[:, 0] == paddedx) & (unpadded_to_padded_indexmap[:, 1] == paddedy))
    assert len(optional) == 1  # TODO: remove
    optional = optional[0]
    if len(optional) == 1:
        return optional[0]
    if len(optional) == 0:
        return None
    raise Exception("len(optional) > 1, impossible case")

# the applicable ks are, at a high level, the intersection of the respective neighbourhoods of i and j
def k_summation_range_given_ij(i, j, pad_amount, unpadded_to_padded_indexmap):
    # i, j \in {0, ..., HW - 1}
    i_paddedx, i_paddedy = unpadded_to_padded_indexmap[i]
    i_neighbourhood = {
        (paddedx, paddedy)
        for paddedx in range(i_paddedx - pad_amount, i_paddedx + pad_amount + 1)
        for paddedy in range(i_paddedy - pad_amount, i_paddedy + pad_amount + 1)
    }
    j_paddedx, j_paddedy = unpadded_to_padded_indexmap[j]
    j_neighbourhood = {
        (paddedx, paddedy)
        for paddedx in range(j_paddedx - pad_amount, j_paddedx + pad_amount + 1)
        for paddedy in range(j_paddedy - pad_amount, j_paddedy + pad_amount + 1)
    }
    return [
        k
        for paddedx, paddedy in i_neighbourhood.intersection(j_neighbourhood)
        if (k := padded_to_unpadded_indexmap(unpadded_to_padded_indexmap, paddedx, paddedy)) is not None
    ]  # syntax courtesy of https://stackoverflow.com/a/48609910/7254995

def parallelcompute_L_ij(ij, I, mu_ks, Sigma_ks, M, pad_amount, unpadded_to_padded_indexmap, epsilon=1e-7):
    i, j = ij
    k_summation_range = k_summation_range_given_ij(i, j, pad_amount, unpadded_to_padded_indexmap)
    if len(k_summation_range) == 0:
        return 0

    result = 0 if i != j else len(k_summation_range)
    I_i = I[i]    # [:, None] converts to column-vector (2D)
    I_j = I[j]    # [:, None] converts to column-vector (2D)
    M2 = M ** 2
    for k in k_summation_range:
        mu_k = mu_ks[k]        # length-3 vector (1D)
        Sigma_k = Sigma_ks[k]  # 3 x 3 matrix (2D)
        # NB. [:, None] converts 1D vector to column-vector (2D)
        # NB. [None, :] converts 1D vector to row-vector (2D)
        result -= (1 + ((I_i - mu_k)[None, :] @ np.linalg.inv(Sigma_k + epsilon / M2 * np.eye(3)) @ (I_j - mu_k)[:, None])) / M2
    return result

if __name__ == "__main__":
    image_name = "GT01.png"
    image_path = os.path.join("../datasets", "input_training_lowres", image_name)
    im = skimage.io.imread(image_path)
    I = im.reshape(-1, 3)  # indices {0, ..., HW - 1} is row-major traversal of image; row i in this matrix is I_i
    H, W, C = im.shape  # dim 0: rows. dim 1: cols. dim 2: colours.


    M = 3  # window size
    assert M % 2 != 0, f"Window size M should be even, got {M}"
    pad_amount = (M - 1) // 2
    zero_padded_im = np.zeros((H + pad_amount * 2, W + pad_amount * 2, 3))
    zero_padded_im[pad_amount:-pad_amount, pad_amount:-pad_amount] = im
    zero_padded_im[:,:,0]

    t = np.zeros((H + pad_amount * 2, W + pad_amount * 2))
    t[pad_amount:-pad_amount, pad_amount:-pad_amount] = 1
    # given i \in {0, ..., HW - 1} outputs coordinates in padded image (paddedx, paddedy)
    unpadded_to_padded_indexmap = np.array([np.array([x, y]) for x, y in zip(*t.nonzero())])
    # padded_to_unpadded_indexmap: defined as function at top level

    neighbourhoods = []
    ij_to_k_summation_range = defaultdict(list) #  {(i, j): [] for i in range(H * W) for j in range(H * W)} will OOM
    print("Computing neighbourhoods centred at each image pixel...")
    # k is centre of window
    for k in tqdm(np.arange(H * W)):
        k_paddedx, k_paddedy = unpadded_to_padded_indexmap[k]
        neighbourhood_of_k = zero_padded_im[
            k_paddedx - pad_amount:k_paddedx + pad_amount + 1,
            k_paddedy - pad_amount:k_paddedy + pad_amount + 1
        ].reshape(-1, 3)
        neighbourhoods.append(neighbourhood_of_k)

        # ! THIS CODE BLOCK IS TOO SLOW
        # unpadded_indices_in_neighbourhood = [] # subset of {0, ..., HW - 1}
        # for paddedx in range(k_paddedx - pad_amount, k_paddedx + pad_amount + 1):
        #     for paddedy in range(k_paddedy - pad_amount, k_paddedy + pad_amount + 1):
        #         if (res := padded_to_unpadded_indexmap(unpadded_to_padded_indexmap, paddedx, paddedy)) is not None:
        #             unpadded_indices_in_neighbourhood.append(res)
        # for i, j in itertools.product(unpadded_indices_in_neighbourhood, unpadded_indices_in_neighbourhood):
        #     ij_to_k_summation_range[(i, j)].append(k)

    # split into (approximately) 16 chunks
    print("Computing neighbourhood mean vectors...")
    mu_ks = np.array(process_map(parallelcompute_mu_k, neighbourhoods, chunksize=math.ceil(len(neighbourhoods) / 16)))
    np.save("mu_ks.npy", np.array(mu_ks))
    print("Computing neighbourhood covariance matrices...")
    Sigma_ks = np.array(process_map(parallelcompute_Sigma_k, neighbourhoods, chunksize=math.ceil(len(neighbourhoods) / 16)))
    np.save("Sigma_ks.npy", np.array(Sigma_ks))

    L = sp.sparse.lil_matrix((H * W, H * W))  # sparse matrix
    # mu_ks = np.array(process_map(parallelcompute_mu_k, neighbourhoods, chunksize=math.ceil(len(neighbourhoods) / 16)))

    print("Computing entries of L...")
    # ! WARNING THIS TAKES 400 HOURS LINEARLY. PARALLELISE WON'T HELP.
    for ij in tqdm(itertools.product(range(H * W), range(H * W)), total=H * W * H * W):
        parallelcompute_L_ij(ij, I, mu_ks, Sigma_ks, M, pad_amount, unpadded_to_padded_indexmap, epsilon=1e-7)
    # print(t)
    # L_ijs = process_map(
    #     lambda ij: parallelcompute_L_ij(ij, I, mu_ks, Sigma_ks, M, pad_amount, unpadded_to_padded_indexmap, epsilon=1e-7),
    #     itertools.product(range(H * W), range(H * W)),
    #     chunksize=math.ceil(H * W * H * W / 16)
    # )





