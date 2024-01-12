import argparse
import logging
import os
import time

import numpy as np
import scipy.ndimage as spndimage
import skimage

import closedform.coarsetofine
import closedform.utils
import robust.entry

# python3 src/main.py -v closedform -i datasets/input_training_lowres/GT01.png -s datasets/input_training_lowres/GT01-scribble.png -l 4 -L 2
# python3 src/main.py -v closedform -i datasets/input_training_lowres/GT02.png -s datasets/input_training_lowres/GT02-scribble.png -l 4 -L 2
# python3 src/main.py -v closedform -i matlab/peacock.bmp -s matlab/peacock_m.bmp -l 4 -L 2

# Trimap1 is finer.
# Trimap2 is coarser.
# python3 src/main.py -v robust -i datasets/input_training_lowres/GT01.png -t datasets/trimap_training_lowres/Trimap1/GT01.png
# python3 src/main.py -v robust -i datasets/input_training_lowres/GT01.png -t datasets/trimap_training_lowres/Trimap2/GT01.png
# ...

if __name__ == "__main__":
    LOWRES_IMAGE_DIR = "../datasets/input_training_lowres"
    LOWRES_TRIMAP_DIR = "../datasets/trimap_training_lowres/Trimap2"
    LOWRES_GROUNDTRUTH_DIR = "../datasets/gt_training_lowres"
    LOWRES_OUTPUT_DIR_CLOSEDFORM = "output/lowres/closedform"
    LOWRES_OUTPUT_DIR_ROBUST = "output/lowres/robust"
    HIGHRES_IMAGE_DIR = "../datasets/input_training_highres"
    HIGHRES_TRIMAP_DIR = "../datasets/trimap_training_highres/Trimap2"
    HIGHRES_GROUNDTRUTH_DIR = "../datasets/gt_training_highres"
    HIGHRES_OUTPUT_DIR_CLOSEDFORM = "output/highres/closedform"
    HIGHRES_OUTPUT_DIR_ROBUST = "output/highres/robust"

    os.makedirs(LOWRES_OUTPUT_DIR_CLOSEDFORM, exist_ok=True)
    os.makedirs(LOWRES_OUTPUT_DIR_ROBUST, exist_ok=True)
    os.makedirs(HIGHRES_OUTPUT_DIR_CLOSEDFORM, exist_ok=True)
    os.makedirs(HIGHRES_OUTPUT_DIR_ROBUST, exist_ok=True)

    with open("lowres-closedform-timings.txt", "a") as f:
        f.write(f"image_name,H,W,C,known_count,unknown_count,time_taken\n")
    for image_name in os.listdir(LOWRES_IMAGE_DIR):
        image_path = os.path.join(LOWRES_IMAGE_DIR, image_name)
        trimap_path = os.path.join(LOWRES_TRIMAP_DIR, image_name)
        # groundtruth_path = os.path.join(LOWRES_GROUNDTRUTH_DIR, image_name)

        args = argparse.Namespace(
            image_path=image_path,
            trimap_mode=True,
            trimap_erosion_window_size=0,
            scribble_path=trimap_path,
            alpha_threshold=0.02,
            epsilon=1e-7,
            window_size=1,
            levels_count=4,
            explicit_alpha_levels_count=2
        )
        print(f"[lowres-closedform] {image_path} ...")
        start = time.time_ns()

        logging.info(f"Opening {args.image_path}")
        I = skimage.io.imread(args.image_path) / 255
        I = closedform.utils.ensure_3d_image(I)  # H x W x C array

        logging.info(f"Opening {args.scribble_path}")
        I_scribble = skimage.io.imread(args.scribble_path) / 255 # this is mI in MATLAB code
        assert I_scribble.shape == I.shape, f"Shapes of scribble image ({I_scribble.shape}) and image ({I.shape}) don't match"
        I_scribble_gray2D = closedform.utils.matlab_compatible_rgb2gray(I_scribble) if I.shape[2] == 3 else I_scribble  # H x W array
        if args.trimap_mode:
            constrained_map = np.isclose(I_scribble_gray2D, 0, atol=1e-3) | np.isclose(I_scribble_gray2D, 1, atol=1e-3)
            constrained_map = spndimage.binary_erosion(constrained_map, np.ones((1 + 2 * args.trimap_erosion_window_size, 1 + 2
            * args.trimap_erosion_window_size))).astype(float)
        else:
            I_scribble = closedform.utils.ensure_3d_image(I_scribble)  # H x W x C array
            constrained_map = (np.sum(np.absolute(I - I_scribble), axis=2) > 1e-3).astype(float)  # this is consts_map in MATLAB; H x W array
        constrained_vals = I_scribble_gray2D * constrained_map  # NB. * is the element-wise multiply operator; H x W array

        alpha = closedform.coarsetofine.solve_alpha_coarse_to_fine(
            I,
            constrained_map,
            constrained_vals,
            args.levels_count,
            args.explicit_alpha_levels_count,
            args.alpha_threshold,
            args.epsilon,
            args.window_size
        )  # H x W array

        alpha = np.clip(alpha, 0, 1)  # there tends to be some alphas that lie outside of [0, 1]...
        skimage.io.imsave(os.path.join(LOWRES_OUTPUT_DIR_CLOSEDFORM, image_name), skimage.util.img_as_ubyte(alpha))

        end = time.time_ns()
        time_taken = end - start
        print(f"[lowres-closedform] {image_path} took {time_taken} ns")
        H, W, C = I.shape
        unknown_count = np.count_nonzero(~constrained_map.astype(bool))
        known_count = np.count_nonzero(constrained_map.astype(bool))
        with open("lowres-closedform-timings.txt", "a") as f:
            f.write(f"{image_name},{H},{W},{C},{known_count},{unknown_count},{time_taken}\n")

    with open("lowres-robust-timings.txt", "a") as f:
        f.write(f"image_name,H,W,C,known_count,unknown_count,time_taken\n")
    for image_name in os.listdir(LOWRES_IMAGE_DIR):
        image_path = os.path.join(LOWRES_IMAGE_DIR, image_name)
        trimap_path = os.path.join(LOWRES_TRIMAP_DIR, image_name)
        # groundtruth_path = os.path.join(LOWRES_GROUNDTRUTH_DIR, image_name)

        args = argparse.Namespace(
            image_path=image_path,
            trimap_path=trimap_path,
            foreground_samples_count=20,
            background_samples_count=20,
            sampling_method="deterministic_spread_local",
            nearest_candidates_count=200,
            sigma_squared=0.01,
            highest_confidence_pairs_to_select=3,
            epsilon=1e-5,
            gamma=0.1,
            window_size=1
        )
        print(f"[lowres-robust] {image_path} ...")
        start = time.time_ns()

        logging.info(f"Opening {args.image_path}")
        I = skimage.io.imread(args.image_path) / 255
        I = closedform.utils.ensure_3d_image(I)  # H x W x C array

        logging.info(f"Opening {args.trimap_path}")
        I_trimap = skimage.io.imread(args.trimap_path) / 255 # this is mI in MATLAB code
        assert I_trimap.shape == I.shape, f"Shapes of trimap image ({I_trimap.shape}) and image ({I.shape}) don't match"
        I_trimap_gray2D = closedform.utils.matlab_compatible_rgb2gray(I_trimap) if I.shape[2] == 3 else I_trimap  # H x W array

        foreground_map = np.absolute(I_trimap_gray2D - 1) < 1e-3  # H x W bool array
        background_map = np.absolute(I_trimap_gray2D) < 1e-3  # H x W bool array
        unknown_map = np.ones(I_trimap_gray2D.shape, dtype=bool) ^ (foreground_map | background_map)  # H x W bool array

        # skimage.io.imshow(foreground_map)
        # skimage.io.show()
        # skimage.io.imshow(background_map)
        # skimage.io.show()
        # skimage.io.imshow(unknown_map)
        # skimage.io.show()

        alpha = robust.entry.solve_alpha(
            I,
            foreground_map,
            background_map,
            unknown_map,
            args.foreground_samples_count,
            args.background_samples_count,
            args.sampling_method,
            args.nearest_candidates_count,
            args.sigma_squared,
            args.highest_confidence_pairs_to_select,
            args.epsilon,
            args.gamma,
            args.window_size
        )  # H x W array

        alpha = np.clip(alpha, 0, 1)  # there tends to be some alphas that lie outside of [0, 1]...
        skimage.io.imsave(os.path.join(LOWRES_OUTPUT_DIR_ROBUST, image_name), skimage.util.img_as_ubyte(alpha))

        end = time.time_ns()
        time_taken = end - start
        print(f"[lowres-robust] {image_path} took {time_taken} ns")
        H, W, C = I.shape
        unknown_count = np.count_nonzero(unknown_map)
        known_count = np.count_nonzero(~unknown_map)
        with open("lowres-robust-timings.txt", "a") as f:
            f.write(f"{image_name},{H},{W},{C},{known_count},{unknown_count},{time_taken}\n")

    with open("highres-closedform-timings.txt", "a") as f:
        f.write(f"image_name,H,W,C,known_count,unknown_count,time_taken\n")
    for image_name in os.listdir(HIGHRES_IMAGE_DIR):
        image_path = os.path.join(HIGHRES_IMAGE_DIR, image_name)
        trimap_path = os.path.join(HIGHRES_TRIMAP_DIR, image_name)
        # groundtruth_path = os.path.join(HIGHRES_GROUNDTRUTH_DIR, image_name)

        args = argparse.Namespace(
            image_path=image_path,
            trimap_mode=True,
            trimap_erosion_window_size=0,
            scribble_path=trimap_path,
            alpha_threshold=0.02,
            epsilon=1e-7,
            window_size=1,
            levels_count=4,
            explicit_alpha_levels_count=2
        )
        print(f"[highres-closedform] {image_path} ...")
        start = time.time_ns()

        logging.info(f"Opening {args.image_path}")
        I = skimage.io.imread(args.image_path) / 255
        I = closedform.utils.ensure_3d_image(I)  # H x W x C array

        logging.info(f"Opening {args.scribble_path}")
        I_scribble = skimage.io.imread(args.scribble_path) / 255 # this is mI in MATLAB code
        assert I_scribble.shape == I.shape, f"Shapes of scribble image ({I_scribble.shape}) and image ({I.shape}) don't match"
        I_scribble_gray2D = closedform.utils.matlab_compatible_rgb2gray(I_scribble) if I.shape[2] == 3 else I_scribble  # H x W array
        if args.trimap_mode:
            constrained_map = np.isclose(I_scribble_gray2D, 0, atol=1e-3) | np.isclose(I_scribble_gray2D, 1, atol=1e-3)
            constrained_map = spndimage.binary_erosion(constrained_map, np.ones((1 + 2 * args.trimap_erosion_window_size, 1 + 2
            * args.trimap_erosion_window_size))).astype(float)
        else:
            I_scribble = closedform.utils.ensure_3d_image(I_scribble)  # H x W x C array
            constrained_map = (np.sum(np.absolute(I - I_scribble), axis=2) > 1e-3).astype(float)  # this is consts_map in MATLAB; H x W array
        constrained_vals = I_scribble_gray2D * constrained_map  # NB. * is the element-wise multiply operator; H x W array

        alpha = closedform.coarsetofine.solve_alpha_coarse_to_fine(
            I,
            constrained_map,
            constrained_vals,
            args.levels_count,
            args.explicit_alpha_levels_count,
            args.alpha_threshold,
            args.epsilon,
            args.window_size
        )  # H x W array

        alpha = np.clip(alpha, 0, 1)  # there tends to be some alphas that lie outside of [0, 1]...
        skimage.io.imsave(os.path.join(HIGHRES_OUTPUT_DIR_CLOSEDFORM, image_name), skimage.util.img_as_ubyte(alpha))

        end = time.time_ns()
        time_taken = end - start
        print(f"[highres-closedform] {image_path} took {time_taken} ns")
        H, W, C = I.shape
        unknown_count = np.count_nonzero(~constrained_map.astype(bool))
        known_count = np.count_nonzero(constrained_map.astype(bool))
        with open("highres-closedform-timings.txt", "a") as f:
            f.write(f"{image_name},{H},{W},{C},{known_count},{unknown_count},{time_taken}\n")

    with open("highres-robust-timings.txt", "a") as f:
        f.write(f"image_name,H,W,C,known_count,unknown_count,time_taken\n")
    for image_name in os.listdir(HIGHRES_IMAGE_DIR):
        image_path = os.path.join(HIGHRES_IMAGE_DIR, image_name)
        trimap_path = os.path.join(HIGHRES_TRIMAP_DIR, image_name)
        # groundtruth_path = os.path.join(HIGHRES_GROUNDTRUTH_DIR, image_name)

        args = argparse.Namespace(
            image_path=image_path,
            trimap_path=trimap_path,
            foreground_samples_count=20,
            background_samples_count=20,
            sampling_method="deterministic_spread_local",
            nearest_candidates_count=200,
            sigma_squared=0.01,
            highest_confidence_pairs_to_select=3,
            epsilon=1e-5,
            gamma=0.1,
            window_size=1
        )
        print(f"[highres-robust] {image_path} ...")
        start = time.time_ns()

        logging.info(f"Opening {args.image_path}")
        I = skimage.io.imread(args.image_path) / 255
        I = closedform.utils.ensure_3d_image(I)  # H x W x C array

        logging.info(f"Opening {args.trimap_path}")
        I_trimap = skimage.io.imread(args.trimap_path) / 255 # this is mI in MATLAB code
        assert I_trimap.shape == I.shape, f"Shapes of trimap image ({I_trimap.shape}) and image ({I.shape}) don't match"
        I_trimap_gray2D = closedform.utils.matlab_compatible_rgb2gray(I_trimap) if I.shape[2] == 3 else I_trimap  # H x W array

        foreground_map = np.absolute(I_trimap_gray2D - 1) < 1e-3  # H x W bool array
        background_map = np.absolute(I_trimap_gray2D) < 1e-3  # H x W bool array
        unknown_map = np.ones(I_trimap_gray2D.shape, dtype=bool) ^ (foreground_map | background_map)  # H x W bool array

        # skimage.io.imshow(foreground_map)
        # skimage.io.show()
        # skimage.io.imshow(background_map)
        # skimage.io.show()
        # skimage.io.imshow(unknown_map)
        # skimage.io.show()

        alpha = robust.entry.solve_alpha(
            I,
            foreground_map,
            background_map,
            unknown_map,
            args.foreground_samples_count,
            args.background_samples_count,
            args.sampling_method,
            args.nearest_candidates_count,
            args.sigma_squared,
            args.highest_confidence_pairs_to_select,
            args.epsilon,
            args.gamma,
            args.window_size
        )  # H x W array

        alpha = np.clip(alpha, 0, 1)  # there tends to be some alphas that lie outside of [0, 1]...
        skimage.io.imsave(os.path.join(HIGHRES_OUTPUT_DIR_ROBUST, image_name), skimage.util.img_as_ubyte(alpha))

        end = time.time_ns()
        time_taken = end - start
        print(f"[highres-robust] {image_path} took {time_taken} ns")
        H, W, C = I.shape
        unknown_count = np.count_nonzero(unknown_map)
        known_count = np.count_nonzero(~unknown_map)
        with open("highres-robust-timings.txt", "a") as f:
            f.write(f"{image_name},{H},{W},{C},{known_count},{unknown_count},{time_taken}\n")




