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
    OUTPUT_ROOTDIR = "output"
    LOWRES_IMAGE_DIR = "../datasets/input_training_lowres"
    LOWRES_TRIMAP_DIR = "../datasets/trimap_training_lowres/Trimap2"
    LOWRES_GROUNDTRUTH_DIR = "../datasets/gt_training_lowres"
    # HIGHRES_IMAGE_DIR = "../datasets/input_training_highres"
    # HIGHRES_TRIMAP_DIR = "../datasets/trimap_training_highres/Trimap2"
    # HIGHRES_GROUNDTRUTH_DIR = "../datasets/gt_training_highres"


    ##############################
    # LOWRES Closed Form Matting #
    ##############################
    scheme_tag = f"lowres-closedform"
    output_dir = os.path.join(OUTPUT_ROOTDIR, scheme_tag)
    os.makedirs(output_dir, exist_ok=True)
    timings_file_path = os.path.join(OUTPUT_ROOTDIR, f"{scheme_tag}-timings.txt")
    with open(timings_file_path, "a") as f:
        f.write(f"image_name,time_taken\n")
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
        print(f"[{scheme_tag}] {image_path} ...")
        start = time.time_ns()

        logging.info(f"Opening {args.image_path}")
        I = skimage.io.imread(args.image_path) / 255
        I = closedform.utils.ensure_3d_image(I)  # H x W x C array

        logging.info(f"Opening {args.scribble_path}")
        I_scribble = skimage.io.imread(args.scribble_path) / 255 # this is mI in MATLAB code
        assert I_scribble.shape[:2] == I.shape[:2], f"Shapes of scribble image ({I_scribble.shape[:2]}) and image ({I.shape[:2]}) don't match"
        I_scribble_gray2D = closedform.utils.matlab_compatible_rgb2gray(I_scribble) if len(I_scribble.shape) == 3 else I_scribble  # H x W array
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
        skimage.io.imsave(os.path.join(output_dir, image_name), skimage.util.img_as_ubyte(alpha))

        end = time.time_ns()
        time_taken = end - start
        print(f"[{scheme_tag}] {image_path} took {time_taken} ns")
        with open(timings_file_path, "a") as f:
            f.write(f"{image_name},{time_taken}\n")

    #####################################
    # LOWRES Robust: Global Random Same #
    #####################################
    for sampling_method in (
        "global_random_same",
        "global_random",
        "local_random",
        "nearest",
        "global_spread",
        "local_spread"
    ):
        if sampling_method in ("local_random", "local_spread"):
            # These methods use nearest_candidates_count
            nearest_candidates_counts = (50, 100, 200, 500, 1000)
        else:
            # The rest don't
            nearest_candidates_counts = (200,)
        for nearest_candidates_count in nearest_candidates_counts:
            if sampling_method not in ("local_random", "local_spread"):
                scheme_tag = f"lowres-robust-{sampling_method}"
                output_dir = os.path.join(OUTPUT_ROOTDIR, scheme_tag)
                os.makedirs(output_dir, exist_ok=True)
                timings_file_path = os.path.join(OUTPUT_ROOTDIR, f"{scheme_tag}-timings.txt")
            else:
                scheme_tag = f"lowres-robust-{sampling_method}-nearest_candidates_count{nearest_candidates_count}"
                output_dir = os.path.join(OUTPUT_ROOTDIR, scheme_tag)
                os.makedirs(output_dir, exist_ok=True)
                timings_file_path = os.path.join(OUTPUT_ROOTDIR, f"{scheme_tag}-timings.txt")

            with open(timings_file_path, "a") as f:
                f.write(f"image_name,time_taken\n")
            for image_name in os.listdir(LOWRES_IMAGE_DIR):
                image_path = os.path.join(LOWRES_IMAGE_DIR, image_name)
                trimap_path = os.path.join(LOWRES_TRIMAP_DIR, image_name)
                # groundtruth_path = os.path.join(LOWRES_GROUNDTRUTH_DIR, image_name)

                args = argparse.Namespace(
                    image_path=image_path,
                    trimap_path=trimap_path,
                    foreground_samples_count=20,
                    background_samples_count=20,
                    sampling_method=sampling_method,
                    nearest_candidates_count=nearest_candidates_count,
                    sigma_squared=0.01,
                    highest_confidence_pairs_to_select=3,
                    epsilon=1e-5,
                    gamma=0.1,
                    window_size=1
                )
                print(f"[{scheme_tag}] {image_path} ...")
                start = time.time_ns()

                logging.info(f"Opening {args.image_path}")
                I = skimage.io.imread(args.image_path) / 255
                I = closedform.utils.ensure_3d_image(I)  # H x W x C array

                logging.info(f"Opening {args.trimap_path}")
                I_trimap = skimage.io.imread(args.trimap_path) / 255 # this is mI in MATLAB code
                assert I_trimap.shape[:2] == I.shape[:2], f"Shapes of trimap image ({I_trimap.shape[:2]}) and image ({I.shape[:2]}) don't match"
                I_trimap_gray2D = closedform.utils.matlab_compatible_rgb2gray(I_trimap) if len(I_trimap.shape) == 3 else I_trimap  # H x W array

                foreground_map = np.absolute(I_trimap_gray2D - 1) < 1e-3  # H x W bool array
                background_map = np.absolute(I_trimap_gray2D) < 1e-3  # H x W bool array
                unknown_map = np.ones(I_trimap_gray2D.shape, dtype=bool) ^ (foreground_map | background_map)  # H x W bool array

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
                skimage.io.imsave(os.path.join(output_dir, image_name), skimage.util.img_as_ubyte(alpha))

                end = time.time_ns()
                time_taken = end - start
                print(f"[{scheme_tag}] {image_path} took {time_taken} ns")
                with open(timings_file_path, "a") as f:
                    f.write(f"{image_name},{time_taken}\n")
