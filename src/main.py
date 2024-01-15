import argparse
import logging

import numpy as np
import scipy.ndimage as spndimage
import skimage

import closedform.coarsetofine
import robust.entry
import utils

# python3 src/main.py -v closedform -i datasets/input_training_lowres/GT01.png -s datasets/input_training_lowres/GT01-scribble.png -l 4 -L 2
# python3 src/main.py -v closedform -i datasets/input_training_lowres/GT02.png -s datasets/input_training_lowres/GT02-scribble.png -l 4 -L 2
# python3 src/main.py -v closedform -i matlab/peacock.bmp -s matlab/peacock_m.bmp -l 4 -L 2

# Trimap1 is finer.
# Trimap2 is coarser.
# python3 src/main.py -v robust -i datasets/input_training_lowres/GT01.png -t datasets/trimap_training_lowres/Trimap1/GT01.png
# python3 src/main.py -v robust -i datasets/input_training_lowres/GT01.png -t datasets/trimap_training_lowres/Trimap2/GT01.png
# ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python3 main.py",
        description="Perform alpha matting on given inputs",
        epilog="jwek2",
    )
    parser.add_argument("-v", "--verbose", action="count", help="Enable logging, maximum twice", default=0)
    subparsers = parser.add_subparsers(dest="subcommand", title="subcommands", help="additional help", required=True)

    closed_form_parser = subparsers.add_parser("closedform")
    closed_form_parser.add_argument("-i", "--image-path", type=str, help="Path to original image", required=True)
    closed_form_parser.add_argument("-t", "--trimap-mode", action="store_true", help="Use this flag is a trimap is provided via -s/--scribble-path. Erosion on the trimap will be performed to obtain a \"scribble\".")
    closed_form_parser.add_argument("-T", "--trimap-erosion-window-size", type=int, default=0, help="One-sided window size to be used as a square kernel for erosion of the trimap. If 0, no erosion is performed. (default: %(default)s)")
    closed_form_parser.add_argument("-s", "--scribble-path", type=str, help="Path to scribble", required=True)
    closed_form_parser.add_argument("-a", "--alpha-threshold", type=float, default=0.02, help="Alpha values within this close to 0 and 1 are pushed to 0 and 1 respectively (default: %(default)s)")
    closed_form_parser.add_argument("-e", "--epsilon", type=float, default=1e-7, help="Epsilon parameter (default: %(default)s)")
    closed_form_parser.add_argument("-w", "--window-size", type=int, choices=(1, 2), default=1, help="One-sided window size. 1 means 3x3 window. 2 means 5x5 window. (default: %(default)s)")
    closed_form_parser.add_argument("-l", "--levels-count", type=int, default=4, help="Number of layers in coarse-to-fine pyramid, with each coarser layer downsampled from the previous by a factor of 2 (default: %(default)s)")
    closed_form_parser.add_argument("-L", "--explicit-alpha-levels-count", type=int, default=2, help="Number of coarsest levels to explicitly calculate alphas; the finer (levels_count - explicit_alpha_levels_count) levels will not solve for the Laplacian directly but interpolate linear coefficients to derive alpha. Require that active_levels_num <= levels_num. (default: %(default)s)")

    robust_parser = subparsers.add_parser("robust")
    robust_parser.add_argument("-i", "--image-path", type=str, help="Path to original image", required=True)
    robust_parser.add_argument("-t", "--trimap-path", type=str, help="Path to trimap")
    robust_parser.add_argument("-f", "--foreground-samples-count", type=int, default=20, help="Number of foreground pixels to sample for each unknown pixel (default: %(default)s)")
    robust_parser.add_argument("-b", "--background-samples-count", type=int, default=20, help="Number of background pixels to sample for each unknown pixel (default: %(default)s)")
    robust_parser.add_argument("-m", "--sampling-method", type=str, choices=("global_random_same", "global_random", "local_random", "nearest", "global_spread", "local_spread"), default="local_spread", help="How to sample foreground and background pixels for each unknown pixel. For more details, see documentation of the `get_samples()` method in source code (default: %(default)s)")
    robust_parser.add_argument("-x", "--nearest-candidates-count", type=int, default=200, help="Only effective when -m/--sampling-method is `local_random` or `local_spread`. How many nearest candidates to choose before randomly choosing `fore/background-samples-count`. For more details, see documentation of the `get_samples()` method in source code (default: %(default)s)")
    robust_parser.add_argument("-s", "--sigma-squared", type=float, default=0.01, help="Sigma parameter, squared (default: %(default)s)")
    robust_parser.add_argument("-n", "--highest-confidence_pairs_to_select", type=int, default=3, help="Number of highest confidence pairs to select for estimation of alpha for each unknown pixel (default: %(default)s)")
    robust_parser.add_argument("-e", "--epsilon", type=float, default=1e-5, help="Epsilon parameter (default: %(default)s)")
    robust_parser.add_argument("-g", "--gamma", type=float, default=0.1, help="Gamma parameter (default: %(default)s)")
    robust_parser.add_argument("-w", "--window-size", type=int, choices=(1, 2), default=1, help="One-sided window size. 1 means 3x3 window. 2 means 5x5 window. (default: %(default)s)")

    args = parser.parse_args()

    if args.verbose == 1:
        logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%y%m%d %H%M%S", level=logging.INFO)
    elif args.verbose > 1:
        logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%y%m%d %H%M%S", level=logging.DEBUG)
        logging.debug(args)

    if args.subcommand == "closedform":
        logging.info(f"Opening {args.image_path}")
        I = skimage.io.imread(args.image_path) / 255
        I = utils.ensure_3d_image(I)  # H x W x C array

        logging.info(f"Opening {args.scribble_path}")
        I_scribble = skimage.io.imread(args.scribble_path) / 255 # this is mI in MATLAB code
        assert I_scribble.shape[:2] == I.shape[:2], f"Shapes of scribble image ({I_scribble.shape[:2]}) and image ({I.shape[:2]}) don't match"
        I_scribble_gray2D = utils.matlab_compatible_rgb2gray(I_scribble) if len(I_scribble.shape) == 3 else I_scribble  # H x W array
        if args.trimap_mode:
            constrained_map = np.isclose(I_scribble_gray2D, 0, atol=1e-3) | np.isclose(I_scribble_gray2D, 1, atol=1e-3)
            constrained_map = spndimage.binary_erosion(constrained_map, np.ones((1 + 2 * args.trimap_erosion_window_size, 1 + 2
            * args.trimap_erosion_window_size))).astype(float)
        else:
            I_scribble = utils.ensure_3d_image(I_scribble)  # H x W x C array
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
    elif args.subcommand == "robust":
        logging.info(f"Opening {args.image_path}")
        I = skimage.io.imread(args.image_path) / 255
        I = utils.ensure_3d_image(I)  # H x W x C array

        logging.info(f"Opening {args.trimap_path}")
        I_trimap = skimage.io.imread(args.trimap_path) / 255 # this is mI in MATLAB code
        assert I_trimap.shape[:2] == I.shape[:2], f"Shapes of trimap image ({I_trimap.shape[:2]}) and image ({I.shape[:2]}) don't match"
        I_trimap_gray2D = utils.matlab_compatible_rgb2gray(I_trimap) if len(I_trimap.shape) == 3 else I_trimap  # H x W array

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
    else:
        raise NotImplementedError

    alpha = np.clip(alpha, 0, 1)  # there tends to be some alphas that lie outside of [0, 1]...
    skimage.io.imshow(alpha)
    skimage.io.show()
