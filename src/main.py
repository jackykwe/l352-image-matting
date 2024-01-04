import argparse
import logging

import numpy as np
import skimage

import closedform.coarse_to_fine
import closedform.utils
import robust.entry

# python3 src/main.py -v closedform -i datasets/input_training_lowres/GT01.png -s datasets/input_training_lowres/GT01-scribble.png -l 4 -L 2
# python3 src/main.py -v closedform -i matlab/peacock.bmp -s matlab/peacock_m.bmp -l 4 -L 2

# Trimap1 is finer.
# Trimap2 is coarser.
# python3 src/main.py -v robust -i datasets/input_training_lowres/GT01.png -t datasets/trimap_training_lowres/Trimap1/GT01.png

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="L352 Mini-Project",
        description="Perform alpha matting on given inputs",
        epilog="jwek2",
    )
    parser.add_argument("-v", "--verbose", action="count", help="Enable logging, maximum twice", default=0)
    subparsers = parser.add_subparsers(dest="subcommand", title="subcommands", help="additional help", required=True)

    closed_form_parser = subparsers.add_parser("closedform")
    closed_form_parser.add_argument("-i", "--image-path", type=str, help="Path to original image", required=True)
    closed_form_parser.add_argument("-s", "--scribble-path", type=str, help="Path to scribble", required=True)
    closed_form_parser.add_argument("-a", "--alpha-threshold", type=int, default=0.02, help="Alpha values within this close to 0 and 1 are pushed to 0 and 1 respectively (default: %(default)s)")
    closed_form_parser.add_argument("-e", "--epsilon", type=int, default=1e-7, help="Epsilon parameter (default: %(default)s)")
    closed_form_parser.add_argument("-w", "--window-size", type=int, choices=(1, 2), default=1, help="One-sided window size. 1 means 3x3 window. 2 means 5x5 window. (default: %(default)s)")
    closed_form_parser.add_argument("-l", "--levels-count", type=int, default=4, help="Number of layers in coarse-to-fine pyramid, with each coarser layer downsampled from the previous by a factor of 2 (default: %(default)s)")
    closed_form_parser.add_argument("-L", "--explicit-alpha-levels-count", type=int, default=2, help="Number of coarsest levels to explicitly calculate alphas; the finer (levels_count - explicit_alpha_levels_count) levels will not solve for the Laplacian directly but interpolate linear coefficients to derive alpha. Require that active_levels_num <= levels_num. (default: %(default)s)")

    robust_parser = subparsers.add_parser("robust")
    robust_parser.add_argument("-i", "--image-path", type=str, help="Path to original image", required=True)
    robust_parser.add_argument("-t", "--trimap-path", type=str, help="Path to trimap")

    args = parser.parse_args()
    if args.verbose == 1:
        logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%y%m%d %H%M%S", level=logging.INFO)
    elif args.verbose > 1:
        logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%y%m%d %H%M%S", level=logging.DEBUG)

    if args.subcommand == "closedform":
        logging.info(f"Opening {args.image_path}")
        I = skimage.io.imread(args.image_path) / 255
        I = closedform.utils.ensure_3d_image(I)  # H x W x C array

        logging.info(f"Opening {args.scribble_path}")
        I_scribble = skimage.io.imread(args.scribble_path) / 255 # this is mI in MATLAB code
        assert I_scribble.shape == I.shape, f"Shapes of scribble image ({I_scribble.shape}) and image ({I.shape}) don't match"
        I_scribble_gray2D = closedform.utils.matlab_compatible_rgb2gray(I_scribble) if I.shape[2] == 3 else I_scribble  # H x W array
        I_scribble = closedform.utils.ensure_3d_image(I_scribble)  # H x W x C array

        constrained_map = (np.sum(np.absolute(I - I_scribble), axis=2) > 1e-3).astype(float)  # this is consts_map in MATLAB; H x W array
        constrained_vals = I_scribble_gray2D * constrained_map  # NB. * is the element-wise multiply operator; H x W array

        alpha = closedform.coarse_to_fine.solve_alpha_coarse_to_fine(
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
            unknown_map
        )  # H x W array
    else:
        raise NotImplementedError

    alpha = np.clip(alpha, 0, 1)  # there tends to be some alphas that lie outside of [0, 1]...
    skimage.io.imshow(alpha)
    skimage.io.show()
