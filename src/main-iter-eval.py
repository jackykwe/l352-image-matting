import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from tqdm.auto import tqdm

import utils

if __name__ == "__main__":
    OUTPUT_ROOTDIR = "output"
    LOWRES_IMAGE_DIR = "../datasets/input_training_lowres"
    LOWRES_TRIMAP_DIR = "../datasets/trimap_training_lowres/Trimap2"
    LOWRES_GROUNDTRUTH_DIR = "../datasets/gt_training_lowres"
    # HIGHRES_IMAGE_DIR = "../datasets/input_training_highres"
    # HIGHRES_TRIMAP_DIR = "../datasets/trimap_training_highres/Trimap2"
    # HIGHRES_GROUNDTRUTH_DIR = "../datasets/gt_training_highres"

    SCHEME_TAGS_TO_LEGEND = {
        "lowres-closedform": "Closed-form",
        "lowres-robust-global_random_same": "Robust: global_random_same",
        "lowres-robust-global_random": "Robust: global_random",
        "lowres-robust-local_random-nearest_candidates_count50": "Robust: local_random (nearest 50)",
        "lowres-robust-local_random-nearest_candidates_count100": "Robust: local_random (nearest 100)",
        "lowres-robust-local_random-nearest_candidates_count200": "Robust: local_random (nearest 200)",
        "lowres-robust-local_random-nearest_candidates_count500": "Robust: local_random (nearest 500)",
        "lowres-robust-local_random-nearest_candidates_count1000": "Robust: local_random (nearest 1000)",
        "lowres-robust-nearest": "Robust: nearest",
        "lowres-robust-global_spread": "Robust: global_spread",
        "lowres-robust-local_spread-nearest_candidates_count50": "Robust: local_spread (nearest 50)",
        "lowres-robust-local_spread-nearest_candidates_count100": "Robust: local_spread (nearest 100)",
        "lowres-robust-local_spread-nearest_candidates_count200": "Robust: local_spread (nearest 200)",
        "lowres-robust-local_spread-nearest_candidates_count500": "Robust: local_spread (nearest 500)",
        "lowres-robust-local_spread-nearest_candidates_count1000": "Robust: local_spread (nearest 1000)",
        "lowres-aematter": "AEMatter"
    }
    X_AXIS_TO_LEGEND = {
        "#U": "#U",
        "#U/HW": "#U/HW",
        "FAGS": "Foreground Average Gradient Strength",
        "BAGS": "Background Average Gradient Strength",
        "UAGS": "Unknown Average Gradient Strength",
        "AGS": "Average Gradient Strength"
    }
    Y_AXIS_TO_LEGEND = {
        "time_taken": "Time Taken [s]",
        "SAE": "Sum of Absolute Error",
        "MSE": "Mean Squared Error",
        "PSNR": "Peak Signal-to-Noise Ratio [dB]"
    }
    X_AXIS_TO_FILENAMESTR = {
        "#U": "U",
        "#U/HW": "UdivHW",
        "FAGS": "FAGS",
        "BAGS": "BAGS",
        "UAGS": "UAGS",
        "AGS": "AGS"
    }
    Y_AXIS_TO_FILENAMESTR = {
        "time_taken": "t",
        "SAE": "SAE",
        "MSE": "MSE",
        "PSNR": "PSNR"
    }
    ####################
    # IMAGE STATISTICS #
    ####################
    rows = []
    for image_name in tqdm(os.listdir(LOWRES_IMAGE_DIR), desc="Tabulating image statistics"):
        image_path = os.path.join(LOWRES_IMAGE_DIR, image_name)
        groundtruth_path = os.path.join(LOWRES_GROUNDTRUTH_DIR, image_name)

        I = skimage.io.imread(image_path)
        I_greyscale = utils.matlab_compatible_rgb2gray(I)
        H, W = I_greyscale.shape

        alpha_groundtruth = skimage.io.imread(groundtruth_path)
        assert np.all(np.equal(alpha_groundtruth[:, :, 0], alpha_groundtruth[:, :, 1]))
        assert np.all(np.equal(alpha_groundtruth[:, :, 0], alpha_groundtruth[:, :, 2]))
        alpha_groundtruth = alpha_groundtruth[:, :, 0].astype(int)  # H x W array
        assert I_greyscale.shape == alpha_groundtruth.shape

        horizontal_gradients = I_greyscale[:-1, 1:] - I_greyscale[:-1, :-1]  # (H - 1) x (W - 1) image
        vertical_gradients = I_greyscale[1:, :-1] - I_greyscale[:-1, :-1]  # (H - 1) x (W - 1) image
        gradient_strengths = np.sqrt(np.power(horizontal_gradients, 2) + np.power(vertical_gradients, 2))

        foreground_average_gradient_strength = np.mean(gradient_strengths[alpha_groundtruth[:-1, :-1] == 1])
        background_average_gradient_strength = np.mean(gradient_strengths[alpha_groundtruth[:-1, :-1] == 0])
        unknown_average_gradient_strength = np.mean(gradient_strengths[(alpha_groundtruth[:-1, :-1] != 1) & (alpha_groundtruth[:-1, :-1] != 0)])
        average_gradient_strength = np.mean(gradient_strengths)
        known_count = np.sum(alpha_groundtruth == 255) + np.sum(alpha_groundtruth == 0)
        unknown_count = H * W - known_count

        rows.append([
            image_name,
            H,
            W,
            unknown_count,
            known_count,
            unknown_count / H * W,
            foreground_average_gradient_strength,
            background_average_gradient_strength,
            unknown_average_gradient_strength,
            average_gradient_strength
        ])
    df_image_stats = pd.DataFrame(rows, columns=["image_name", "H", "W", "#U", "#K", "#U/HW", "FAGS", "BAGS", "UAGS", "AGS"])

    scheme_tag_to_df_scheme = {}
    for scheme_tag in tqdm(SCHEME_TAGS_TO_LEGEND, desc="Tabulating algorithms' results"):
        output_dir = os.path.join(OUTPUT_ROOTDIR, scheme_tag)
        timings_file_path = os.path.join(OUTPUT_ROOTDIR, f"{scheme_tag}-timings.txt")
        df_scheme = pd.read_csv(timings_file_path).sort_values("image_name")
        df_scheme["time_taken"] /= 1e9  # convert to seconds

        # H, W, #U, #K, #U/HW,
        rows = []
        for image_name in os.listdir(LOWRES_IMAGE_DIR):
            groundtruth_path = os.path.join(LOWRES_GROUNDTRUTH_DIR, image_name)
            output_path = os.path.join(output_dir, image_name)

            alpha_inferred = skimage.io.imread(output_path).astype(int)  # H x W array
            alpha_groundtruth = skimage.io.imread(groundtruth_path)
            assert np.all(np.equal(alpha_groundtruth[:, :, 0], alpha_groundtruth[:, :, 1]))
            assert np.all(np.equal(alpha_groundtruth[:, :, 0], alpha_groundtruth[:, :, 2]))
            alpha_groundtruth = alpha_groundtruth[:, :, 0].astype(int)  # H x W array
            assert alpha_inferred.shape == alpha_groundtruth.shape

            sae = np.sum(np.absolute(alpha_groundtruth - alpha_inferred))
            mse = np.mean(np.power(alpha_groundtruth - alpha_inferred, 2))
            psnr = 20 * math.log10(255) - 10 * math.log10(mse)
            rows.append([
                image_name,
                sae,
                mse,
                psnr
            ])
        scheme_tag_to_df_scheme[scheme_tag] = df_scheme\
            .merge(pd.DataFrame(rows, columns=["image_name", "SAE", "MSE", "PSNR"]), on="image_name")\
            .merge(df_image_stats, on="image_name")
        # if scheme_tag == "lowres-robust-global_random_same":
        #     print(scheme_tag_to_df_scheme[scheme_tag])
        #     break

    print(scheme_tag_to_df_scheme["lowres-robust-local_spread-nearest_candidates_count50"].loc[:, ["image_name", "SAE", "MSE", "PSNR", "time_taken"]])
    # raise

    statistic_table = pd.DataFrame.from_dict(
        {scheme_tag: df_scheme.loc[:, ["SAE", "MSE", "PSNR", "time_taken"]].mean() for scheme_tag, df_scheme in scheme_tag_to_df_scheme.items()},
        orient="index"
    )
    statistic_table.index = pd.Index([SCHEME_TAGS_TO_LEGEND[i] for i in statistic_table.index])
    statistic_table.to_csv(os.path.join(OUTPUT_ROOTDIR, "statistic_table.csv"))

    for y_metric in tqdm(Y_AXIS_TO_LEGEND, desc="Saving figures"):
        for x_metric in X_AXIS_TO_LEGEND:
            fig, ax = plt.subplots(figsize=(8, 5))
            for scheme_tag in SCHEME_TAGS_TO_LEGEND:
                df_scheme = scheme_tag_to_df_scheme[scheme_tag].sort_values(x_metric)
                ax.plot(df_scheme[x_metric], df_scheme[y_metric], label=SCHEME_TAGS_TO_LEGEND[scheme_tag])
                # if scheme_tag == "lowres-robust-global_random_same":
                #     break
            ax.legend()
            ax.set_xlabel(X_AXIS_TO_LEGEND[x_metric])
            ax.set_ylabel(Y_AXIS_TO_LEGEND[y_metric])
            # plt.show()
            plt.savefig(
                os.path.join("../report/imgs", f"{Y_AXIS_TO_FILENAMESTR[y_metric]}-against-{X_AXIS_TO_FILENAMESTR[x_metric]}.png"),
                pad_inches=0
            )
