import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

from .evaluation import evaluation


class mnist_interpolation_func_diffs_pairs:
    def __init__(
        self, eval_inst: evaluation, name: str = "mnist_interpolation_func_diffs_pairs"
    ):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        dataset.test_data()
        samples = pd.DataFrame()

        ctr = 0
        import pdb

        pdb.set_trace()
        # NOTE: dataset.test_labels['label'] will have to be replaced (as it is now a series)
        for label in dataset.test_labels["label"].unique():
            sample_index = (
                dataset.test_labels[dataset.test_labels["label"] == label]
                .sample(1)
                .index
            )
            sample = dataset.test_data().iloc[sample_index]
            samples = pd.concat([samples, sample], axis=0)

        for digit1, digit2 in list(combinations(list(samples.index), 2)):
            insts = dataset.test_data().loc[[digit1, digit2]]

            label_from = dataset.test_labels.loc[digit1]["label"]
            label_to = dataset.test_labels.loc[digit2]["label"]

            interpolations = interpolate(insts, 1000)
            interp_func_pairs = algorithm.assign_lin_subfcts(
                algorithm.module, interpolations
            )
            interp_func_df = pd.DataFrame(
                interp_func_pairs, columns=["sample", "function"]
            )
            interp_func_df["seq_func_dist"] = 0
            interp_func_df["seq_func_dist_bias"] = 0
            interp_func_df["seq_func_dist_mat"] = 0
            interp_func_df["sign_sum"] = 0
            interp_func_df["sign_sum_weighted"] = 0
            # if does not work: use index
            reconstructions = algorithm.predict(interpolations)
            ctr = 0
            for row_ind in interp_func_df.index:
                # for row_ind in range(0,interp_func_df.shape[0],10):
                if row_ind == 0:
                    ctr += 1
                    continue
                cur = interp_func_df.iloc[row_ind]["function"]
                prev = interp_func_df.iloc[row_ind - 1]["function"]
                dist = cur.dist(prev)
                dist_bias = cur.dist_bias(prev)
                dist_mat = cur.dist_mat(prev)
                dist_sign_sum = cur.dist_sign(prev)
                dist_sign_sum_weighted = cur.dist_sign_weighted(prev)
                interp_func_df.loc[row_ind, "seq_func_dist"] = dist
                interp_func_df.loc[row_ind, "seq_func_dist_bias"] = dist_bias
                interp_func_df.loc[row_ind, "seq_func_dist_mat"] = dist_mat
                interp_func_df.loc[row_ind, "sign_sum"] = dist_sign_sum
                interp_func_df.loc[
                    row_ind, "sign_sum_weighted"
                ] = dist_sign_sum_weighted

                image = interpolations.iloc[row_ind].values
                reconstr = reconstructions.iloc[row_ind].values

                # for image_2, reconstr_2 in zip(interpolations.values, reconstructions.values):
                # import pdb; pdb.set_trace()
                lrp, relevance_bias = algorithm.lrp_ae(
                    algorithm.module, torch.tensor(image)
                )
                lin_func_feature_imp = algorithm.lin_func_feature_imp(
                    algorithm.module, torch.tensor(image)
                )
                lin_func_applied_no_bias = algorithm.apply_lin_func_without_bias(
                    algorithm.module, torch.tensor(image)
                )
                lin_func_bias_imp = algorithm.lin_func_bias_imp(
                    algorithm.module, torch.tensor(image)
                )
                relevance_image = lrp.sum()
                fig, axs = plt.subplots(nrows=1, ncols=6, figsize=[20, 20])
                # figure is given as 1d array
                fig_side = int(np.sqrt(image.shape))
                image_resh = image.reshape(fig_side, fig_side)
                reconstr_resh = reconstr.reshape(fig_side, fig_side)
                lrp_resh = lrp.reshape(fig_side, fig_side)
                lin_func_feature_imp = lin_func_feature_imp.reshape(fig_side, fig_side)
                lin_func_applied_no_bias = lin_func_applied_no_bias.reshape(
                    fig_side, fig_side
                )
                lin_func_bias_imp = lin_func_bias_imp.reshape(fig_side, fig_side)
                axs[0].imshow(image_resh, cmap="gray")
                axs[0].set_title("Orig")
                axs[1].imshow(reconstr_resh, cmap="gray")
                axs[1].set_title("Reconstruction")
                axs[2].imshow(lrp_resh, cmap="gray")
                axs[2].set_title("LRP")
                axs[3].imshow(lin_func_feature_imp, cmap="gray")
                axs[3].set_title("Lin_Func_Feature_Imp")
                axs[4].imshow(lin_func_applied_no_bias, cmap="gray")
                axs[4].set_title("Lin_func_applied_no_bias")
                axs[5].imshow(lin_func_bias_imp, cmap="gray")
                axs[5].set_title("Lin_Func_Bias_Imp")
                fig.suptitle(
                    f"""Relevance of image: {relevance_image}, Relevance
                    of bias: {relevance_bias}, Dist to Prev: {dist}"""
                )
                self.evaluation.save_figure(
                    fig,
                    f"plot_interpolations_lrp_feature_bias_imp_{ctr}",
                    subfolder=f"{label_from}_{label_to}",
                )
                plt.close("all")
                ctr += 1

            fig, axs = plt.subplots(nrows=5, ncols=1, figsize=[20, 20])
            interp_func_df = interp_func_df[interp_func_df["seq_func_dist"] > 0]
            axs[0].plot(
                interp_func_df.index,
                interp_func_df["seq_func_dist"].values,
                color="b",
                label="dist_sum",
            )
            axs[0].legend()
            axs[1].plot(
                interp_func_df.index,
                interp_func_df["seq_func_dist_bias"].values,
                color="r",
                label="dist_bias",
            )
            axs[1].legend()
            axs[2].plot(
                interp_func_df.index,
                interp_func_df["seq_func_dist_mat"].values,
                color="g",
                label="dist_mat",
            )
            axs[2].legend()
            axs[3].plot(
                interp_func_df.index,
                interp_func_df["sign_sum"].values,
                color="m",
                label="sign_sum",
            )
            axs[3].legend()
            axs[4].plot(
                interp_func_df.index,
                interp_func_df["sign_sum_weighted"].values,
                color="y",
                label="sign_sum_weighted",
            )
            axs[4].legend()
            plt.suptitle(
                f"""
                    Total seq_func_dist:
                    {interp_func_df['seq_func_dist'].sum()} \n
                    Variance seq_func_dist {interp_func_df['seq_func_dist'].var()}
                    """
            )
            self.evaluation.save_figure(
                fig, "interp_func_dist_plot", subfolder=f"{label_from}_{label_to}"
            )
            plt.close("all")
        # plot interpolations


def interpolate(insts, res_elems):
    if insts.shape[0] != 2:
        raise Exception("Cannot interpolate between more than two elements")

    inst1 = insts.values[0]
    inst2 = insts.values[1]
    interp_steps = res_elems - 1
    interp_np = inst1
    for i in range(1, interp_steps + 1):
        interp_inst = ((interp_steps - i) / interp_steps) * inst1 + (
            i / interp_steps
        ) * inst2
        interp_np = np.vstack((interp_np, interp_inst))
    result_df = pd.DataFrame(interp_np)
    return result_df
