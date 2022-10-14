import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

from .evaluation import evaluation


class interpolation_func_diffs_pairs:
    def __init__(self, eval_inst: evaluation, name: str = "interpolation_func_diffs"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        dataset.test_data()
        # Code for particular instances
        #        insts = pd.DataFrame()
        #        label_inds = []
        #        for label in dataset.test_labels.unique():
        #            label_ind = dataset.test_labels[dataset.test_labels == label].sample(1).index
        #            label_inds.append(label_ind)
        #            label_inst = dataset.test_data().loc[label_ind]
        #            import pdb; pdb.set_trace()
        #            insts = pd.concat([insts, label_mean], axis=0)
        #        for label_ind1, label_ind2 in list(combinations(label_inds,2)):
        #            label1 = dataset.test_labels[label_ind1].item()
        #            label2 = dataset.test_labels[label_ind2].item()
        #            subfolder = str(label1) + '_' + str(label2)
        #            inst_pair = pd.DataFrame()
        #            inst_pair = pd.concat([inst_pair, insts.loc[label_ind1]])
        #            inst_pair = pd.concat([inst_pair, insts.loc[label_ind2]])
        #            self.evaluate_inst_pair(dataset, algorithm, inst_pair, subfolder)

        # Code for label means
        test_labels = []
        means = []
        for label in dataset.test_labels.unique():
            test_labels.append(label)
            label_data = dataset.test_data().loc[
                dataset.test_labels[dataset.test_labels == label].index
            ]
            label_mean = label_data.mean()
            means.append(label_mean)
        label_mean_tuples = list(zip(test_labels, means))
        for label_mean1, label_mean2 in list(combinations(label_mean_tuples, 2)):
            label1 = label_mean1[0]
            label2 = label_mean2[0]
            mean1 = pd.DataFrame(label_mean1[1]).transpose()
            mean2 = pd.DataFrame(label_mean2[1]).transpose()
            subfolder = str(label1) + "_" + str(label2)
            inst_pair = pd.DataFrame()
            inst_pair = pd.concat([inst_pair, mean1])
            inst_pair = pd.concat([inst_pair, mean2])
            self.evaluate_inst_pair(dataset, algorithm, inst_pair, subfolder)

    def evaluate_inst_pair(self, dataset, algorithm, insts, subfolder):
        # insts = dataset.test_data().sample(2)
        # do for each pair of in insts
        # create folder for it as well?!
        interpolations = interpolate(insts, 30)
        interp_func_pairs = algorithm.assign_lin_subfcts(
            algorithm.module, interpolations
        )
        interp_func_df = pd.DataFrame(interp_func_pairs, columns=["sample", "function"])
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
            interp_func_df.loc[row_ind, "sign_sum_weighted"] = dist_sign_sum_weighted

            orig = interpolations.iloc[row_ind].values
            reconstr = reconstructions.iloc[row_ind].values

            # for image_2, reconstr_2 in zip(interpolations.values, reconstructions.values):
            lrp, relevance_bias = algorithm.lrp_ae(algorithm.module, torch.tensor(orig))
            lin_func_feature_imp = algorithm.lin_func_feature_imp(
                algorithm.module, torch.tensor(orig)
            )
            lin_func_applied_no_bias = algorithm.apply_lin_func_without_bias(
                algorithm.module, torch.tensor(orig)
            )
            lin_func_bias_imp = algorithm.lin_func_bias_imp(
                algorithm.module, torch.tensor(orig)
            )
            relevance_orig = lrp.sum()
            fig, axs = plt.subplots(nrows=4, ncols=1, figsize=[20, 14], sharey=True)
            # figure is given as 1d array
            # fig_side = int(np.sqrt(orig.shape))
            # image_resh = image.reshape(fig_side,fig_side)
            # reconstr_resh = reconstr.reshape(fig_side,fig_side)
            # lrp_resh = lrp.reshape(fig_side,fig_side)
            # lin_func_feature_imp = lin_func_feature_imp.reshape(fig_side,fig_side)
            # lin_func_applied_no_bias = lin_func_applied_no_bias.reshape(
            # fig_side, fig_side)
            # lin_func_bias_imp = lin_func_bias_imp.reshape(fig_side,fig_side)
            axs[0].plot(orig, label="Orig", color="blue")
            axs[0].plot(reconstr, label="Reconstruction", color="red")
            axs[0].set_title("Orig-Recon")
            axs[0].legend()
            # axs[1].plot(reconstr)
            # axs[1].set_title('Reconstruction')
            axs[1].plot(reconstr, label="Reconstruction", color="red")
            axs[1].plot(lin_func_applied_no_bias, label="lin_func_applied_without_bias")
            axs[1].plot(lin_func_bias_imp, label="Lin_func_bias")
            axs[1].axhline(color="grey", alpha=0.3)
            axs[1].set_title("Reconstr-appliedMatrix-bias")
            axs[1].legend()
            axs[2].plot(lrp)
            axs[2].set_title("LRP")
            axs[3].plot(lin_func_feature_imp)
            axs[3].set_title("Lin_Func_Feature_Imp")
            # axs[4].plot(lin_func_applied_no_bias)
            # axs[4].set_title('Lin_func_applied_no_bias')
            # axs[5].plot(lin_func_bias_imp)
            # axs[5].set_title('Lin_Func_Bias_Imp')
            fig.suptitle(
                f"""Relevance of Orig: {relevance_orig}, Relevance
                of bias: {relevance_bias}, Dist to Prev: {dist}"""
            )
            self.evaluation.save_figure(
                fig,
                f"plot_interpolations_lrp_feature_bias_imp_{ctr}",
                subfolder=subfolder,
            )
            res_df = pd.DataFrame(
                np.stack([orig, reconstr]).transpose(), columns=["orig", "recon"]
            )

            self.evaluation.save_csv(
                res_df, f"plot_interpolations_{ctr}", subfolder=subfolder
            )
            plt.close("all")
            ctr += 1

        fig = plt.figure(figsize=[20, 20])
        interp_func_df = interp_func_df[interp_func_df["seq_func_dist"] > 0]
        plt.plot(
            interp_func_df.index,
            interp_func_df["seq_func_dist"].values,
            color="b",
            label="dist_sum",
        )
        # import pdb; pdb.set_trace()
        plt.gca().set_xticks(list(interp_func_df.index))
        # plt.ylim(bottom=0)
        # plt.plot(interp_func_df.index,
        # interp_func_df['seq_func_dist_bias'].values, color = 'r',
        # label='dist_bias')
        # plt.plot(interp_func_df.index, interp_func_df['seq_func_dist_mat'].values, color =
        #'g', label='dist_mat')
        # plt.plot(interp_func_df.index, interp_func_df['sign_sum'].values, color =
        #'m', label='sign_sum')
        # plt.plot(interp_func_df.index, interp_func_df['sign_sum_weighted'].values, color =
        #'y', label='sign_sum_weighted')
        plt.legend()
        self.evaluation.save_figure(fig, "interp_func_dist_plot", subfolder=subfolder)
        self.evaluation.save_csv(
            interp_func_df["seq_func_dist"], "interp_func_dist", subfolder=subfolder
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
