import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd

from .evaluation import evaluation

# for images


class fct_change_by_border_dist_qual:
    def __init__(
        self, eval_inst: evaluation, name: str = "fct_change_by_border_dist_qual"
    ):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        sample_dist_pairs = algorithm.assign_most_far_border_dists(
            algorithm.module, dataset.test_data()
        )
        sample_dist_df = pd.DataFrame(sample_dist_pairs, columns=["sample", "dist"])
        df_sorted = sample_dist_df.sort_values(by=["dist"], ascending=False)
        df_sorted["new_index"] = range(sample_dist_df.shape[0])
        df_sorted.set_index("new_index", inplace=True)

        largest_dist_df = df_sorted.iloc[:10]
        mid_dist_df_ind = int(df_sorted.shape[0] / 2)
        mid_dist_df = df_sorted.iloc[mid_dist_df_ind - 5 : mid_dist_df_ind + 5]
        smallest_dist_df = df_sorted.iloc[-10:]
        ctr = 0
        for df_entry in largest_dist_df.values:
            image = df_entry[0]
            dist_value = df_entry[1]
            fig_side = int(np.sqrt(image.shape))
            fig, axs = plt.subplots(nrows=3, ncols=6, figsize=[20, 20])

            after_cross, after_cross_fct = algorithm.get_most_far_afterCross_fct(
                algorithm.module, image
            )

            reconstruction = np.array(
                algorithm.predict(pd.DataFrame(image).transpose())
            )
            image_np = np.array(image)
            fig.suptitle(f"""Dist_value is {dist_value}""")
            lrp, relevance_bias = algorithm.lrp_ae(algorithm.module, image)
            lin_func_feature_imp = algorithm.lin_func_feature_imp(
                algorithm.module, image
            )
            lin_func_applied_no_bias = algorithm.apply_lin_func_without_bias(
                algorithm.module, image
            )
            lin_func_bias_imp = algorithm.lin_func_bias_imp(algorithm.module, image)
            image_resh = image_np.reshape(fig_side, fig_side)
            reconstr_resh = reconstruction.reshape(fig_side, fig_side)
            lrp_resh = lrp.reshape(fig_side, fig_side)
            lin_func_feature_imp = lin_func_feature_imp.reshape(fig_side, fig_side)
            lin_func_applied_no_bias = lin_func_applied_no_bias.reshape(
                fig_side, fig_side
            )
            lin_func_bias_imp = lin_func_bias_imp.reshape(fig_side, fig_side)

            ac_recon = np.array(
                algorithm.predict(pd.DataFrame(after_cross).transpose())
            )
            ac_np = np.array(after_cross)
            ac_lrp, ac_relevance_bias = algorithm.lrp_ae(algorithm.module, after_cross)
            ac_lin_func_feature_imp = algorithm.lin_func_feature_imp(
                algorithm.module, after_cross
            )
            ac_lin_func_applied_no_bias = algorithm.apply_lin_func_without_bias(
                algorithm.module, after_cross
            )
            ac_lin_func_bias_imp = algorithm.lin_func_bias_imp(
                algorithm.module, after_cross
            )
            ac_resh = ac_np.reshape(fig_side, fig_side)
            ac_reconstr_resh = ac_recon.reshape(fig_side, fig_side)
            ac_lrp_resh = ac_lrp.reshape(fig_side, fig_side)
            ac_lin_func_feature_imp = ac_lin_func_feature_imp.reshape(
                fig_side, fig_side
            )
            ac_lin_func_applied_no_bias = ac_lin_func_applied_no_bias.reshape(
                fig_side, fig_side
            )
            ac_lin_func_bias_imp = ac_lin_func_bias_imp.reshape(fig_side, fig_side)

            orig_ac_diff = image_resh - ac_resh
            recon_diff = reconstr_resh - ac_reconstr_resh
            lrp_diff = lrp_resh - ac_lrp_resh
            lin_fct_diff = lin_func_feature_imp - ac_lin_func_feature_imp
            lin_fct_no_bias_diff = (
                lin_func_applied_no_bias - ac_lin_func_applied_no_bias
            )
            bias_fct_diff = lin_func_bias_imp - ac_lin_func_bias_imp

            vmin_ac_orig = min(image_resh.min(), ac_resh.min())
            vmax_ac_orig = max(image_resh.max(), ac_resh.max())

            vmin_recon = min(reconstr_resh.min(), ac_reconstr_resh.min())
            vmax_recon = max(reconstr_resh.max(), ac_reconstr_resh.max())

            vmin_lrp = min(lrp_resh.min(), ac_lrp_resh.min())
            vmax_lrp = max(lrp_resh.max(), ac_lrp_resh.max())

            vmin_fct_imp = min(
                lin_func_feature_imp.min(), ac_lin_func_feature_imp.min()
            )
            vmax_fct_imp = max(
                lin_func_feature_imp.max(), ac_lin_func_feature_imp.max()
            )

            vmin_fct_applied_no_bias = min(
                lin_func_applied_no_bias.min(), ac_lin_func_applied_no_bias.min()
            )
            vmax_fct_applied_no_bias = max(
                lin_func_applied_no_bias.max(), ac_lin_func_applied_no_bias.max()
            )

            vmin_bias = min(lin_func_bias_imp.min(), ac_lin_func_bias_imp.min())
            vmax_bias = max(lin_func_bias_imp.max(), ac_lin_func_bias_imp.max())

            axs[0, 0].imshow(
                image_resh, cmap="gray", vmin=vmin_ac_orig, vmax=vmax_ac_orig
            )
            axs[0, 0].set_title("Orig")
            axs[0, 1].imshow(
                reconstr_resh, cmap="gray", vmin=vmin_recon, vmax=vmax_recon
            )
            axs[0, 1].set_title("Reconstruction")
            axs[0, 2].imshow(lrp_resh, cmap="gray", vmin=vmin_lrp, vmax=vmax_lrp)
            axs[0, 2].set_title("LRP")
            axs[0, 3].imshow(
                lin_func_feature_imp, cmap="gray", vmin=vmin_fct_imp, vmax=vmax_fct_imp
            )
            axs[0, 3].set_title("Lin_Func_Feature_Imp")
            axs[0, 4].imshow(
                lin_func_applied_no_bias,
                cmap="gray",
                vmin=vmin_fct_applied_no_bias,
                vmax=vmax_fct_applied_no_bias,
            )
            axs[0, 4].set_title("Lin_func_applied_no_bias")
            axs[0, 5].imshow(
                lin_func_bias_imp, cmap="gray", vmin=vmin_bias, vmax=vmax_bias
            )
            axs[0, 5].set_title("Lin_Func_Bias_Imp")
            axs[1, 0].imshow(ac_resh, cmap="gray", vmin=vmin_ac_orig, vmax=vmax_ac_orig)
            axs[1, 0].set_title("After_cross")
            axs[1, 1].imshow(
                ac_reconstr_resh, cmap="gray", vmin=vmin_recon, vmax=vmax_recon
            )
            axs[1, 1].set_title("AC_Reconstruction")
            axs[1, 2].imshow(ac_lrp_resh, cmap="gray", vmin=vmin_lrp, vmax=vmax_lrp)
            axs[1, 2].set_title("AC_LRP")
            axs[1, 3].imshow(
                ac_lin_func_feature_imp,
                cmap="gray",
                vmin=vmin_fct_imp,
                vmax=vmax_fct_imp,
            )
            axs[1, 3].set_title("AC_Lin_Func_Feature_Imp")
            axs[1, 4].imshow(
                ac_lin_func_applied_no_bias,
                cmap="gray",
                vmin=vmin_fct_applied_no_bias,
                vmax=vmax_fct_applied_no_bias,
            )
            axs[1, 4].set_title("AC_Lin_func_applied_no_bias")
            axs[1, 5].imshow(
                ac_lin_func_bias_imp, cmap="gray", vmin=vmin_bias, vmax=vmax_bias
            )
            axs[1, 5].set_title("AC_Lin_Func_Bias_Imp")
            axs[2, 0].imshow(
                orig_ac_diff, cmap="gray", vmin=vmin_ac_orig, vmax=vmax_ac_orig
            )
            axs[2, 0].set_title("orig_ac_diff")
            axs[2, 1].imshow(recon_diff, cmap="gray", vmin=vmin_recon, vmax=vmax_recon)
            axs[2, 1].set_title("recon_diff")
            axs[2, 2].imshow(lrp_diff, cmap="gray", vmin=vmin_lrp, vmax=vmax_lrp)
            axs[2, 2].set_title("lrp_diff")
            axs[2, 3].imshow(
                lin_fct_diff, cmap="gray", vmin=vmin_fct_imp, vmax=vmax_fct_imp
            )
            axs[2, 3].set_title("lin_fct_diff")
            axs[2, 4].imshow(
                lin_fct_no_bias_diff,
                cmap="gray",
                vmin=vmin_fct_applied_no_bias,
                vmax=vmax_fct_applied_no_bias,
            )
            axs[2, 4].set_title("AC_Lin_func_applied_no_bias_diff")
            axs[2, 5].imshow(bias_fct_diff, cmap="gray", vmin=vmin_bias, vmax=vmax_bias)
            axs[2, 5].set_title("bias_fct_diff")
            self.evaluation.save_figure(
                fig, f"plot_mnist_sample_largest_dist_withAC{ctr}"
            )
            plt.close("all")
            ctr += 1

        ctr = 0
        for df_entry in mid_dist_df.values:
            image = df_entry[0]
            dist_value = df_entry[1]
            fig_side = int(np.sqrt(image.shape))
            fig, axs = plt.subplots(nrows=3, ncols=6, figsize=[20, 20])

            after_cross, after_cross_fct = algorithm.get_most_far_afterCross_fct(
                algorithm.module, image
            )

            reconstruction = np.array(
                algorithm.predict(pd.DataFrame(image).transpose())
            )
            image_np = np.array(image)
            fig.suptitle(f"""Dist_value is {dist_value}""")
            lrp, relevance_bias = algorithm.lrp_ae(algorithm.module, image)
            lin_func_feature_imp = algorithm.lin_func_feature_imp(
                algorithm.module, image
            )
            lin_func_applied_no_bias = algorithm.apply_lin_func_without_bias(
                algorithm.module, image
            )
            lin_func_bias_imp = algorithm.lin_func_bias_imp(algorithm.module, image)
            image_resh = image_np.reshape(fig_side, fig_side)
            reconstr_resh = reconstruction.reshape(fig_side, fig_side)
            lrp_resh = lrp.reshape(fig_side, fig_side)
            lin_func_feature_imp = lin_func_feature_imp.reshape(fig_side, fig_side)
            lin_func_applied_no_bias = lin_func_applied_no_bias.reshape(
                fig_side, fig_side
            )
            lin_func_bias_imp = lin_func_bias_imp.reshape(fig_side, fig_side)

            ac_recon = np.array(
                algorithm.predict(pd.DataFrame(after_cross).transpose())
            )
            ac_np = np.array(after_cross)
            ac_lrp, ac_relevance_bias = algorithm.lrp_ae(algorithm.module, after_cross)
            ac_lin_func_feature_imp = algorithm.lin_func_feature_imp(
                algorithm.module, after_cross
            )
            ac_lin_func_applied_no_bias = algorithm.apply_lin_func_without_bias(
                algorithm.module, after_cross
            )
            ac_lin_func_bias_imp = algorithm.lin_func_bias_imp(
                algorithm.module, after_cross
            )
            ac_resh = ac_np.reshape(fig_side, fig_side)
            ac_reconstr_resh = ac_recon.reshape(fig_side, fig_side)
            ac_lrp_resh = ac_lrp.reshape(fig_side, fig_side)
            ac_lin_func_feature_imp = ac_lin_func_feature_imp.reshape(
                fig_side, fig_side
            )
            ac_lin_func_applied_no_bias = ac_lin_func_applied_no_bias.reshape(
                fig_side, fig_side
            )
            ac_lin_func_bias_imp = ac_lin_func_bias_imp.reshape(fig_side, fig_side)

            orig_ac_diff = image_resh - ac_resh
            recon_diff = reconstr_resh - ac_reconstr_resh
            lrp_diff = lrp_resh - ac_lrp_resh
            lin_fct_diff = lin_func_feature_imp - ac_lin_func_feature_imp
            lin_fct_no_bias_diff = (
                lin_func_applied_no_bias - ac_lin_func_applied_no_bias
            )
            bias_fct_diff = lin_func_bias_imp - ac_lin_func_bias_imp

            vmin_ac_orig = min(image_resh.min(), ac_resh.min())
            vmax_ac_orig = max(image_resh.max(), ac_resh.max())

            vmin_recon = min(reconstr_resh.min(), ac_reconstr_resh.min())
            vmax_recon = max(reconstr_resh.max(), ac_reconstr_resh.max())

            vmin_lrp = min(lrp_resh.min(), ac_lrp_resh.min())
            vmax_lrp = max(lrp_resh.max(), ac_lrp_resh.max())

            vmin_fct_imp = min(
                lin_func_feature_imp.min(), ac_lin_func_feature_imp.min()
            )
            vmax_fct_imp = max(
                lin_func_feature_imp.max(), ac_lin_func_feature_imp.max()
            )

            vmin_fct_applied_no_bias = min(
                lin_func_applied_no_bias.min(), ac_lin_func_applied_no_bias.min()
            )
            vmax_fct_applied_no_bias = max(
                lin_func_applied_no_bias.max(), ac_lin_func_applied_no_bias.max()
            )

            vmin_bias = min(lin_func_bias_imp.min(), ac_lin_func_bias_imp.min())
            vmax_bias = max(lin_func_bias_imp.max(), ac_lin_func_bias_imp.max())

            axs[0, 0].imshow(
                image_resh, cmap="gray", vmin=vmin_ac_orig, vmax=vmax_ac_orig
            )
            axs[0, 0].set_title("Orig")
            axs[0, 1].imshow(
                reconstr_resh, cmap="gray", vmin=vmin_recon, vmax=vmax_recon
            )
            axs[0, 1].set_title("Reconstruction")
            axs[0, 2].imshow(lrp_resh, cmap="gray", vmin=vmin_lrp, vmax=vmax_lrp)
            axs[0, 2].set_title("LRP")
            axs[0, 3].imshow(
                lin_func_feature_imp, cmap="gray", vmin=vmin_fct_imp, vmax=vmax_fct_imp
            )
            axs[0, 3].set_title("Lin_Func_Feature_Imp")
            axs[0, 4].imshow(
                lin_func_applied_no_bias,
                cmap="gray",
                vmin=vmin_fct_applied_no_bias,
                vmax=vmax_fct_applied_no_bias,
            )
            axs[0, 4].set_title("Lin_func_applied_no_bias")
            axs[0, 5].imshow(
                lin_func_bias_imp, cmap="gray", vmin=vmin_bias, vmax=vmax_bias
            )
            axs[0, 5].set_title("Lin_Func_Bias_Imp")
            axs[1, 0].imshow(ac_resh, cmap="gray", vmin=vmin_ac_orig, vmax=vmax_ac_orig)
            axs[1, 0].set_title("After_cross")
            axs[1, 1].imshow(
                ac_reconstr_resh, cmap="gray", vmin=vmin_recon, vmax=vmax_recon
            )
            axs[1, 1].set_title("AC_Reconstruction")
            axs[1, 2].imshow(ac_lrp_resh, cmap="gray", vmin=vmin_lrp, vmax=vmax_lrp)
            axs[1, 2].set_title("AC_LRP")
            axs[1, 3].imshow(
                ac_lin_func_feature_imp,
                cmap="gray",
                vmin=vmin_fct_imp,
                vmax=vmax_fct_imp,
            )
            axs[1, 3].set_title("AC_Lin_Func_Feature_Imp")
            axs[1, 4].imshow(
                ac_lin_func_applied_no_bias,
                cmap="gray",
                vmin=vmin_fct_applied_no_bias,
                vmax=vmax_fct_applied_no_bias,
            )
            axs[1, 4].set_title("AC_Lin_func_applied_no_bias")
            axs[1, 5].imshow(
                ac_lin_func_bias_imp, cmap="gray", vmin=vmin_bias, vmax=vmax_bias
            )
            axs[1, 5].set_title("AC_Lin_Func_Bias_Imp")
            axs[2, 0].imshow(
                orig_ac_diff, cmap="gray", vmin=vmin_ac_orig, vmax=vmax_ac_orig
            )
            axs[2, 0].set_title("orig_ac_diff")
            axs[2, 1].imshow(recon_diff, cmap="gray", vmin=vmin_recon, vmax=vmax_recon)
            axs[2, 1].set_title("recon_diff")
            axs[2, 2].imshow(lrp_diff, cmap="gray", vmin=vmin_lrp, vmax=vmax_lrp)
            axs[2, 2].set_title("lrp_diff")
            axs[2, 3].imshow(
                lin_fct_diff, cmap="gray", vmin=vmin_fct_imp, vmax=vmax_fct_imp
            )
            axs[2, 3].set_title("lin_fct_diff")
            axs[2, 4].imshow(
                lin_fct_no_bias_diff,
                cmap="gray",
                vmin=vmin_fct_applied_no_bias,
                vmax=vmax_fct_applied_no_bias,
            )
            axs[2, 4].set_title("AC_Lin_func_applied_no_bias_diff")
            axs[2, 5].imshow(bias_fct_diff, cmap="gray", vmin=vmin_bias, vmax=vmax_bias)
            axs[2, 5].set_title("bias_fct_diff")
            self.evaluation.save_figure(fig, f"plot_mnist_sample_mid_dist_withAC{ctr}")
            plt.close("all")
            ctr += 1

        ctr = 0
        for df_entry in smallest_dist_df.values:
            image = df_entry[0]
            dist_value = df_entry[1]
            fig_side = int(np.sqrt(image.shape))
            fig, axs = plt.subplots(nrows=3, ncols=6, figsize=[20, 20])

            after_cross, after_cross_fct = algorithm.get_most_far_afterCross_fct(
                algorithm.module, image
            )

            reconstruction = np.array(
                algorithm.predict(pd.DataFrame(image).transpose())
            )
            image_np = np.array(image)
            fig.suptitle(f"""Dist_value is {dist_value}""")
            lrp, relevance_bias = algorithm.lrp_ae(algorithm.module, image)
            lin_func_feature_imp = algorithm.lin_func_feature_imp(
                algorithm.module, image
            )
            lin_func_applied_no_bias = algorithm.apply_lin_func_without_bias(
                algorithm.module, image
            )
            lin_func_bias_imp = algorithm.lin_func_bias_imp(algorithm.module, image)
            image_resh = image_np.reshape(fig_side, fig_side)
            reconstr_resh = reconstruction.reshape(fig_side, fig_side)
            lrp_resh = lrp.reshape(fig_side, fig_side)
            lin_func_feature_imp = lin_func_feature_imp.reshape(fig_side, fig_side)
            lin_func_applied_no_bias = lin_func_applied_no_bias.reshape(
                fig_side, fig_side
            )
            lin_func_bias_imp = lin_func_bias_imp.reshape(fig_side, fig_side)

            ac_recon = np.array(
                algorithm.predict(pd.DataFrame(after_cross).transpose())
            )
            ac_np = np.array(after_cross)
            ac_lrp, ac_relevance_bias = algorithm.lrp_ae(algorithm.module, after_cross)
            ac_lin_func_feature_imp = algorithm.lin_func_feature_imp(
                algorithm.module, after_cross
            )
            ac_lin_func_applied_no_bias = algorithm.apply_lin_func_without_bias(
                algorithm.module, after_cross
            )
            ac_lin_func_bias_imp = algorithm.lin_func_bias_imp(
                algorithm.module, after_cross
            )
            ac_resh = ac_np.reshape(fig_side, fig_side)
            ac_reconstr_resh = ac_recon.reshape(fig_side, fig_side)
            ac_lrp_resh = ac_lrp.reshape(fig_side, fig_side)
            ac_lin_func_feature_imp = ac_lin_func_feature_imp.reshape(
                fig_side, fig_side
            )
            ac_lin_func_applied_no_bias = ac_lin_func_applied_no_bias.reshape(
                fig_side, fig_side
            )
            ac_lin_func_bias_imp = ac_lin_func_bias_imp.reshape(fig_side, fig_side)

            orig_ac_diff = image_resh - ac_resh
            recon_diff = reconstr_resh - ac_reconstr_resh
            lrp_diff = lrp_resh - ac_lrp_resh
            lin_fct_diff = lin_func_feature_imp - ac_lin_func_feature_imp
            lin_fct_no_bias_diff = (
                lin_func_applied_no_bias - ac_lin_func_applied_no_bias
            )
            bias_fct_diff = lin_func_bias_imp - ac_lin_func_bias_imp

            vmin_ac_orig = min(image_resh.min(), ac_resh.min())
            vmax_ac_orig = max(image_resh.max(), ac_resh.max())

            vmin_recon = min(reconstr_resh.min(), ac_reconstr_resh.min())
            vmax_recon = max(reconstr_resh.max(), ac_reconstr_resh.max())

            vmin_lrp = min(lrp_resh.min(), ac_lrp_resh.min())
            vmax_lrp = max(lrp_resh.max(), ac_lrp_resh.max())

            vmin_fct_imp = min(
                lin_func_feature_imp.min(), ac_lin_func_feature_imp.min()
            )
            vmax_fct_imp = max(
                lin_func_feature_imp.max(), ac_lin_func_feature_imp.max()
            )

            vmin_fct_applied_no_bias = min(
                lin_func_applied_no_bias.min(), ac_lin_func_applied_no_bias.min()
            )
            vmax_fct_applied_no_bias = max(
                lin_func_applied_no_bias.max(), ac_lin_func_applied_no_bias.max()
            )

            vmin_bias = min(lin_func_bias_imp.min(), ac_lin_func_bias_imp.min())
            vmax_bias = max(lin_func_bias_imp.max(), ac_lin_func_bias_imp.max())

            axs[0, 0].imshow(
                image_resh, cmap="gray", vmin=vmin_ac_orig, vmax=vmax_ac_orig
            )
            axs[0, 0].set_title("Orig")
            axs[0, 1].imshow(
                reconstr_resh, cmap="gray", vmin=vmin_recon, vmax=vmax_recon
            )
            axs[0, 1].set_title("Reconstruction")
            axs[0, 2].imshow(lrp_resh, cmap="gray", vmin=vmin_lrp, vmax=vmax_lrp)
            axs[0, 2].set_title("LRP")
            axs[0, 3].imshow(
                lin_func_feature_imp, cmap="gray", vmin=vmin_fct_imp, vmax=vmax_fct_imp
            )
            axs[0, 3].set_title("Lin_Func_Feature_Imp")
            axs[0, 4].imshow(
                lin_func_applied_no_bias,
                cmap="gray",
                vmin=vmin_fct_applied_no_bias,
                vmax=vmax_fct_applied_no_bias,
            )
            axs[0, 4].set_title("Lin_func_applied_no_bias")
            axs[0, 5].imshow(
                lin_func_bias_imp, cmap="gray", vmin=vmin_bias, vmax=vmax_bias
            )
            axs[0, 5].set_title("Lin_Func_Bias_Imp")
            axs[1, 0].imshow(ac_resh, cmap="gray", vmin=vmin_ac_orig, vmax=vmax_ac_orig)
            axs[1, 0].set_title("After_cross")
            axs[1, 1].imshow(
                ac_reconstr_resh, cmap="gray", vmin=vmin_recon, vmax=vmax_recon
            )
            axs[1, 1].set_title("AC_Reconstruction")
            axs[1, 2].imshow(ac_lrp_resh, cmap="gray", vmin=vmin_lrp, vmax=vmax_lrp)
            axs[1, 2].set_title("AC_LRP")
            axs[1, 3].imshow(
                ac_lin_func_feature_imp,
                cmap="gray",
                vmin=vmin_fct_imp,
                vmax=vmax_fct_imp,
            )
            axs[1, 3].set_title("AC_Lin_Func_Feature_Imp")
            axs[1, 4].imshow(
                ac_lin_func_applied_no_bias,
                cmap="gray",
                vmin=vmin_fct_applied_no_bias,
                vmax=vmax_fct_applied_no_bias,
            )
            axs[1, 4].set_title("AC_Lin_func_applied_no_bias")
            axs[1, 5].imshow(
                ac_lin_func_bias_imp, cmap="gray", vmin=vmin_bias, vmax=vmax_bias
            )
            axs[1, 5].set_title("AC_Lin_Func_Bias_Imp")
            axs[2, 0].imshow(
                orig_ac_diff, cmap="gray", vmin=vmin_ac_orig, vmax=vmax_ac_orig
            )
            axs[2, 0].set_title("orig_ac_diff")
            axs[2, 1].imshow(recon_diff, cmap="gray", vmin=vmin_recon, vmax=vmax_recon)
            axs[2, 1].set_title("recon_diff")
            axs[2, 2].imshow(lrp_diff, cmap="gray", vmin=vmin_lrp, vmax=vmax_lrp)
            axs[2, 2].set_title("lrp_diff")
            axs[2, 3].imshow(
                lin_fct_diff, cmap="gray", vmin=vmin_fct_imp, vmax=vmax_fct_imp
            )
            axs[2, 3].set_title("lin_fct_diff")
            axs[2, 4].imshow(
                lin_fct_no_bias_diff,
                cmap="gray",
                vmin=vmin_fct_applied_no_bias,
                vmax=vmax_fct_applied_no_bias,
            )
            axs[2, 4].set_title("AC_Lin_func_applied_no_bias_diff")
            axs[2, 5].imshow(bias_fct_diff, cmap="gray", vmin=vmin_bias, vmax=vmax_bias)
            axs[2, 5].set_title("bias_fct_diff")
            self.evaluation.save_figure(
                fig, f"plot_mnist_sample_smallest_dist_withAC{ctr}"
            )
            plt.close("all")
            ctr += 1
