import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd

from .evaluation import evaluation


class qual_by_border_dist_plot:
    def __init__(self, eval_inst: evaluation, name: str = "qual_by_border_dist_plot"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        sample_dist_pairs = algorithm.assign_border_dists(
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
            fig, axs = plt.subplots(nrows=1, ncols=5, figsize=[20, 20])
            reconstruction = np.array(
                algorithm.predict(pd.DataFrame(image).transpose())
            )
            image_np = np.array(image)
            fig.suptitle(f"""Dist_value is {dist_value}""")
            lrp, relevance_bias = algorithm.lrp_ae(algorithm.module, image)
            lin_func_feature_imp = algorithm.lin_func_feature_imp(
                algorithm.module, image
            )
            lin_func_bias_imp = algorithm.lin_func_bias_imp(algorithm.module, image)
            image_resh = image_np.reshape(fig_side, fig_side)
            reconstr_resh = reconstruction.reshape(fig_side, fig_side)
            lrp_resh = lrp.reshape(fig_side, fig_side)
            lin_func_feature_imp = lin_func_feature_imp.reshape(fig_side, fig_side)
            lin_func_bias_imp = lin_func_bias_imp.reshape(fig_side, fig_side)
            axs[0].imshow(image_resh, cmap="gray")
            axs[0].set_title("Orig")
            axs[1].imshow(reconstr_resh, cmap="gray")
            axs[1].set_title("Reconstruction")
            axs[2].imshow(lrp_resh, cmap="gray")
            axs[2].set_title("LRP")
            axs[3].imshow(lin_func_feature_imp, cmap="gray")
            axs[3].set_title("Lin_Func_Feature_Imp")
            axs[4].imshow(lin_func_bias_imp, cmap="gray")
            axs[4].set_title("Lin_Func_Bias_Imp")
            self.evaluation.save_figure(fig, f"plot_mnist_sample_largest_dist{ctr}")
            plt.close("all")
            ctr += 1

        ctr = 0
        for df_entry in mid_dist_df.values:
            image = df_entry[0]
            dist_value = df_entry[1]
            fig_side = int(np.sqrt(image.shape))

            fig, axs = plt.subplots(nrows=1, ncols=5, figsize=[20, 20])
            reconstruction = np.array(
                algorithm.predict(pd.DataFrame(image).transpose())
            )
            image_np = np.array(image)
            fig.suptitle(f"""Dist_value is {dist_value}""")

            lrp, relevance_bias = algorithm.lrp_ae(algorithm.module, image)
            lin_func_feature_imp = algorithm.lin_func_feature_imp(
                algorithm.module, image
            )
            lin_func_bias_imp = algorithm.lin_func_bias_imp(algorithm.module, image)
            image_resh = image_np.reshape(fig_side, fig_side)
            reconstr_resh = reconstruction.reshape(fig_side, fig_side)
            lrp_resh = lrp.reshape(fig_side, fig_side)
            lin_func_feature_imp = lin_func_feature_imp.reshape(fig_side, fig_side)
            lin_func_bias_imp = lin_func_bias_imp.reshape(fig_side, fig_side)
            axs[0].imshow(image_resh, cmap="gray")
            axs[0].set_title("Orig")
            axs[1].imshow(reconstr_resh, cmap="gray")
            axs[1].set_title("Reconstruction")
            axs[2].imshow(lrp_resh, cmap="gray")
            axs[2].set_title("LRP")
            axs[3].imshow(lin_func_feature_imp, cmap="gray")
            axs[3].set_title("Lin_Func_Feature_Imp")
            axs[4].imshow(lin_func_bias_imp, cmap="gray")
            axs[4].set_title("Lin_Func_Bias_Imp")
            self.evaluation.save_figure(fig, f"plot_mnist_sample_mid_dist{ctr}")
            plt.close("all")
            ctr += 1

        ctr = 0
        for df_entry in smallest_dist_df.values:
            image = df_entry[0]
            dist_value = df_entry[1]
            fig_side = int(np.sqrt(image.shape))
            fig, axs = plt.subplots(nrows=1, ncols=5, figsize=[20, 20])
            reconstruction = np.array(
                algorithm.predict(pd.DataFrame(image).transpose())
            )
            image_np = np.array(image)
            fig.suptitle(f"""Dist_value is {dist_value}""")
            lrp, relevance_bias = algorithm.lrp_ae(algorithm.module, image)
            lin_func_feature_imp = algorithm.lin_func_feature_imp(
                algorithm.module, image
            )
            lin_func_bias_imp = algorithm.lin_func_bias_imp(algorithm.module, image)
            image_resh = image_np.reshape(fig_side, fig_side)
            reconstr_resh = reconstruction.reshape(fig_side, fig_side)
            lrp_resh = lrp.reshape(fig_side, fig_side)
            lin_func_feature_imp = lin_func_feature_imp.reshape(fig_side, fig_side)
            lin_func_bias_imp = lin_func_bias_imp.reshape(fig_side, fig_side)
            axs[0].imshow(image_resh, cmap="gray")
            axs[0].set_title("Orig")
            axs[1].imshow(reconstr_resh, cmap="gray")
            axs[1].set_title("Reconstruction")
            axs[2].imshow(lrp_resh, cmap="gray")
            axs[2].set_title("LRP")
            axs[3].imshow(lin_func_feature_imp, cmap="gray")
            axs[3].set_title("Lin_Func_Feature_Imp")
            axs[4].imshow(lin_func_bias_imp, cmap="gray")
            axs[4].set_title("Lin_Func_Bias_Imp")
            self.evaluation.save_figure(fig, f"plot_mnist_sample_smallest_dist{ctr}")
            plt.close("all")
            ctr += 1
