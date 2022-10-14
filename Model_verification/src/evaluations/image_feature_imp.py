import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

from .evaluation import evaluation


class image_feature_imp:
    def __init__(self, eval_inst: evaluation, name: str = "image_feature_imp"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        images = dataset.test_data().sample(100)
        reconstructions = algorithm.predict(images)
        ctr = 0
        for image, reconstr in zip(images.values, reconstructions.values):
            lrp, relevance_bias = algorithm.lrp_ae(
                algorithm.module, torch.tensor(image)
            )
            lin_func_feature_imp = algorithm.lin_func_feature_imp(
                algorithm.module, torch.tensor(image)
            )
            lin_func_bias_imp = algorithm.lin_func_bias_imp(
                algorithm.module, torch.tensor(image)
            )
            relevance_image = lrp.sum()
            fig, axs = plt.subplots(nrows=1, ncols=5, figsize=[20, 20])
            # figure is given as 1d array
            fig_side = int(np.sqrt(image.shape))
            image_resh = image.reshape(fig_side, fig_side)
            reconstr_resh = reconstr.reshape(fig_side, fig_side)
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
            fig.suptitle(
                f"""Relevance of image: {relevance_image}, Relevance
                of bias: {relevance_bias}"""
            )
            self.evaluation.save_figure(
                fig, f"plot_mnist_sample_lrp_feature_bias_imp_{ctr}"
            )
            plt.close("all")
            ctr += 1


# Idea: have another 'error' by subtracting the function imps from the input
