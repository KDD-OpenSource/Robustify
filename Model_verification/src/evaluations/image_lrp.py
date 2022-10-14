import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

from .evaluation import evaluation


class image_lrp:
    def __init__(self, eval_inst: evaluation, name: str = "image_lrp"):
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
            relevance_image = lrp.sum()
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=[20, 20])
            # figure is given as 1d array
            fig_side = int(np.sqrt(image.shape))
            image_resh = image.reshape(fig_side, fig_side)
            reconstr_resh = reconstr.reshape(fig_side, fig_side)
            lrp_resh = lrp.reshape(fig_side, fig_side)
            axs[0].imshow(image_resh, cmap="gray")
            axs[1].imshow(reconstr_resh, cmap="gray")
            axs[2].imshow(lrp_resh, cmap="gray")
            fig.suptitle(
                f"""Relevance of image: {relevance_image}, Relevance
                of bias: {relevance_bias}"""
            )
            self.evaluation.save_figure(fig, f"plot_mnist_sample_lrp_{ctr}")
            plt.close("all")
            ctr += 1
