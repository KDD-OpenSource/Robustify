import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

from .evaluation import evaluation


class plot_mnist_samples:
    def __init__(self, eval_inst: evaluation, name: str = "plot_mnist_samples"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        images = dataset.test_data().sample(10)
        reconstructions = algorithm.predict(images)
        ctr = 0
        for image, reconstr in zip(images.values, reconstructions.values):
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[20, 20])
            image_resh = image.reshape(28, 28)
            reconstr_resh = reconstr.reshape(28, 28)
            axs[0].imshow(image_resh, cmap="gray")
            axs[1].imshow(reconstr_resh, cmap="gray")
            self.evaluation.save_figure(fig, f"plot_mnist_sample_{ctr}")
            plt.close("all")
            ctr += 1
