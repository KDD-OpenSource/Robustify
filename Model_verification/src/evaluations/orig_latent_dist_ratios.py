import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from .evaluation import evaluation


class orig_latent_dist_ratios:
    def __init__(self, eval_inst: evaluation, name: str = "orig_latent_dist_ratios"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        latent_repr = algorithm.extract_latent(dataset.test_data())
        orig_dists = pdist(dataset.test_data(), mse_dist)
        latent_dists = pdist(latent_repr, mse_dist)
        ratios = latent_dists / orig_dists
        joined_array = np.vstack([orig_dists, ratios])
        sorted_joined_array = joined_array[:, joined_array[0, :].argsort()]
        fig = plt.figure(figsize=[20, 20])
        # plt.plot(sorted(ratios))
        plt.plot(sorted_joined_array[0], label="orig_dist")
        plt.plot(sorted_joined_array[1], label="ratios")
        plt.legend()

        # plot without the extreme values (90% ?quantile?)
        # save figure
        self.evaluation.save_figure(fig, "orig_latent_dist_ratios")
        plt.close("all")


def mse_dist(u, v):
    return np.sqrt(((u - v) ** 2).sum())
