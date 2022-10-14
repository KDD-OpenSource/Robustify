import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd

from .evaluation import evaluation


class border_dist_2d:
    def __init__(self, eval_inst: evaluation, name: str = "border_dist_2d"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        input_dim = algorithm.topology[0]
        if input_dim != 2:
            raise Exception("cannot plot in 2d unless input dim is 2d too")

        sample_dist_pairs = algorithm.assign_border_dists(
            algorithm.module, dataset.test_data()
        )
        points = pd.DataFrame(map(lambda x: x[0], sample_dist_pairs))
        dists = pd.DataFrame(map(lambda x: x[1], sample_dist_pairs), columns=[2])
        joined = pd.concat([points, dists], axis=1)
        fig = plt.figure(figsize=[20, 20])
        plt.scatter(joined[0], joined[1], c=joined[2], alpha=0.5, cmap="Greens")
        # save figure
        self.evaluation.save_figure(fig, "border_dist_2d")
        plt.close("all")
