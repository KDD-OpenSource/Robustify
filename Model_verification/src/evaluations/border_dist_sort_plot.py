import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd

from .evaluation import evaluation


class border_dist_sort_plot:
    def __init__(self, eval_inst: evaluation, name: str = "border_dist_sort_plot"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        sample_dist_pairs = algorithm.assign_border_dists(
            algorithm.module, dataset.test_data()
        )
        points = pd.DataFrame(map(lambda x: x[0], sample_dist_pairs))
        dists = pd.Series(map(lambda x: x[1], sample_dist_pairs))
        dists_sorted = dists.sort_values(ascending=False).values
        fig = plt.figure(figsize=[20, 20])
        ##plt.plot(dists_sorted)
        import pdb

        pdb.set_trace()
        # plot without the extreme values (90% ?quantile?)
        # save figure
        self.evaluation.save_figure(fig, "dist_plot")
        plt.close("all")
