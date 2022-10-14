import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from .evaluation import evaluation
from ..algorithms.neural_net import smallest_k_dist_loss


class plot_closest_border_dist:
    def __init__(self, eval_inst: evaluation, name: str =
            "plot_closest_border_dist"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        num_closest_borders = 5
        data_loader = DataLoader(
            dataset=dataset.test_data().values,
            batch_size=20,
            drop_last=False,
            pin_memory=True,
        )
        closest_dists = []
        for inst_batch in data_loader:
            for instance in inst_batch:
                subfcts = algorithm.get_neuron_border_subFcts(algorithm.module,
                        instance)
                dists = sorted(
                        algorithm.get_dists_from_border_subFcts(instance,
                    subfcts))
                closest_dists.append(
                        (dists[0],
                        sum(dists[:num_closest_borders])
                            ))

        closest_dists_sorted = sorted(closest_dists)
        fig, ax = plt.subplots(2,1,figsize = (20,10))
        ax[0].plot(list(map(lambda x:x[0], closest_dists_sorted)))
        ax[1].plot(list(map(lambda x:x[1], closest_dists_sorted)))
        self.evaluation.save_figure(fig, 'plot_closest_border_dist')
