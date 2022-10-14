import matplotlib.pyplot as plt
import ast
import numpy as np
import pandas as pd

from .evaluation import evaluation


class plot_entire_train_set:
    def __init__(
        self, eval_inst: evaluation, name: str = "plot_entire_train_set"):
        self.name = name
        self.evaluation = eval_inst
        self.plots = []

    def evaluate(self, dataset, algorithm):
        # input_points: pd.DataFrame,
        # output_points: pd.DataFrame):
        # sample indices
        points = dataset.train_data()
        labels = dataset.train_labels
        tot_min = points.min().min()
        tot_max = points.max().max()
        num_labels = len(labels.unique())
        fig, ax = plt.subplots(num_labels, 1, figsize=(20,20))
        if 'scales' in dataset.__dict__.keys():
            scales = ast.literal_eval(dataset.scales)
        for ind, label in enumerate(labels.unique()):
            label_subset = points[labels == label]
            ax[ind].set_ylim(tot_min, tot_max)
            ax[ind].plot(label_subset.transpose(), color = 'grey', alpha = 0.01)
            if 'scales' in dataset.__dict__.keys():
                ax[ind].set_title(f'Scale: {scales[ind]}')
        self.evaluation.save_figure(fig, 'plot_entire_train_set')
