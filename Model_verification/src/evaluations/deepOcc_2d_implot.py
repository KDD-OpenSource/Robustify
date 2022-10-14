import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from itertools import combinations

from .evaluation import evaluation


class deepOcc_2d_implot:
    def __init__(self, eval_inst: evaluation, name: str =
            "deepOcc_2d_implot"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        predictions = algorithm.predict(dataset.test_data())
        fig, ax = plt.subplots(2,1, figsize=(30,60))
        #plt.set_cmap('gist_rainbow')
        cmap = matplotlib.cm.get_cmap("tab20")
        ax[0].scatter(predictions.iloc[:,0], predictions.iloc[:,1],
                color = cmap(dataset.test_labels))
        ax[0].scatter(algorithm.center[0], algorithm.center[1], color='green')
        circle = plt.Circle((algorithm.center[0],
            algorithm.center[1]), algorithm.anom_radius, edgecolor='blue',
            fill=False)
        ax[0].add_artist(circle)
        #self.evaluation.save_figure(fig, "deepOcc_2d_implot")

        import pdb; pdb.set_trace()

        pred_normal = predictions[dataset.test_labels>=0]
        labels_normal = dataset.test_labels[dataset.test_labels>=0]
        ax[1].scatter(pred_normal.iloc[:,0], pred_normal.iloc[:,1],
                color = cmap(labels_normal))
        ax[1].scatter(algorithm.center[0], algorithm.center[1], color='green')
        circle = plt.Circle((algorithm.center[0],
            algorithm.center[1]), algorithm.anom_radius, edgecolor='blue',
            fill=False)
        ax[1].add_artist(circle)
        self.evaluation.save_figure(fig, "deepOcc_2d_implot_normal")
