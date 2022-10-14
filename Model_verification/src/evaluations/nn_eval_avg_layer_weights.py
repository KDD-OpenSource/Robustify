import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score

from .evaluation import evaluation


class nn_eval_avg_layer_weights:
    def __init__(self, eval_inst: evaluation, name: str =
            "nn_eval_avg_layer_weights"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        layer_stats_weights = algorithm.calc_layer_stats_weights()
        layer_stats_biases = algorithm.calc_layer_stats_biases()
        layer_stats_weights_pd = pd.DataFrame(layer_stats_weights).transpose()
        if len(layer_stats_biases) == 0:
            fig, ax = plt.subplots(1,1,figsize=(20,10))
            ax.plot(layer_stats_weights_pd['mean'])
            ax.plot(layer_stats_weights_pd['std'])
            ax.plot(layer_stats_weights_pd['min'])
            ax.plot(layer_stats_weights_pd['max'])
            ax.set_title('weights')
        else:
            fig, ax = plt.subplots(2,1,figsize=(20,10))
            layer_stats_biases_pd = pd.DataFrame(layer_stats_biases).transpose()
            ax[0].plot(layer_stats_weights_pd['mean'])
            ax[0].plot(layer_stats_weights_pd['std'])
            ax[0].plot(layer_stats_weights_pd['min'])
            ax[0].plot(layer_stats_weights_pd['max'])
            ax[0].set_title('weights')
            ax[1].plot(layer_stats_biases_pd['mean'])
            ax[1].plot(layer_stats_biases_pd['std'])
            ax[1].plot(layer_stats_biases_pd['min'])
            ax[1].plot(layer_stats_biases_pd['max'])
            ax[1].set_title('biases')
        self.evaluation.save_figure(fig, 'nn_eval_avg_layer_weights')
