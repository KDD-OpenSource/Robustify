import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from .evaluation import evaluation
from ..algorithms.neural_net import smallest_k_dist_loss


class num_subfcts_per_class:
    def __init__(self, eval_inst: evaluation, name: str =
            "num_subfcts_per_class"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        test_data = dataset.test_data()
        test_labels = dataset.test_labels
        result_dict = {}
        for label in test_labels.unique():
            label_data = test_data[test_labels == label]
            label_subfcts = len(algorithm.count_lin_subfcts(algorithm.module,
                    label_data))
            result_dict[f'Test_data: {str(label)}'] = label_subfcts

        # add noise to them
        normal_labels = test_labels.unique()[test_labels.unique()>=0]
        for label in normal_labels:
            label_data = test_data[test_labels==label]
            label_var = label_data.var()
            noisy_label_data = label_data + np.random.normal(
                    loc = 0, scale = 2*label_var, size = label_data.shape)
            noisy_subfcts = len(algorithm.count_lin_subfcts(algorithm.module,
                noisy_label_data))
            result_dict[f'Noisy Testdata: {str(label)}'] = noisy_subfcts

        self.evaluation.save_json(result_dict, 'num_subfcts_per_class')
