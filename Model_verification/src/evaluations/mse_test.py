import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

from .evaluation import evaluation


class mse_test:
    def __init__(self, eval_inst: evaluation, name: str = "mse_test"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        dataset.test_data()
        test_labels = []
        label_error = []
        for label in dataset.test_labels.unique():
            test_labels.append(label)
            label_data = dataset.test_data().loc[
                dataset.test_labels[dataset.test_labels == label].index
            ]
            label_data_pred = algorithm.predict(label_data)
            mse_label = np.sqrt(((label_data_pred - label_data) ** 2).sum(axis=1)).sum()
            label_error.append(mse_label)
            # label_mean = label_data.mean()
            # check if right function
            # label_var = label_data.var()
            # means.append(label_mean)
            # variances.append(label_var)
        tot_error = np.sqrt(
            ((algorithm.predict(dataset.test_data()) - dataset.test_data()) ** 2).sum(
                axis=1
            )
        ).sum()
        result_dict = {}
        label_mean_tuples = list(
            zip(test_labels, list(map(lambda x: x.astype(np.float64), label_error)))
        )
        for label in label_mean_tuples:
            result_dict[str(label[0])] = label[1]

        self.evaluation.save_json(result_dict, "label_error")
        result_dict_tot = {}
        result_dict_tot["total_error_test_ds"] = tot_error.astype(np.float64)
        self.evaluation.save_json(result_dict_tot, "tot_error")
