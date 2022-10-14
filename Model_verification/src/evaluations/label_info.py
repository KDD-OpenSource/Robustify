import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

from .evaluation import evaluation


class label_info:
    def __init__(self, eval_inst: evaluation, name: str = "label_info"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        dataset.test_data()
        test_labels = []
        means = []
        variances = []
        for label in dataset.test_labels.unique():
            test_labels.append(label)
            label_data = dataset.test_data().loc[
                dataset.test_labels[dataset.test_labels == label].index
            ]
            label_mean = label_data.mean()
            # check if right function
            label_var = label_data.var()
            means.append(label_mean)
            variances.append(label_var)

        label_mean_tuples = list(zip(test_labels, means))
        label_var_tuples_dims = dict(
            zip(
                list(map(lambda x: str(x), test_labels)),
                list(map(lambda x: list(x.values.astype(np.float64)), variances)),
            )
        )
        label_var_tuples_aggr = dict(
            zip(
                list(map(lambda x: str(x), test_labels)),
                list(map(lambda x: x.var().astype(np.float64), variances)),
            )
        )
        result_dict = {}
        for label_mean1, label_mean2 in list(combinations(label_mean_tuples, 2)):
            label1 = label_mean1[0]
            label2 = label_mean2[0]
            mean1 = pd.DataFrame(label_mean1[1]).transpose()
            mean2 = pd.DataFrame(label_mean2[1]).transpose()
            mean_dist = (
                np.sqrt(((mean1 - mean2) ** 2).sum(axis=1)).values[0].astype(np.float64)
            )
            result_dict[str(label1) + "_" + str(label2)] = mean_dist
        self.evaluation.save_json(result_dict, "label_mean_dist")
        self.evaluation.save_json(dict(label_var_tuples_dims), "label_vars_dims")
        self.evaluation.save_json(dict(label_var_tuples_aggr), "label_vars_aggr")
