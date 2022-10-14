import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score

from .evaluation import evaluation


class num_anomalies_sanity:
    def __init__(self, eval_inst: evaluation, name: str =
            "num_anomalies_sanity"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        anomaly_scores = algorithm.calc_anomalyScores(algorithm.module,
                dataset.test_data())
        num_anoms = (anomaly_scores > algorithm.anom_radius).sum()
        num_normal = (anomaly_scores <= algorithm.anom_radius).sum()
        true_num_anoms = dataset.test_labels[dataset.test_labels == -1].shape[0]
        true_anom_series = dataset.get_anom_labels_from_test_labels()
        pred_anom_series = algorithm.pred_anom_labels(dataset.test_data())
        acc_score = accuracy_score(true_anom_series, pred_anom_series)
        import pdb; pdb.set_trace()
        result_dict = {}
        result_dict['true_num_anomalies'] = float(true_num_anoms)
        result_dict['num_anomalies'] = float(num_anoms)
        result_dict['num_normal'] = float(num_normal)
        result_dict['acc'] = acc_score
        self.evaluation.save_json(result_dict, "num_anomalies")
