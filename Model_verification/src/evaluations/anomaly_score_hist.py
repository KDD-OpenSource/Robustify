import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

from .evaluation import evaluation


class anomaly_score_hist:
    def __init__(self, eval_inst: evaluation, name: str = "anomaly_score_hist"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        n_bins = 50
        num_anomalies = 1000
        anomaly_eps = 1.0
        test_data = dataset.test_data()
        anomaly_base_samples = test_data.sample(num_anomalies)
        anomaly_vects = np.random.uniform(-1,1, size =
                anomaly_base_samples.shape).astype(np.float32)
        anomaly_base_vect_sum = anomaly_base_samples + anomaly_vects
        anomaly_result_unscaled = (anomaly_base_samples[(anomaly_base_vect_sum <
                -1) | (anomaly_base_vect_sum > 1)] -
                anomaly_vects).fillna(anomaly_base_vect_sum)
        diff = anomaly_base_samples - anomaly_result_unscaled
        diff_lengths = np.sqrt((diff**2).sum(axis=1))
        rescaled_anomaly_vects = diff.divide(diff_lengths, axis=0)
        eps_scaled_anomaly_vects = anomaly_eps * rescaled_anomaly_vects
        anomalous_samples = anomaly_base_samples - eps_scaled_anomaly_vects
        # diff should have the same length as anomaly vects and should be the
        # real vectors...


        anomaly_scores_test = algorithm.calc_anomalyScores(algorithm.module, test_data)
        anomaly_scores_anomalies = algorithm.calc_anomalyScores(
                algorithm.module, anomalous_samples)
        import pdb; pdb.set_trace()
        all_scores = pd.concat([anomaly_scores_test,anomaly_scores_anomalies])



        fig, ax = plt.subplots(1,1,figsize=(20,10))
        ax.hist(all_scores, bins = n_bins)

        self.evaluation.save_figure(fig, f"anomaly_hist_{n_bins}")
        #self.evaluation.save_json(result_dict, "label_error")
        #result_dict_tot = {}
        #result_dict_tot["total_error_test_ds"] = tot_error.astype(np.float64)
        #self.evaluation.save_json(result_dict_tot, "tot_error")
