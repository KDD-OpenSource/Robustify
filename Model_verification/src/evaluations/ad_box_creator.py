import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

from .evaluation import evaluation


class ad_box_creator:
    def __init__(self, eval_inst: evaluation, name: str =
            "ad_box_creator"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        num_normal_samples = 1
        num_anomalous_samples = 1
        eps_range = [i/1000 for i in range(1,11)]
        test_data = dataset.test_data()

        anomaly_scores_normal = pd.DataFrame(algorithm.calc_anomalyScores(algorithm.module,
                test_data), columns = ['anomaly_score'])
        test_data_scores = pd.concat([test_data, anomaly_scores_normal], axis=1)
        norm_quantile = np.quantile(test_data_scores['anomaly_score'],0.5)
        anom_quantile = np.quantile(test_data_scores['anomaly_score'],0.999)

        test_data_score_norm = test_data[test_data_scores['anomaly_score'] <
                norm_quantile]
        test_data_score_anom = test_data[test_data_scores['anomaly_score'] >
                anom_quantile]

        norm_data = test_data_score_norm.sample(num_normal_samples)
        anom_data = test_data_score_anom.sample(num_anomalous_samples)

        for eps in eps_range:
            lb = (norm_data.values - eps).clip(-1,1)
            ub = (norm_data.values + eps).clip(-1,1)
            for sample in range(len(norm_data)):
                bounds = [[a,b] for (a,b) in zip(lb[sample], ub[sample])]
                self.save_bounds(bounds, f'normal_{sample}_box_{eps}')

        for eps in eps_range:
            lb = (anom_data.values - eps).clip(-1,1)
            ub = (anom_data.values + eps).clip(-1,1)
            for sample in range(len(anom_data)):
                bounds = [[a,b] for (a,b) in zip(lb[sample], ub[sample])]
                self.save_bounds(bounds, f'anom_{sample}_box_{eps}')

        self.evaluation.save_csv(norm_data.transpose(), 'norm_data')
        self.evaluation.save_csv(anom_data.transpose(), 'anom_data')
        norm_data_pred = algorithm.predict(norm_data)
        anom_data_pred = algorithm.predict(anom_data)
        self.evaluation.save_csv(norm_data_pred.transpose(), 'norm_data_pred')
        self.evaluation.save_csv(anom_data_pred.transpose(), 'anom_data_pred')
        anom_quantile_dict = {'anom_quantile':anom_quantile}
        self.evaluation.save_json(anom_quantile_dict, 'anom_quantile')



    def save_bounds(self, bounds, name, subfolder=None):
        # bounds is list of list
        if subfolder:
            dir_path = os.path.join(self.evaluation.run_folder, subfolder)
            file_path = os.path.join(dir_path, name)
            os.makedirs(dir_path, exist_ok=True)
            with open(os.path.join(file_path, name), 'w') as f:
                for interval in bounds:
                    f.write(str(interval) + '\n')

            #figure.savefig(os.path.join(self.run_folder, subfolder, name))
        else:
            file_path = os.path.join(self.evaluation.run_folder, name)
            with open(file_path, 'w') as f:
                for interval in bounds:
                    f.write(str(interval) + '\n')
