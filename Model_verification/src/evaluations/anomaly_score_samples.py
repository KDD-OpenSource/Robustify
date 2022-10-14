import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

from .evaluation import evaluation


class anomaly_score_samples:
    def __init__(self, eval_inst: evaluation, name: str =
            "anomaly_score_samples"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        num_samples = 10
        test_data = dataset.test_data()
        normal_data = test_data.sample(num_samples)
        noise_data = pd.DataFrame(np.random.uniform(-1,1, size =
                normal_data.shape).astype(np.float32))


        anomaly_eps = 5.0
        anomaly_base_samples = test_data.sample(num_samples)
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

        anomaly_scores_normal = algorithm.calc_anomalyScores(algorithm.module,
                normal_data)
        anomaly_scores_noise = algorithm.calc_anomalyScores(
                algorithm.module, noise_data)
        anomaly_scores_anom = algorithm.calc_anomalyScores(
                algorithm.module, anomalous_samples)



        ctr = 0
        im_dim_size = int(np.sqrt(test_data.shape[1]))
        for image, anom_score in zip(normal_data.values,
                anomaly_scores_normal.values):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 10])
            image_resh = image.reshape(im_dim_size, im_dim_size)
            ax.imshow(image_resh, cmap="gray")
            ax.set_title(f'Anomaly_score: {anom_score}')
            self.evaluation.save_figure(fig, f"plot_mnist_sample_{ctr}")
            plt.close("all")
            ctr += 1


        ctr = 0
        for image, anom_score in zip(noise_data.values,
                anomaly_scores_noise.values):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 10])
            image_resh = image.reshape(im_dim_size, im_dim_size)
            ax.imshow(image_resh, cmap="gray")
            ax.set_title(f'Anomaly_score: {anom_score}')
            self.evaluation.save_figure(fig, f"plot_noise_sample_{ctr}")
            plt.close("all")
            ctr += 1

        ctr = 0
        for image, anom_score in zip(anomalous_samples.values,
                anomaly_scores_anom.values):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 10])
            image_resh = image.reshape(im_dim_size, im_dim_size)
            ax.imshow(image_resh, cmap="gray")
            ax.set_title(f'Anomaly_score: {anom_score}')
            self.evaluation.save_figure(fig, f"plot_anom_sample_{ctr}")
            plt.close("all")
            ctr += 1
