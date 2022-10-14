import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from .dataset import dataset


class uniformClouds(dataset):
    def __init__(
        self,
        name: str = "uniformClouds",
        file_path: str = None,
        subsample: int = None,
        spacedim: int = 20,
        clouddim: int = 5,
        num_clouds: int = 5,
        num_samples: int = 2000,
        num_anomalies: int = 20,
        noise: bool = False,
    ):
        super().__init__(name, file_path, subsample)
        self.spacedim = spacedim
        self.clouddim = clouddim
        self.num_clouds = num_clouds
        self.num_samples = num_samples
        self.num_anomalies = num_anomalies
        self.noise = noise

    def create(self):
        """
        creates a synthetic DS by projecting a uniform cloud with unit
        width into a high dim space
        """
        points_per_cloud = int(self.num_samples / self.num_clouds)
        random_matrices = []
        random_centers = []
        random_points = []
        label = 0
        self.train_labels = pd.Series()
        for _ in range(self.num_clouds):
            random_matrices.append(
                np.random.uniform(low=-1, high=1, size=(self.spacedim, self.clouddim))
            )
            random_centers.append(
                np.random.uniform(low=-1, high=1, size=(self.clouddim))
            )
            random_points.append(
                np.random.uniform(
                    low=np.array(random_centers[-1]) - 0.5,
                    high=np.array(random_centers[-1]) + 0.5,
                    size=(points_per_cloud, self.clouddim),
                ).transpose()
            )
            random_points[-1] = np.dot(random_matrices[-1], random_points[-1])
            labels = pd.Series(label, range(points_per_cloud))
            self.train_labels = self.train_labels.append(labels, ignore_index=True)
            label += 1
            if self.noise:
                random_points[-1] += np.random.normal(
                    loc=0, scale=0.01, size=random_points[-1].shape
                )

        anomalies = np.random.uniform(
            low=-1, high=1, size=(self.num_anomalies, self.spacedim)
        )

        data = np.hstack(random_points).transpose()
        data = np.vstack([data, anomalies])
        data = pd.DataFrame(data)
        self._train_data = data
        anom_labels = pd.Series(-1, range(self.num_anomalies))
        self.train_labels = self.train_labels.append(anom_labels, ignore_index=True)
