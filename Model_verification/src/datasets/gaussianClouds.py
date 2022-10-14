import numpy as np
import pandas as pd
from .dataset import dataset


class gaussianClouds(dataset):
    def __init__(
        self,
        name: str = "gaussianClouds",
        file_path: str = None,
        subsample: int = None,
        spacedim: int = 20,
        clouddim: int = 5,
        num_clouds: int = 5,
        num_samples: int = 2000,
        scale: bool = False,
        num_anomalies: int = 20,
        num_testpoints: int = 1000,
    ):
        super().__init__(name, file_path, subsample)
        self.spacedim = spacedim
        self.clouddim = clouddim
        self.num_clouds = num_clouds
        self.num_samples = num_samples
        self.num_anomalies = num_anomalies
        self.num_testpoints = num_testpoints
        self.scale = scale
        self.scale_type = 'MinMax'
        self.scale_min = -1
        self.scale_max = 1

    def create(self):
        """
        creates a synthetic DS by projecting a gaussian cloud with unit
        variance into a high dim space
        """
        points_per_cloud_train = int(self.num_samples / self.num_clouds)
        points_per_cloud_test = int(self.num_testpoints / self.num_clouds)
        random_matrices = []
        random_centers = []
        random_points_train = []
        random_points_test = []
        label = 0
        self.train_labels = pd.Series()
        self.test_labels = pd.Series()
        for _ in range(self.num_clouds):
            random_matrices.append(
                np.random.uniform(low=-1, high=1, size=(self.spacedim, self.clouddim))
            )
            random_centers.append(
                np.random.uniform(low=0, high=1, size=(self.clouddim))
            )
            random_points_train.append(
                np.random.normal(
                    loc=random_centers[-1],
                    scale=1,
                    size=(points_per_cloud_train, self.clouddim),
                ).transpose()
            )
            random_points_test.append(
                np.random.normal(
                    loc=random_centers[-1],
                    scale=1,
                    size=(points_per_cloud_test, self.clouddim),
                ).transpose()
            )
            random_points_train[-1] = np.dot(
                random_matrices[-1], random_points_train[-1]
            )
            random_points_test[-1] = np.dot(random_matrices[-1], random_points_test[-1])
            train_labels = pd.Series(label, range(points_per_cloud_train))
            test_labels = pd.Series(label, range(points_per_cloud_test))
            self.train_labels = self.train_labels.append(
                train_labels, ignore_index=True
            )
            self.test_labels = self.test_labels.append(test_labels, ignore_index=True)
            label += 1

        data_train = np.hstack(random_points_train).transpose()
        anomalies = np.random.uniform(
            low=data_train.min(),
            high=data_train.max(),
            size=(self.num_anomalies, self.spacedim),
        )
        data_train = np.vstack([data_train, anomalies])
        anom_labels = pd.Series(-1, range(self.num_anomalies))
        self.train_labels = self.train_labels.append(anom_labels, ignore_index=True)
        data_train = pd.DataFrame(data_train)
        self._train_data = data_train

        data_test = np.hstack(random_points_test).transpose()
        anomalies = np.random.uniform(
            low=data_test.min(),
            high=data_test.max(),
            size=(self.num_anomalies, self.spacedim),
        )
        data_test = np.vstack([data_test, anomalies])
        anom_labels = pd.Series(-1, range(self.num_anomalies))
        self.test_labels = self.test_labels.append(anom_labels, ignore_index=True)
        data_test = pd.DataFrame(data_test)
        self._test_data = data_test
        # add diffs to means by adding a series (self.dists_to_means)
