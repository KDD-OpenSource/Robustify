import numpy as np
import pandas as pd
from .dataset import dataset


class parabola(dataset):
    def __init__(
        self,
        name: str = "parabola",
        file_path: str = None,
        subsample: int = None,
        num_samples: int = 2000,
        num_anomalies: int = 20,
        noise: float = 0.1,
        spacedim=2,
    ):
        super().__init__(name, file_path, subsample)
        self.num_samples = num_samples
        self.num_anomalies = num_anomalies
        self.noise = noise
        self.spacedim = spacedim

    def create(self):
        """
        creates a synthetic DS by projecting a uniform cloud with unit
        width into a high dim space
        """
        base_uniform_points = np.random.uniform(
            low=-5, high=5, size=(self.num_samples, self.spacedim - 1)
        )
        squared_dim = np.square(base_uniform_points).sum(axis=1)
        joined_data = np.hstack([base_uniform_points, squared_dim[:, np.newaxis]])
        anomalies = np.random.uniform(
            low=-5, high=5, size=(self.num_anomalies, self.spacedim)
        )
        data = np.vstack([joined_data, anomalies])
        data = pd.DataFrame(data)
        self._train_data = data
        self.train_labels = pd.Series(0, range(self.num_samples))
        anom_labels = pd.Series(-1, range(self.num_anomalies))
        self.train_labels = self.train_labels.append(anom_labels, ignore_index=True)
