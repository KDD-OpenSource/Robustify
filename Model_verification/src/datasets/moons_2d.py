from sklearn.datasets import make_moons
import numpy as np
import pandas as pd
from .dataset import dataset


class moons_2d(dataset):
    def __init__(
        self,
        name: str = "moons_2d",
        file_path: str = None,
        subsample: int = None,
        num_samples: int = 2000,
        num_anomalies: int = 20,
        noise: float = 0.1,
    ):
        super().__init__(name, file_path, subsample)
        self.num_samples = num_samples
        self.num_anomalies = num_anomalies
        self.noise = noise

    def create(self):
        """
        creates a synthetic DS by projecting a uniform cloud with unit
        width into a high dim space
        """
        dataset = make_moons(self.num_samples, noise=self.noise)[0]
        anomalies = np.random.uniform(low=-1, high=1, size=(self.num_anomalies, 2))
        data = np.vstack([dataset, anomalies])
        data = pd.DataFrame(data)
        self.train_labels = pd.Series(0, range(self.num_samples))
        anom_labels = pd.Series(-1, range(self.num_anomalies))
        self.train_labels = self.train_labels.append(anom_labels, ignore_index=True)
        self._train_data = data
