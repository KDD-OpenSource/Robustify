import numpy as np
import pandas as pd
from .dataset import dataset
from sklearn.datasets import fetch_openml


class creditcardFraud(dataset):
    def __init__(
        self,
        name: str = "creditcardFraud",
        file_path: str = None,
        subsample: int = None,
        num_samples: int = 2000,
        num_anomalies: int = 20,
    ):
        super().__init__(name, file_path, subsample)
        self.num_samples = num_samples
        self.num_anomalies = num_anomalies

    def create(self):
        # NOTE: THIS IS PROBABLY NOT THE WAY TO DO THE ANOMALIES!
        creditcard_data = pd.read_csv("./datasets/creditcardFraud/creditcard.csv")
        if self.num_samples != -1:
            creditcard_data = creditcard_data.sample(n=self.num_samples)
        creditcard_data.drop(["Class", "Time"], inplace=True, axis=1)
        max_per_col = creditcard_data.max().values
        min_per_col = creditcard_data.min().values
        anomalies = np.random.uniform(
            low=min_per_col,
            high=max_per_col,
            size=(self.num_anomalies, creditcard_data.shape[1]),
        )
        data = np.vstack([creditcard_data, anomalies])
        data = pd.DataFrame(data)
        self._train_data = data
        self.train_labels = pd.Series(0, range(self.num_samples))
        anom_labels = pd.Series(-1, range(self.num_anomalies))
        self.train_labels = self.train_labels.append(anom_labels, ignore_index=True)
