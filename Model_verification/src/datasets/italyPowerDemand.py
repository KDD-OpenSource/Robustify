import numpy as np
import pandas as pd
from .dataset import dataset
from sklearn.datasets import fetch_openml


class italyPowerDemand(dataset):
    def __init__(
        self,
        name: str = "italyPowerDemand",
        file_path: str = None,
        subsample: int = None,
    ):
        super().__init__(name, file_path, subsample)

    def create(self):
        # NOTE: THIS IS PROBABLY NOT THE WAY TO DO THE ANOMALIES!
        italyPowerDemand_train = pd.read_csv(
            "./datasets/ItalyPowerDemand/ItalyPowerDemand_TRAIN", header=None
        )
        italyPowerDemand_test = pd.read_csv(
            "./datasets/ItalyPowerDemand/ItalyPowerDemand_TEST", header=None
        )
        italyPowerDemand_data = pd.concat(
            [italyPowerDemand_train, italyPowerDemand_test], ignore_index=True
        )
        italyPowerDemand_train = italyPowerDemand_data[:500]
        italyPowerDemand_test = italyPowerDemand_data[500:]
        self.train_labels = italyPowerDemand_train[0]
        italyPowerDemand_train.drop([0], inplace=True, axis=1)
        self._train_data = italyPowerDemand_train
        self.test_labels = italyPowerDemand_test[0]
        italyPowerDemand_test.drop([0], inplace=True, axis=1)
        self._test_data = italyPowerDemand_test
