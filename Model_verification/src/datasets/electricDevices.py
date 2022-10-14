import numpy as np
import pandas as pd
from .dataset import dataset
from sklearn.datasets import fetch_openml


class electricDevices(dataset):
    def __init__(
        self,
        name: str = "electricDevices",
        file_path: str = None,
        subsample: int = None,
    ):
        super().__init__(name, file_path, subsample)

    def create(self):
        dataset_train = pd.read_csv(
            "./datasets/ElectricDevices/ElectricDevices_TRAIN", header=None
        )
        dataset_test = pd.read_csv(
            "./datasets/ElectricDevices/ElectricDevices_TEST", header=None
        )
        # electricDevices_data = pd.concat([electricDevices_train,
        # electricDevices_test], ignore_index=True)
        self.train_labels = dataset_train[0]
        dataset_train.drop([0], inplace=True, axis=1)
        self._train_data = dataset_train
        self.test_labels = dataset_test[0]
        dataset_test.drop([0], inplace=True, axis=1)
        self._test_data = dataset_test
