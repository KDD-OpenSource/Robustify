import numpy as np
import pandas as pd
from .dataset import dataset
from sklearn.datasets import fetch_openml


class ecg5000(dataset):
    def __init__(
        self,
        name: str = "ecg5000",
        file_path: str = None,
        subsample: int = None,
    ):
        super().__init__(name, file_path, subsample)

    def create(self):
        # NOTE: THIS IS PROBABLY NOT THE WAY TO DO THE ANOMALIES!
        ecg5000_train = pd.read_csv("./datasets/ECG5000/ECG5000_TRAIN", header=None)
        ecg5000_test = pd.read_csv("./datasets/ECG5000/ECG5000_TEST", header=None)
        # ecg5000_data = pd.concat([ecg5000_train,
        # ecg5000_test], ignore_index=True)
        self.train_labels = ecg5000_train[0]
        ecg5000_train.drop([0], inplace=True, axis=1)
        self._train_data = ecg5000_train
        self.test_labels = ecg5000_test[0]
        ecg5000_test.drop([0], inplace=True, axis=1)
        self._test_data = ecg5000_test
