import numpy as np
import pandas as pd
from .dataset import dataset
from src.utils.utils import get_proj_root


class mnist(dataset):
    def __init__(
        self,
        name: str = "mnist",
        file_path: str = None,
        subsample: int = None,
        scale: bool = True,
        num_samples: int = 2000,
        num_anomalies: int = 20,
        normal_class: int = None,
    ):
        super().__init__(name, file_path, subsample)
        self.num_samples = num_samples
        self.num_anomalies = num_anomalies
        self.scale = scale
        self.normal_class = normal_class

    def create(self):
        root = get_proj_root()
        dataset_train = pd.read_csv(str(root) + "/datasets/mnist/mnist_train.csv")
        dataset_test = pd.read_csv(str(root) + "/datasets/mnist/mnist_test.csv")
        if self.normal_class is None:
            self.prep_all_classes(dataset_train, dataset_test)
        else:
            self.prep_normal_class(dataset_train, dataset_test)

    def prep_all_classes(self, dataset_train, dataset_test):
        self.train_labels = dataset_train["label"]
        dataset_train.drop(["label"], inplace=True, axis=1)
        self._train_data = dataset_train
        self.test_labels = dataset_test["label"]
        dataset_test.drop(["label"], inplace=True, axis=1)
        self._test_data = dataset_test

    def prep_normal_class(self, dataset_train, dataset_test):
        dataset_train = dataset_train[dataset_train.label == self.normal_class]
        self.train_labels = dataset_train["label"]
        dataset_train = dataset_train.drop(["label"], axis=1)
        self._train_data = dataset_train
        self.test_labels = dataset_test["label"]
        self.test_labels[self.test_labels != self.normal_class] = -1
        dataset_test.drop(["label"], inplace=True, axis=1)
        self._test_data = dataset_test
