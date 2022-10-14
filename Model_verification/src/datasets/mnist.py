import numpy as np
import pandas as pd
from .dataset import dataset
from sklearn.datasets import fetch_openml


class mnist(dataset):
    def __init__(
        self,
        name: str = "mnist",
        file_path: str = None,
        subsample: int = None,
        scale: bool = True,
        num_samples: int = 2000,
        num_anomalies: int = 20,
    ):
        super().__init__(name, file_path, subsample)
        self.num_samples = num_samples
        self.num_anomalies = num_anomalies
        self.scale = scale

    def create(self):
        # simple version
        dataset_train = pd.read_csv("./datasets/mnist/mnist_train.csv")
        dataset_test = pd.read_csv("./datasets/mnist/mnist_test.csv")
        # complex version
        dataset_train = dataset_train[dataset_train.label == 7]
        self.train_labels = dataset_train['label']
        dataset_train.drop(['label'], inplace=True, axis=1)
        self._train_data = dataset_train

        self.test_labels = dataset_test['label']
        dataset_test.drop(['label'], inplace=True, axis=1)
        self._test_data = dataset_test

#        mnist_data_file_tot = pd.concat([mnist_data_file_train, mnist_data_file_test])
#        mnist_data_file_tot["new_index"] = range(mnist_data_file_tot.shape[0])
#        mnist_data_file_tot.set_index("new_index", inplace=True)
#
#        # mnist_data_file_tot.drop('label', inplace=True, axis=1)
#        mnist_data = mnist_data_file_tot.sample(n=self.num_samples)
#        # self.train_labels = mnist_data['label']
#        self.train_labels = pd.DataFrame(mnist_data["label"])
#        self.train_labels["ind"] = range(self.train_labels.shape[0])
#        self.train_labels.set_index("ind", inplace=True)
#        self.train_labels = self.train_labels["label"]
#        anomaly_labels = pd.Series(-1, range(self.num_anomalies))
#        self.train_labels = self.train_labels.append(anomaly_labels, ignore_index=True)
#        mnist_data.drop("label", inplace=True, axis=1)
#        anomalies = np.random.uniform(
#            low=0, high=255, size=(self.num_anomalies, mnist_data.shape[1])
#        )
#        data = np.vstack([mnist_data, anomalies])
#        # data = data.astype(np.float32)
#        data = pd.DataFrame(data)
#        self._train_data = data
        # data = self.scale_data(data, min_val = -1, max_val=1)
        # anomaly_labels = pd.Series(-1, range(self.num_samples,
        # self.num_samples + self.num_anomalies))

        # add label '-1' for anomalous points
        # self.anomalies = pd.DataFrame(data.iloc[-self.num_anomalies :])
        # self.normal_data = pd.DataFrame(data.iloc[:-self.num_anomalies])
