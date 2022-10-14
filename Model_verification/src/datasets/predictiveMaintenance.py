import numpy as np
import pandas as pd
from .dataset import dataset
from sklearn.datasets import fetch_openml


class predictiveMaintenance(dataset):
    def __init__(
        self,
        name: str = "predictiveMaintenance",
        file_path: str = None,
        subsample: int = None,
        num_samples: int = 2000,
        num_anomalies: int = 20,
        window_size: int = 50,
    ):
        super().__init__(name, file_path, subsample)
        self.num_samples = num_samples
        self.num_anomalies = num_anomalies
        self.window_size = window_size

    def create(self):
        """
        creates a synthetic DS by projecting a uniform cloud with unit
        width into a high dim space
        """
        predMaintenance_data = pd.read_csv(
            "./datasets/predictiveMaintenance/ai4i2020.csv"
        )
        predMaintenance_data.drop(
            ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"],
            inplace=True,
            axis=1,
        )
        # take subsequent values -> sample index and do index: index +
        if self.num_samples != -1:
            length = predMaintenance_data.shape[0]
            rand_ind = np.random.randint(length - self.window_size - self.num_samples)
            # self.num_samples
            predMaintenance_data = predMaintenance_data.iloc[
                rand_ind : rand_ind + self.num_samples
            ]
        predMaintenance_data["Type"].replace("L", 0, inplace=True)
        predMaintenance_data["Type"].replace("M", 1, inplace=True)
        predMaintenance_data["Type"].replace("H", 2, inplace=True)

        one_sensor_data = predMaintenance_data["Air temperature [K]"]

        self.num_sensors = 6
        time_windows = self.create_time_windows(one_sensor_data)

        max_per_col = time_windows.max().values
        min_per_col = time_windows.min().values
        anomalies = np.random.uniform(
            low=min_per_col,
            high=max_per_col,
            size=(self.num_anomalies, time_windows.shape[1]),
        )
        data = np.vstack([time_windows, anomalies])
        self._train_data = data
        self.train_labels = pd.Series(0, range(self.num_samples))
        anom_labels = pd.Series(-1, range(self.num_anomalies))
        self.train_labels = self.train_labels.append(anom_labels, ignore_index=True)

    def create_time_windows(self, data: pd.DataFrame):
        data_values = data.values
        result = pd.DataFrame()
        for time_ind in range(data_values.shape[0] - self.window_size + 1):
            window_values = data_values[
                time_ind : time_ind + self.window_size
            ].transpose()
            flattened_window = window_values.flatten()
            result = pd.concat(
                [result, pd.DataFrame(flattened_window).transpose()], axis=0
            )

        return result
