import abc
import os
import json
import torch
import csv
import pandas as pd
import numpy as np
from src.utils.utils import get_proj_root
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors


class dataset:
    def __init__(
        self,
        name: str,
        file_path: str,
        subsample: int = None,
        scale=True,
        scale_type="MinMax",
        scale_min=0,
        scale_max=1,
    ):
        self.name = name
        self.file_path = file_path
        self.subsample = subsample
        self._train_data = None
        self._test_data = None
        self.scale = scale
        self.scale_type = scale_type
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __str__(self) -> str:
        return self.name

    def create(self):
        """create dataset from file"""
        root = get_proj_root()
        train_file = "/datasets/" + self.name + "/" + self.name + "_TRAIN.csv"
        test_file = "/datasets/" + self.name + "/" + self.name + "_TEST.csv"

        dataset_train = pd.read_csv(
            str(root) + train_file,
            header=self.header,
            delimiter=self.delimiter,
            index_col=self.index_col,
        )
        dataset_test = pd.read_csv(
            str(root) + test_file,
            header=self.header,
            delimiter=self.delimiter,
            index_col=self.index_col,
        )

        if hasattr(self, "balance"):
            dataset = self.join_and_shuffle(dataset_train, dataset_test)
            dataset_train, dataset_test = self.balance_split(dataset)

        if self.label_col_train is None:
            self.train_labels = pd.Series(0, index=dataset_train.index).astype(
                np.float32
            )
        else:
            self.train_labels = dataset_train.loc[:, self.label_col_train].astype(
                np.float32
            )
            dataset_train = dataset_train.drop([self.label_col_train], axis=1)
        self._train_data = dataset_train.astype(np.float32)
        self.test_labels = dataset_test[self.label_col_test].astype(np.float32)
        dataset_test = dataset_test.drop([self.label_col_test], axis=1)
        self._test_data = dataset_test.astype(np.float32)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        joined_df_train = self.train_data().merge(
            pd.DataFrame(self.train_labels.rename("label"), columns=["label"]),
            left_index=True,
            right_index=True,
        )
        joined_df_train.to_csv(path + "/" + self.name + "_train.csv")
        joined_df_test = self.test_data().merge(
            pd.DataFrame(self.test_labels.rename("label"), columns=["label"]),
            left_index=True,
            right_index=True,
        )
        joined_df_test.to_csv(path + "/" + self.name + "_test.csv")
        self_dict = self.__dict__.copy()
        self_dict.pop("_train_data")
        self_dict.pop("_test_data")
        self_dict.pop("train_labels")
        self_dict.pop("test_labels")
        with open(path + "/" + self.name + "_properties.json", "w") as json_file:
            json.dump(self_dict, json_file, indent=4)

    def load_used_ds(self):
        """load dataset which has been saved based on save function"""
        file_path = self.file_path
        for dataset_file in os.listdir(file_path):
            if "properties.json" in dataset_file:
                with open(os.path.join(self.file_path, dataset_file)) as json_file:
                    self_dict = json.load(json_file)
                for key, value in self_dict.items():
                    self.__dict__[key] = value
            elif "train.csv" in dataset_file:
                data_df = pd.read_csv(
                    os.path.join(file_path, dataset_file), index_col=0
                ).astype(np.float32)
                self.train_labels = data_df["label"]
                data_df.drop(["label"], axis=1, inplace=True)
                self._train_data = data_df
            elif "test.csv" in dataset_file:
                data_df = pd.read_csv(
                    os.path.join(file_path, dataset_file), index_col=0
                ).astype(np.float32)
                self.test_labels = data_df["label"]
                data_df.drop(["label"], axis=1, inplace=True)
                self._test_data = data_df
            elif "readme.md" in dataset_file:
                pass
            else:
                print("No appropriate keyword in datafile")

    def preprocess(self):
        self._train_data = self._train_data.astype(np.float32)
        self._test_data = self._test_data.astype(np.float32)
        if self.subsample:
            self._train_data = self._train_data.sample(self.subsample)
            self.train_labels = self.train_labels[self._train_data.index]
            self.train_labels = pd.Series(self.train_labels)
        if self.scale == True:
            train_data, scaler = self.scale_train_data(
                self._train_data, return_scaler=True
            )
            self._train_data = train_data
            test_data = pd.DataFrame(
                scaler.transform(self._test_data),
                columns=self._test_data.columns,
                index=self._test_data.index,
            )
            self._test_data = test_data

    def kth_nearest_neighbor_dist(self, k):
        # delete for code publication of submission?
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(self.train_data())
        dist, ind = neigh.kneighbors(self.train_data())
        # calculates the distance to the kth next neighbor (not the k nearest
        # neighbors)
        dists = [dist[i][k - 1] for i in range(len(dist))]
        # dists is a vector of length given by self.train_data().shape[0]
        return dists

    def kth_nearest_neighbor_model(self, k):
        # delete for code publication of submission?
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(self.train_data())
        return neigh

    def get_last_nearest_neighbor_dist(self, neigh, instance):
        # delete for code publication of submission?
        # instance is supposed to be a single sample
        dist, ind = neigh.kneighbors(instance.reshape(1, -1))
        return dist.flatten()[-1]

    def get_nearest_neighbor_insts(self, neigh, instance):
        # delete for code publication of submission?
        # instance is supposed to be a single sample
        dist, ind = neigh.kneighbors(instance.reshape(1, -1))
        res = torch.tensor(self.train_data().loc[ind.flatten()[1:]].values)
        return res

    def scale_train_data(self, data: pd.DataFrame, return_scaler=False):
        if self.scale_type == "MinMax":
            scaler = MinMaxScaler(feature_range=(self.scale_min, self.scale_max))
        elif self.scale_type == "centered":  # x - mean(x)/ max_value
            scaler = CenteredMaxAbsScaler()

        data_index = data.index
        data_columns = data.columns
        scaler.fit(data)
        if not return_scaler:
            return pd.DataFrame(
                scaler.transform(data), columns=data_columns, index=data_index
            )
        else:
            return (
                pd.DataFrame(
                    scaler.transform(data), columns=data_columns, index=data_index
                ),
                scaler,
            )

    def calc_label_means(self, subset):
        label_means = {}
        if subset == "train":
            if self._train_data is None:
                self.train_data()
            for label in self.train_labels.unique():
                label_mean = self.train_data().loc[self.train_labels == label].mean()
                label_means[label] = label_mean
        elif subset == "test":
            if self._test_data is None:
                self.test_data()
            for label in self.test_labels.unique():
                label_mean = self.test_data().loc[self.test_labels == label].mean()
                label_means[label] = label_mean
        else:
            raise Exception("Not yet implemented")
        return label_means

    def calc_dists_to_label_mean(self, subset):
        if subset == "train":
            self.train_data()
            dists_to_label_mean = pd.Series(0, self.train_labels.index)
            label_means = self.calc_label_means("train")
            for label in self.train_labels.unique():
                label_mean = label_means[label]
                diffs = self.train_data().loc[self.train_labels == label] - label_mean
                dists = ((diffs**2).sum(axis=1)) ** (1 / 2)
                dists_to_label_mean.loc[self.train_labels == label] = dists
        elif subset == "test":
            self.test_data()
            self.dists_to_label_mean = pd.Series(0, self.test_labels.index)
            label_means = self.calc_label_means("test")
            for label in self.test_labels.unique():
                label_mean = label_means[label]
                diffs = self.test_data().loc[self.test_labels == label] - label_mean
                dists = ((diffs**2).sum(axis=1)) ** (1 / 2)
                dists_to_label_mean.loc[self.test_labels == label] = dists
        else:
            raise Exception("Not yet implemented")
        return dists_to_label_mean

    def train_data(self):
        if self._train_data is None:
            if self.file_path is None:
                self.create()
                self.preprocess()
            else:
                self.load_used_ds()
        return self._train_data

    def test_data(self):
        if self._test_data is None:
            if self.file_path is None:
                self.create()
                self.preprocess()
            else:
                self.load_used_ds()
        return self._test_data

    # old code
    #        # note that we never create test data (this must happen when the
    #        # traindata is created as it should follow its distribution
    #        if self._test_data is None:
    #            self.load_used_ds()
    #            # we do not preprocess any more. Preprocessing is done during
    #            # training as it is based on training data's values
    #            # self.preprocess()
    #        return self._test_data

    def get_anom_labels_from_test_labels(self):
        # returns 1 for anomalies, 0 for normal data
        anom_series = (self.test_labels < 0).astype(int)
        return anom_series

    def join_and_shuffle(self, dataset_train, dataset_test):
        dataset_train = dataset_train.sample(frac=1)
        dataset_test = dataset_test.sample(frac=1)
        dataset = pd.concat([dataset_train, dataset_test])
        return dataset

    def balance_split(self, dataset):
        tot_num_samples = dataset.shape[0]
        train_num_samples = int(tot_num_samples / 2)
        test_num_samples = int(tot_num_samples / 2)
        dataset_train = dataset[:train_num_samples]
        dataset_test = dataset[train_num_samples : train_num_samples + test_num_samples]
        return dataset_train, dataset_test


class CenteredMaxAbsScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.max_abs_ = None
        self.mean_ = None

    def fit(self, X, y=None):
        self.max_abs_ = X.max().max()
        self.mean_ = X.mean()
        return self

    def transform(self, X):
        return (X - self.mean_) / self.max_abs_
