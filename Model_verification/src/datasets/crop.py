import numpy as np
import re
import random
import pandas as pd
from .dataset import dataset
from sklearn.datasets import fetch_openml


class crop(dataset):
    def __init__(
        self,
        name: str = "crop",
        class1: int = None,
        class2: int = None,
        file_path: str = None,
        subsample: int = None,
    ):
        self.class1 = class1
        self.class2 = class2
        super().__init__(name, file_path, subsample)

    def create(self):
        dataset_train = pd.read_csv(
            "./datasets/Crop/Crop_TRAIN",
            header=None,
            delimiter="\t",
        )
        dataset_test = pd.read_csv(
            "./datasets/Crop/Crop_TEST",
            header=None,
            delimiter="\t",
        )
        # electricDevices_data = pd.concat([electricDevices_train,

        # if sublabel are to be chosen
        num_list = re.findall(r"\d+", self.name)
        if len(num_list) == 2:
            int1 = int(num_list[0])
            int2 = int(num_list[1])
        else:
            int1 = self.class1
            int2 = self.class2
        #            rand_ind_1 = random.randint(0,24)
        #            rand_ind_2 = random.randint(0,24)
        #            while rand_ind_2 == rand_ind_1:
        #                rand_ind_2 = random.randint(0,24)
        #            int1 = rand_ind_1
        #            int2 = rand_ind_2
        if int1 is not None and int2 is not None:
            if int1 == int2:
                raise Exception("No two same classes are allowed")
            dataset_train = pd.concat(
                [
                    dataset_train[dataset_train[0] == int1],
                    dataset_train[dataset_train[0] == int2],
                ]
            )
            dataset_test = pd.concat(
                [
                    dataset_test[dataset_test[0] == int1],
                    dataset_test[dataset_test[0] == int2],
                ]
            )
            self.name = self.name + "_" + str(int1) + "_" + str(int2)

        self.train_labels = dataset_train[0]
        dataset_train.drop([0], inplace=True, axis=1)
        self._train_data = dataset_train
        self.test_labels = dataset_test[0]
        dataset_test.drop([0], inplace=True, axis=1)
        self._test_data = dataset_test
