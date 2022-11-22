import numpy as np
import pandas as pd
from .dataset import dataset
from sklearn.datasets import fetch_openml


class page_blocks(dataset):
    def __init__(
        self,
        name: str = "page_blocks",
        file_path: str = None,
        subsample: int = None,
    ):
        super().__init__(name, file_path, subsample)
        self.balance = False
        self.delimiter = ","
        self.header = 0
        self.index_col = 0
        self.label_col_train = 'label'
        self.label_col_test = 'label'
