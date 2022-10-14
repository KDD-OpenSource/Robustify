import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from .evaluation import evaluation
from ..algorithms.neural_net import smallest_k_dist_loss


class avg_min_fctborder_dist:
    def __init__(self, eval_inst: evaluation, name: str =
            "avg_min_fctborder_dist"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        avg_min_fctborder_dist_before = algorithm.calc_min_avg_border_dist(
                dataset.test_data())
        algorithm.push_closest_fctborders_set(dataset.test_data(),
                avg_min_fctborder_dist_before)
        avg_min_fctborder_dist_after = algorithm.calc_min_avg_border_dist(
                dataset.test_data())
        result_dict = {}
        result_dict[
            'avg_min_fctborder_dist_before_push'] = avg_min_fctborder_dist_before
        result_dict[
            'avg_min_fctborder_dist_after_push'] = avg_min_fctborder_dist_after
        self.evaluation.save_json(result_dict, 'avg_min_fctborder_dist')
