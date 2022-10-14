import maraboupy
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import time
from maraboupy import Marabou
from maraboupy import MarabouCore
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

from .evaluation import evaluation


class deepoc_adv_derivative:
    def __init__(
        self,
        eval_inst: evaluation,
        name: str = "deepoc_adv_derivative",
        # accuracy wrt distance in input space
    ):
        self.name = name
        self.evaluation = eval_inst

    def get_samples(self, dataset, sampling_method: str, num_points=None):
        if sampling_method == 'random_points':
            sample_list = []
            for point in range(num_points):
                sample = dataset.test_data().sample(1, random_state = point)
                sample_list.append(sample)
            return sample_list
        else:
            raise Exception('could not return points as no method was specified')

    def find_adv_attack(self, input_sample, algorithm):
        input_sample = torch.tensor(input_sample.iloc[0], requires_grad=True)
        output = algorithm.module(input_sample)[0]
        center = algorithm.center
        loss = nn.MSELoss()(output, center)
        algorithm.module.zero_grad()
        loss.backward()

        acc = 0.0001
        bin_search_start = self.find_bin_search_start(input_sample, algorithm)
        adv_sample = self.find_adv_sample(input_sample, algorithm,
                bin_search_start, acc)
        L2_dist = nn.MSELoss()(adv_sample, input_sample)
        L_infty_dist = abs(adv_sample - input_sample).max()
        return adv_sample, L2_dist, L_infty_dist

    def find_bin_search_start(self, input_sample, algorithm):
        found_adv_sample = False
        eps = 1
        center = algorithm.center
        while not found_adv_sample:
            eps = 2 * eps
            adv_sample = input_sample + eps * input_sample.grad
            adv_pred = algorithm.module(adv_sample)[0]
            error = nn.MSELoss()(adv_pred, center)
            if error > algorithm.anom_radius:
                found_adv_sample = True
        return eps

    def find_adv_sample(self, input_sample, algorithm, bin_search_start, acc):
        found_closest_adv_sample = False
        eps = bin_search_start
        eps_change = bin_search_start/2
        center = algorithm.center
        cur_best_adv = None
        anom_rad = algorithm.anom_radius
        while eps_change > acc:
            print(eps)
            adv_cand = input_sample + eps * input_sample.grad
            adv_cand_pred = algorithm.module(adv_cand)[0]
            error = nn.MSELoss()(adv_cand_pred, center)
            if error > anom_rad:
                cur_best_adv = adv_cand
                eps = eps - eps_change
            else:
                eps = eps + eps_change
            eps_change = eps_change/2
        return cur_best_adv

    def calc_cosine_sim(self, input_sample, adv_attack, algorithm):
        output_sample = algorithm.predict(input_sample).values[0]
        border_point = algorithm.calc_border_point(output_sample).values[0]
        adv_attack_im = (algorithm.module(torch.tensor(
            adv_attack.astype(np.float32)))[0]).detach().numpy()
        border_point_center = border_point - algorithm.center.numpy()
        adv_attack_im_center = adv_attack_im - algorithm.center.numpy()
        cos_sim = cosine_similarity(border_point_center[np.newaxis],
                adv_attack_im_center[np.newaxis]).astype(np.float64)[0][0]
        return cos_sim

    # def calc_border_point(self, point, algorithm):
        # center = algorithm.center.numpy()
        # if not isinstance(point, np.ndarray):
            # point = point.values[0]
        # anom_radius = algorithm.anom_radius
        # diff = point - center
        # diff_length = np.sqrt(np.square(diff).sum())
        # border_point = center + anom_radius*(diff/diff_length)
        # return pd.DataFrame(border_point).transpose()


    def evaluate(self, dataset, algorithm):
        collapsing = test_for_collapsing(dataset, algorithm)
        samples = self.get_samples(dataset, 'random_points', num_points=100)
        result_dict = {}
        result_dict_stats = {}
        for i,input_sample in enumerate(samples):
            output_sample = algorithm.predict(input_sample)
            adv_attack, L2_dist, L_infty_dist = self.find_adv_attack(input_sample, algorithm)
            adv_attack = adv_attack.detach().numpy().astype(np.float64)
            # L2_dist = list(L2_dist.detach().numpy().astype(np.float64)[np.newaxis])
            L2_dist = L2_dist.tolist()
            L_infty_dist = L_infty_dist.tolist()
            # L_infty_dist = list(
                    # L_infty_dist.detach().numpy().astype(np.float64)[np.newaxis])
            cos_sim = self.calc_cosine_sim(input_sample, adv_attack, algorithm)
            sample_frac = algorithm.calc_frac_to_border(output_sample).astype(np.float64)
            check_anomaly = algorithm.check_anomalous(
                    pd.DataFrame(adv_attack).transpose()).tolist()
            result_dict[i] = {
                    'input_sample' : list(input_sample.iloc[0]),
                    'adv_attack' : list(adv_attack), 
                    }
            result_dict_stats[i] = {
                    'L2_dist' : L2_dist,
                    'L_infty_dist' : L_infty_dist,
                    'cos_sim': cos_sim,
                    'sample_frac' : sample_frac,
                    'check_anomalous' : check_anomaly 
                    }
        self.evaluation.save_json(result_dict, "deepoc_adv_derivative")
        self.evaluation.save_json(result_dict_stats, "deepoc_adv_derivative_stats")



def test_for_collapsing(dataset, algorithm):
    pred_dataset = algorithm.predict(dataset.test_data())
    if pred_dataset.var().sum() < 0.00001:
        return True
    else:
        return False
