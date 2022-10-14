import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product

from .evaluation import evaluation


class subfunc_distmat:
    def __init__(
        self,
        eval_inst: evaluation,
        name: str = "subfunc_distmat",
    ):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        # sample indices
        last_epoch = len(algorithm.lin_sub_fct_Counters)
        linsubfct_final = algorithm.lin_sub_fct_Counters[last_epoch - 1]
        num_funcs = len(linsubfct_final)
        tot_dist = np.zeros(shape=(num_funcs, num_funcs))
        mat_dist = np.zeros(shape=(num_funcs, num_funcs))
        bias_dist = np.zeros(shape=(num_funcs, num_funcs))
        functions = list(map(lambda x: x[0], linsubfct_final))
        function_ind = range(len(functions))
        for func_ind_1, func_ind_2 in product(function_ind, function_ind):
            func1 = functions[func_ind_1]
            func2 = functions[func_ind_2]
            tot_dist[func_ind_1][func_ind_2] = self.calc_func_dist(func1, func2)
            mat_dist[func_ind_1][func_ind_2] = self.calc_mat_dist(
                func1.matrix, func2.matrix
            )
            bias_dist[func_ind_1][func_ind_2] = self.calc_bias_dist(
                func1.bias, func2.bias
            )
        self.evaluation.save_csv(tot_dist, f"tot_dist_epoch_{last_epoch}")
        self.evaluation.save_csv(mat_dist, f"mat_dist_epoch_{last_epoch}")
        self.evaluation.save_csv(bias_dist, f"bias_dist_epoch_{last_epoch}")

    def calc_func_dist(self, func1, func2):
        mat_dist = self.calc_mat_dist(func1.matrix, func2.matrix)
        bias_dist = self.calc_bias_dist(func1.bias, func2.bias)
        return mat_dist + bias_dist

    def calc_mat_dist(self, mat1, mat2):
        mat_diff = mat1 - mat2
        fro_norm = np.linalg.norm(mat_diff, ord="fro")
        return fro_norm

    def calc_bias_dist(dist, bias1, bias2):
        bias_diff = bias1 - bias2
        eucl_norm = np.linalg.norm(bias_diff)
        return eucl_norm
