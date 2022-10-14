import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

from .evaluation import evaluation


class interpolation_error_plot:
    def __init__(self, eval_inst: evaluation, name: str = "interpolation_error_plot"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        dataset.test_data()
        # Code for particular instances
        #        insts = pd.DataFrame()
        #        label_inds = []
        #        for label in dataset.test_labels.unique():
        #            label_ind = dataset.test_labels[dataset.test_labels == label].sample(1).index
        #            label_inds.append(label_ind)
        #            label_inst = dataset.test_data().loc[label_ind]
        #            import pdb; pdb.set_trace()
        #            insts = pd.concat([insts, label_mean], axis=0)
        #        for label_ind1, label_ind2 in list(combinations(label_inds,2)):
        #            label1 = dataset.test_labels[label_ind1].item()
        #            label2 = dataset.test_labels[label_ind2].item()
        #            subfolder = str(label1) + '_' + str(label2)
        #            inst_pair = pd.DataFrame()
        #            inst_pair = pd.concat([inst_pair, insts.loc[label_ind1]])
        #            inst_pair = pd.concat([inst_pair, insts.loc[label_ind2]])
        #            self.evaluate_inst_pair(dataset, algorithm, inst_pair, subfolder)

        # Code for label means
        test_labels = []
        means = []
        for label in dataset.test_labels.unique():
            test_labels.append(label)
            label_data = dataset.test_data().loc[
                dataset.test_labels[dataset.test_labels == label].index
            ]
            label_mean = label_data.mean()
            means.append(label_mean)
        label_mean_tuples = list(zip(test_labels, means))
        for label_mean1, label_mean2 in list(combinations(label_mean_tuples, 2)):
            label1 = label_mean1[0]
            label2 = label_mean2[0]
            mean1 = pd.DataFrame(label_mean1[1]).transpose()
            mean2 = pd.DataFrame(label_mean2[1]).transpose()
            subfolder = str(label1) + "_" + str(label2)
            inst_pair = pd.DataFrame()
            inst_pair = pd.concat([inst_pair, mean1])
            inst_pair = pd.concat([inst_pair, mean2])
            self.evaluate_inst_pair(dataset, algorithm, inst_pair, subfolder)

    def evaluate_inst_pair(self, dataset, algorithm, insts, subfolder):
        interpolations = over_interpolate(insts, 100)
        reconstructions = algorithm.predict(interpolations)
        fig = plt.figure(figsize=[20, 20])
        interp_errors = np.sqrt(
            ((interpolations - reconstructions) ** 2).sum(axis=1).values
        )
        error_mean1 = np.sqrt((reconstructions - insts.iloc[0]) ** 2).sum(axis=1).values
        error_mean2 = np.sqrt((reconstructions - insts.iloc[1]) ** 2).sum(axis=1).values
        plt.plot(interp_errors, label="interpolation_errors")
        plt.plot(error_mean1, label="error_reconstr_label1")
        plt.plot(error_mean2, label="error_reconstr_label2")
        plt.legend()
        self.evaluation.save_figure(fig, "interpolation_errors", subfolder=subfolder)
        self.evaluation.save_csv(
            pd.DataFrame(interp_errors, columns=["interp_errors"]),
            "interpolation_errors",
            subfolder=subfolder,
        )
        plt.close("all")


def interpolate(insts, res_elems):
    if insts.shape[0] != 2:
        raise Exception("Cannot interpolate between more than two elements")

    inst1 = insts.values[0]
    inst2 = insts.values[1]
    interp_steps = res_elems - 1
    interp_np = inst1
    for i in range(1, interp_steps + 1):
        interp_inst = ((interp_steps - i) / interp_steps) * inst1 + (
            i / interp_steps
        ) * inst2
        interp_np = np.vstack((interp_np, interp_inst))
    result_df = pd.DataFrame(interp_np)
    return result_df


def over_interpolate(insts, res_elems):
    if insts.shape[0] != 2:
        raise Exception("Cannot interpolate between more than two elements")

    orig_inst1 = insts.values[0]
    orig_inst2 = insts.values[1]
    # we have 1/3 anomaly path, 1/3 interp and 1/3 anomaly path
    anom_inst1 = 2 * orig_inst1 - orig_inst2
    anom_inst2 = 2 * orig_inst2 - orig_inst1
    interp_steps = res_elems - 1
    interp_np = anom_inst1
    for i in range(1, interp_steps + 1):
        interp_inst = ((interp_steps - i) / interp_steps) * anom_inst1 + (
            i / interp_steps
        ) * anom_inst2
        interp_np = np.vstack((interp_np, interp_inst))
    result_df = pd.DataFrame(interp_np)
    return result_df
