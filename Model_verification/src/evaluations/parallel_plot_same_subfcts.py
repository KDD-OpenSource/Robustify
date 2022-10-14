import pandas as pd
import torch
import ast
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from .evaluation import evaluation
from ..algorithms.neural_net import smallest_k_dist_loss


class parallel_plot_same_subfcts:
    def __init__(self, eval_inst: evaluation, name: str =
            "parallel_plot_same_subfcts"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        # for every subfct plot all testpoints of that subfct in a seperate
        # plot
        test_data = dataset.test_data()
        test_labels = dataset.test_labels

        normal_labels = test_labels.unique()[test_labels.unique()>=0]
        fig, ax = plt.subplots(len(normal_labels),1,figsize = (20,10))
        if 'scales' in dataset.__dict__.keys():
            scales = ast.literal_eval(dataset.scales)
        for ind, label in enumerate(normal_labels):
            label_data = test_data[test_labels==label]
            min_avg_border_dist = algorithm.calc_min_avg_border_dist(label_data,
                    subsample = min(label_data.shape[0], 1000))

            inst_fct_pairs = algorithm.assign_lin_subfcts(algorithm.module,
                    label_data)
            inst_fct_pairs_data = list(map(lambda x:(np.array(x[0].tolist()), x[1]),
                inst_fct_pairs))
            inst_fct_df = pd.DataFrame(inst_fct_pairs_data, columns = ['insts', 'fct'])
            fct_means = inst_fct_df.groupby(['fct'],
                    sort=False).insts.apply(np.mean)
            fct_means.rename('means', inplace=True)
            fct_counts = inst_fct_df.groupby(['fct'], sort=False).insts.count()
            fct_counts.rename('count', inplace=True)
            fct_info = pd.concat([fct_means, fct_counts], axis=1)
            for idx, _ in enumerate(fct_info.index):
                thickness = fct_info.iloc[idx]['count']/inst_fct_df.shape[0]
                ax[ind].plot(fct_info.iloc[idx]['means'], linewidth = thickness)
            if 'scales' in dataset.__dict__.keys():
                ax[ind].set_title(f'''Num_fcts: {len(fct_means)}, label: {label},
                    min_avg_border_dist: {min_avg_border_dist}, Scale:
                    {scales[ind]}''')
            else:
                ax[ind].set_title(f'''Num_fcts: {len(fct_means)}, label: {label},
                        min_avg_border_dist: {min_avg_border_dist}''')
        self.evaluation.save_figure(fig, 'parallel_plot_subfct_means')
