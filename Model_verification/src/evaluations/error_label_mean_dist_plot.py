import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd

from .evaluation import evaluation


class error_label_mean_dist_plot:
    def __init__(self, eval_inst: evaluation, name: str = "error_label_mean_dist_plot"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        dataset.calc_dist_to_label_mean(subset="test")
        #        sample_dist_pairs = algorithm.assign_border_dists(algorithm.module,
        #                dataset.test_data())
        errors = algorithm.calc_errors(algorithm.module, dataset.test_data())
        joined_df = pd.concat(
            [
                pd.DataFrame(
                    dataset.dists_to_label_mean, columns=["dists_to_label_mean"]
                ),
                pd.DataFrame(errors, columns=["errors"]),
            ],
            axis=1,
        )
        df_sorted = joined_df.sort_values(by=["dists_to_label_mean"], ascending=False)
        df_sorted["new_index"] = range(joined_df.shape[0])
        df_sorted.set_index("new_index", inplace=True)
        fig = plt.figure(figsize=[20, 20])
        # plt.plot(df_sorted.index, df_sorted['dists_to_label_mean'].values, color =
        plt.semilogy(
            df_sorted.index,
            df_sorted["dists_to_label_mean"].values,
            color="blue",
            label="dists_to_label_mean",
        )
        # plt.plot(df_sorted.index, df_sorted['errors'].values, color = 'red',
        plt.semilogy(
            df_sorted.index, df_sorted["errors"].values, color="red", label="errors"
        )
        # plt.plot(df_sorted.index, np.zeros(len(df_sorted.index)), color='gray')
        plt.semilogy(df_sorted.index, np.zeros(len(df_sorted.index)), color="gray")
        plt.ylim(0.0001, 10)
        plt.legend()

        self.evaluation.save_csv(df_sorted, "error_label_mean_dist_data")
        # save figure
        self.evaluation.save_figure(fig, "error_label_mean_dist_plot")
        plt.close("all")
