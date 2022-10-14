import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd

from .evaluation import evaluation


class error_border_dist_plot:
    def __init__(self, eval_inst: evaluation, name: str = "error_border_dist_plot"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        sample_dist_pairs = algorithm.assign_border_dists(
            algorithm.module, dataset.test_data()
        )
        sample_error_pairs = algorithm.assign_errors(
            algorithm.module, dataset.test_data()
        )
        sample_dist_df = pd.DataFrame(
            sample_dist_pairs, columns=["sample_dist", "dist"]
        )
        sample_error_df = pd.DataFrame(
            sample_error_pairs, columns=["sample_error", "error"]
        )
        merged_df = sample_dist_df.merge(
            sample_error_df, left_index=True, right_index=True
        )
        merged_df["checksum"] = merged_df["sample_dist"] - merged_df["sample_error"]
        if merged_df["checksum"].sum().sum() > 0.001:
            print(
                """Your indices do not overlap. Hence error and dist are not
                    from the same samples"""
            )

        df_sorted = merged_df.sort_values(by=["error"], ascending=False)
        df_sorted["new_index"] = range(merged_df.shape[0])
        df_sorted.set_index("new_index", inplace=True)
        fig = plt.figure(figsize=[20, 20])
        plt.plot(
            df_sorted.index, df_sorted["dist"].values, color="blue", label="border_dist"
        )
        plt.plot(df_sorted.index, df_sorted["error"].values, color="red", label="error")
        plt.legend()
        plt.plot(df_sorted.index, np.zeros(len(df_sorted.index)), color="gray")

        # save figure
        self.evaluation.save_figure(fig, "error_border_dist_plot")
        # import pdb; pdb.set_trace()
        self.evaluation.save_csv(df_sorted[["error", "dist"]], "error_border_dist_plot")
        plt.close("all")
