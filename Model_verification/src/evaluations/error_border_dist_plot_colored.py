import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd

from .evaluation import evaluation


class error_border_dist_plot_colored:
    def __init__(
        self, eval_inst: evaluation, name: str = "error_border_dist_plot_colored"
    ):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        sample_dist_pairs = algorithm.assign_border_dists(
            algorithm.module, dataset.test_data()
        )
        sample_error_pairs = algorithm.assign_errors(
            algorithm.module, dataset.test_data()
        )
        sample_color_pairs = algorithm.assign_lin_subfcts_ind(
            algorithm.module, dataset.test_data()
        )
        sample_dist_df = pd.DataFrame(
            sample_dist_pairs, columns=["sample_dist", "dist"]
        )
        sample_error_df = pd.DataFrame(
            sample_error_pairs, columns=["sample_error", "error"]
        )
        sample_color_df = pd.DataFrame(
            sample_color_pairs, columns=["sample_color", "color"]
        )
        merged_df_pre = sample_dist_df.merge(
            sample_error_df, left_index=True, right_index=True
        )
        merged_df = merged_df_pre.merge(
            sample_color_df, left_index=True, right_index=True
        )
        merged_df["checksum"] = (
            merged_df["sample_dist"]
            - merged_df["sample_error"]
            + merged_df["sample_dist"]
            - merged_df["sample_color"]
        )
        if merged_df["checksum"].sum().sum() > 0.001:
            print(
                """Your indices do not overlap. Hence error and dist are not
                    from the same samples"""
            )

        df_sorted = merged_df.sort_values(by=["color", "dist"], ascending=False)
        df_sorted["new_index"] = range(merged_df.shape[0])
        df_sorted.set_index("new_index", inplace=True)
        fig = plt.figure(figsize=[20, 20])
        cmap = matplotlib.cm.get_cmap("tab20")
        for color in df_sorted["color"].unique():
            df_subset = df_sorted[df_sorted["color"] == color]
            rgba_color = cmap(color)
            plt.plot(df_subset.index, df_subset["dist"].values, color=rgba_color)
            plt.plot(df_subset.index, df_subset["error"].values, color="red")

        # save figure
        self.evaluation.save_figure(fig, "error_border_dist_plot_colored")
        plt.close("all")
