import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd

from .evaluation import evaluation


class error_border_dist_plot_anomalies:
    def __init__(
        self, eval_inst: evaluation, name: str = "error_border_dist_plot_anomalies"
    ):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        sample_dist_pairs = algorithm.assign_border_dists(
            algorithm.module, dataset.test_data()
        )
        sample_top_k_dist_pairs = algorithm.assign_top_k_border_dists(
            algorithm.module, dataset.test_data(), 5
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
        sample_top_k_dist_df = pd.DataFrame(
            sample_top_k_dist_pairs, columns=["sample_top_k_dist", "top_k_dist"]
        )
        merged_df = sample_dist_df.merge(
            sample_error_df, left_index=True, right_index=True
        )
        merged_tot_df = merged_df.merge(
            sample_top_k_dist_df, left_index=True, right_index=True
        )
        # change by adding from self.train_test_labels
        merged_tot_df["anomaly_label"] = dataset.test_labels
        # merged_tot_df.iloc[-dataset.anomalies.shape[0]:,-1] = 1
        merged_tot_df["checksum"] = merged_df["sample_dist"] - merged_df["sample_error"]
        if merged_tot_df["checksum"].sum().sum() > 0.001:
            print(
                """Your indices do not overlap. Hence error and dist are not
                    from the same samples"""
            )
            print(merged_tot_df["checksum"].sum().sum())

        # one possible solution: split up and sort
        import pdb

        pdb.set_trace()
        merged_tot_df["dist_float"] = merged_tot_df["dist"].map(float)
        df_sorted = merged_tot_df.sort_values(
            by=["anomaly_label", "dist_float"], ascending=False
        )
        df_sorted["new_index"] = range(merged_tot_df.shape[0])
        df_sorted.set_index("new_index", inplace=True)
        fig = plt.figure(figsize=[20, 20])
        cmap = matplotlib.cm.get_cmap("tab20")
        for color in df_sorted["anomaly_label"].unique():
            df_subset = df_sorted[df_sorted["anomaly_label"] == color]
            rgba_color = cmap(color)
            plt.semilogy(df_subset.index, df_subset["dist"].values, color=rgba_color)
            plt.semilogy(df_subset.index, df_subset["error"].values, color="red")
            plt.semilogy(df_subset.index, df_subset["top_k_dist"].values, color="blue")

        # save figure
        self.evaluation.save_figure(fig, "error_border_dist_plot_top_k_anomalies")
        plt.close("all")

    def assign_anomaly_ground_truth(dataset):
        sample_anomaly_pairs = []
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
        )
        for inst_batch in data_loader:
            for inst in inst_batch:
                import pdb

                pdb.set_trace()
        return inst_func_pairs
