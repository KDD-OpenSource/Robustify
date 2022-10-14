import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd

from .evaluation import evaluation


class bias_feature_imp:
    def __init__(self, eval_inst: evaluation, name: str = "bias_feature_imp"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        imp_results = algorithm.assign_bias_feature_imps(
            algorithm.module, dataset.test_data()
        )
        imp_results_pd = pd.DataFrame(
            imp_results, columns=["sample", "sum_imp", "feature_imp", "bias_imp"]
        ).drop("sample", axis=1)
        imp_results_pd["anomaly_label"] = dataset.test_labels
        # drop anomalies
        imp_results_pd = imp_results_pd[imp_results_pd.anomaly_label != -1]
        try:
            imp_results_sorted = imp_results_pd.sort_values(
                ["anomaly_label", "sum_imp"], ascending=False
            )
        except:
            imp_results_sorted = imp_results_pd.sort_values(
                ["anomaly_label"], ascending=False
            )

        imp_results_sorted["new_index"] = range(imp_results_sorted.shape[0])
        imp_results_sorted.set_index("new_index", inplace=True)
        fig = plt.figure(figsize=[20, 20])
        cmap = matplotlib.cm.get_cmap("tab20")
        for color in imp_results_sorted["anomaly_label"].unique():
            df_subset = imp_results_sorted[imp_results_sorted["anomaly_label"] == color]
            rgba_color = cmap(int(color))
            plt.plot(
                df_subset.index,
                df_subset["sum_imp"].values,
                color=rgba_color,
                label=f"sum_imp_{color}",
            )
            plt.plot(
                df_subset.index,
                df_subset["feature_imp"].values,
                color="green",
                #label=f"feature_imp_{color}",
            )
            plt.plot(
                df_subset.index,
                df_subset["bias_imp"].values,
                color="red",
                #label=f"bias_imp_{color}",
            )
        plt.legend()
        # save figure
        self.evaluation.save_figure(fig, "bias_feature_imp_plot")
        plt.close("all")
