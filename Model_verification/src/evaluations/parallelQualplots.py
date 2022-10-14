import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import evaluation


class parallelQualplots:
    def __init__(
        self, eval_inst: evaluation, name: str = "parallelQualplot", num_plots: int = 20
    ):
        self.name = name
        self.evaluation = eval_inst
        self.num_plots = num_plots
        self.plots = []

    def evaluate(self, dataset, algorithm):
        # input_points: pd.DataFrame,
        # output_points: pd.DataFrame):
        # sample indices
        input_points = dataset.test_data()
        output_points = algorithm.predict(input_points)
        for label in dataset.test_labels.unique():
            label_indices = (
                dataset.test_labels[dataset.test_labels == label].sample(10).index
            )
            label_data = dataset.test_data().loc[
                dataset.test_labels[dataset.test_labels == label].index
            ]
            label_mean = label_data.mean()
            for ind in label_indices:
                fig = plt.figure(figsize=(20, 10))
                mean_squared_error = (
                    (input_points.loc[ind, :] - output_points.loc[ind, :]) ** 2
                ).sum() / input_points.shape[1]
                dist_to_mean = (
                    (input_points.loc[ind, :] - label_mean) ** 2
                ).sum() / input_points.shape[1]
                # change limits to -1, 1
                plt.ylim(-1, 1)
                plt.plot(input_points.loc[ind, :], color="blue", label="Orig")
                plt.plot(output_points.loc[ind, :], color="orange", label="Reconstr")
                plt.plot(label_mean, label="label_mean")
                plt.legend()
                plt.title(
                    f"""MSE: {mean_squared_error}; Dist_to_mean:
                        {dist_to_mean}"""
                )
                self.evaluation.save_figure(
                    fig, "parallelPlot_" + str(label) + "_" + str(ind)
                )
                plt.close("all")
