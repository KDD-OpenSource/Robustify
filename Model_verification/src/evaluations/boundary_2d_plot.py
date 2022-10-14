import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd

from .evaluation import evaluation


class boundary_2d_plot:
    def __init__(
        self, eval_inst: evaluation, name: str = "boundary_2d_plot", num_points=20000
    ):
        self.name = name
        self.evaluation = eval_inst
        self.num_points = num_points

    def evaluate(self, dataset, algorithm):
        input_dim = algorithm.topology[0]
        if input_dim != 2:
            raise Exception("cannot plot in 2d unless input dim is 2d too")

        # plot background points
        randPoints = pd.DataFrame(
            np.random.uniform(low=-1, high=1, size=(self.num_points, input_dim))
        )
        inst_func_pairs = algorithm.assign_lin_subfcts_ind(algorithm.module, randPoints)
        points = pd.DataFrame(map(lambda x: x[0], inst_func_pairs))
        colors = pd.DataFrame(map(lambda x: x[1], inst_func_pairs), columns=[2])
        num_colors = len(colors[2].unique())
        joined = pd.concat([points, colors], axis=1)
        fig = plt.figure(figsize=[20, 20])
        fig.suptitle(f"Number of colors is {num_colors}")
        plt.scatter(joined[0], joined[1], c=joined[2], alpha=0.5, cmap="tab20")
        # plot element from each 'color'
        color_repr = []
        for elem in joined[2].unique():
            color_repr.append(pd.DataFrame(joined[joined[2] == elem].iloc[0]))
        joined_color_repr = pd.concat(color_repr, axis=1).transpose()
        plt.scatter(joined_color_repr[0], joined_color_repr[1], c="blue")

        # map anomalies and points
        data_loader = DataLoader(
            dataset=dataset.anomalies.values,
            drop_last=False,
            pin_memory=True,
        )
        mapped_anomalies = []
        for inst in data_loader:
            inst = inst.float()
            mapped_anomalies.append(algorithm.module(inst)[0])
        mapped_anomalies = np.array(
            list(map(lambda x: x.detach().numpy().flatten(), mapped_anomalies))
        )
        data_loader = DataLoader(
            dataset=dataset.test_data().values,
            drop_last=False,
            pin_memory=True,
        )
        mapped_points = []
        for inst in data_loader:
            inst = inst.float()
            mapped_points.append(algorithm.module(inst)[0])
        mapped_points = np.array(
            list(map(lambda x: x.detach().numpy().flatten(), mapped_points))
        )

        # plot points and anomalies
        plt.scatter(dataset.test_data()[0], dataset.test_data()[1], color="gray")
        # plt.scatter(mapped_anomalies[:,0], mapped_anomalies[:,1],
        #        color='black', s = 250)
        plt.scatter(mapped_points[:, 0], mapped_points[:, 1], color="green")
        for point1, point2 in zip(dataset.anomalies.values, mapped_anomalies):
            point_pair = np.vstack([point1, point2]).transpose()
            plt.plot(point_pair[0], point_pair[1])

        # plot boundaries
        inst = torch.tensor(dataset.test_data().values[0])
        plt.scatter(inst[0], inst[1], color="red", s=250)
        boundary_points = algorithm.get_all_funcBoundaries(algorithm.module, inst)
        boundary_points_filtered = list(
            filter(
                lambda point: torch.all(point < 1) and torch.all(point > -1),
                boundary_points,
            )
        )
        x_points = list(map(lambda x: x[0], boundary_points_filtered))
        y_points = list(map(lambda x: x[1], boundary_points_filtered))
        plt.scatter(x_points[:5], y_points[:5], color="blue", s=200)
        plt.scatter(x_points[5:], y_points[5:], color="blue", s=200)

        # save figure
        self.evaluation.save_figure(fig, "scatter_2d_boundaries")
        plt.close("all")
        area_fcts = algorithm.get_fct_area(algorithm.module, inst)
