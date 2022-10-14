import matplotlib.pyplot as plt
import matplotlib.lines as ln
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd

from .evaluation import evaluation


class inst_area_2d_plot:
    def __init__(
        self, eval_inst: evaluation, name: str = "inst_area_2d_plot",
        num_points=20000
    ):
        self.name = name
        self.evaluation = eval_inst
        self.num_points = num_points

    def evaluate(self, dataset, algorithm):
        input_dim = algorithm.topology[0]
        if input_dim != 2:
            raise Exception("cannot plot in 2d unless input dim is 2d too")
        fig = plt.figure(figsize=[20, 20])

        # plot background points
        randPoints = pd.DataFrame(
            np.random.uniform(low=-1, high=1, size=(self.num_points, input_dim))
        )
        inst_func_pairs = algorithm.assign_lin_subfcts_ind(algorithm.module, randPoints)
        points = pd.DataFrame(map(lambda x: x[0], inst_func_pairs))
        colors = pd.DataFrame(map(lambda x: x[1], inst_func_pairs), columns=[2])
        num_colors = len(colors[2].unique())
        joined = pd.concat([points, colors], axis=1)
        fig.suptitle(f"Number of colors is {num_colors}")
        plt.scatter(joined[0], joined[1], c=joined[2], alpha=0.5, cmap="tab20")
        #plt.scatter(joined[0], joined[1], c=joined[2], alpha=0.5)
        # plot element from each 'color'
        color_repr = []
        for elem in joined[2].unique():
            color_repr.append(pd.DataFrame(joined[joined[2] == elem].iloc[0]))
        joined_color_repr = pd.concat(color_repr, axis=1).transpose()
        plt.scatter(joined_color_repr[0], joined_color_repr[1], c="blue")
        
        # save data
        points_np = pd.DataFrame(map(lambda x:x[0].numpy(), inst_func_pairs))
        joined_np = pd.concat([points_np, colors], axis=1)
        self.evaluation.save_csv(joined_np, f'scatter_2d_bound_data_{num_colors}')


        # plot boundaries
#        inst = torch.tensor(dataset.test_data().values[0].astype(np.float32))
#        plt.scatter(inst[0], inst[1], color="red", s=250)
##
        # plot trainset
        #train_points = dataset.train_data()
        #plt.scatter(train_points.values[:,0], train_points.values[:,1], color='red')
#        crosspoints = algorithm.get_all_funcBoundaries(algorithm.module, inst)
#        #area_fcts = algorithm.get_fct_area(algorithm.module, inst)
#        edge_points = []
#        for cross in crosspoints:
#            edge_points.append(get_edge_points(inst, cross))
#        
#        for edge_point in edge_points:
#            try:
#                plt.plot([edge_point[0][0], edge_point[1][0]], [edge_point[0][1],
#                    edge_point[1][1]])
#            except:
#                pass
            #fig.add_artist(ln.Line2D([0,1],[0,1]),
                    #clip_box = [[-1,1],[-1,1]])

            #import pdb; pdb.set_trace()
            #fig.add_artist(
            #    ln.Line2D(np.array(edge_point)[0],
            #        np.array(edge_point)[1]))

#        boundary_points_filtered = list(
#            filter(
#                lambda point: torch.all(point < 1) and torch.all(point > -1),
#                boundary_points,
#            )
#        )
#        x_points = list(map(lambda x: x[0], boundary_points_filtered))
#        y_points = list(map(lambda x: x[1], boundary_points_filtered))
#        plt.scatter(x_points[:5], y_points[:5], color="blue", s=200)
#        plt.scatter(x_points[5:], y_points[5:], color="blue", s=200)

        self.evaluation.save_figure(fig, "scatter_2d_boundaries")
        plt.close("all")

def get_edge_points(inst, cross):
    points = []
    # we calculate four cases
    # edge_1 = 1 -> edge_2 in [-1,1]
    edge_2 = float(-((inst[0] - cross[0])*(1-cross[0]))/(inst[1]-cross[1]) +
            cross[1])
    if edge_2 >= -1 and edge_2 <= 1:
        points.append([1, edge_2])

    #edge_1 = -1 -> edge_2 in [-1,1]
    edge_2 = float(-((inst[0] - cross[0])*(-1-cross[0]))/(inst[1]-cross[1]) +
            cross[1])
    if edge_2 >= -1 and edge_2 <= 1:
        points.append([-1, edge_2])

    #edge_2 = 1 -> edge_1 in [-1,1]
    edge_1 = float(-((1-cross[1])*(inst[1] - cross[1]))/(inst[0]-cross[0]) +
            cross[0])
    if edge_1 >= -1 and edge_1 <= 1:
        points.append([edge_1, 1])

    #edge_2 = -1 -> edge_1 in [-1,1]
    edge_1 = float(-((-1-cross[1])*(inst[1] - cross[1]))/(inst[0]-cross[0]) +
            cross[0])
    if edge_1 >= -1 and edge_1 <= 1:
        points.append([edge_1, -1])
    return points
