import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from matplotlib import animation
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from .evaluation import evaluation


class tsne_latent:
    def __init__(self, eval_inst: evaluation, name: str = "tsne_latent", tsne_dim=2):
        self.name = name
        self.evaluation = eval_inst
        self.tsne_dim = tsne_dim

    def evaluate(self, dataset, algorithm):
        # import pdb; pdb.set_trace()
        latent_repr = algorithm.extract_latent(dataset.test_data())
        if self.tsne_dim == 2:
            tsne = TSNE(n_components=2, random_state=0)
            tsne_obj = tsne.fit_transform(latent_repr)
            fig = plt.figure(figsize=(20, 10))
            plt.scatter(
                tsne_obj[:, 0],
                tsne_obj[:, 1],
                c=dataset.test_labels.values,
                cmap="tab20",
            )
            # plt.plot(label_mean,
            # label='label_mean')
            # plt.legend()
            # plt.title(f'''MSE: {mean_squared_error}; Dist_to_mean:
            # {dist_to_mean}''')
            self.evaluation.save_figure(fig, "tsne_latent_2d")
            plt.close("all")
        if self.tsne_dim == 3:
            tsne = TSNE(n_components=3, random_state=0)
            tsne_obj = tsne.fit_transform(latent_repr)
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(tsne_obj[:, 0], tsne_obj[:, 1], tsne_obj[:, 2])

            def rotate(angle):
                ax.view_init(azim=angle)

            angle = 3
            ani = animation.FuncAnimation(
                fig, rotate, frames=np.arange(0, 360, angle), interval=50
            )
            ani.save(
                os.path.join(self.evaluation.run_folder, "tsne_latent_3d_rotate.gif"),
                writer=animation.PillowWriter(fps=20),
            )
            # figure.savefig(os.path.join(self.run_folder, name))
            # self.evaluation.save_figure(fig, "tsne_latent_3d")
            plt.close("all")


# def rotate(angle):
# ax.view_init(azim=angle)
