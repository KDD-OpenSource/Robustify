import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import evaluation


class linsubfct_parallelPlots:
    def __init__(
        self,
        eval_inst: evaluation,
        name: str = "linsubfct_parallelPlots",
        num_plots: int = 20,
    ):
        self.name = name
        self.evaluation = eval_inst
        self.num_plots = num_plots

    def evaluate(self, dataset, algorithm):
        input_points = dataset.test_data()
        output_points = algorithm.predict(input_points)
        ymax = pd.DataFrame([input_points.max(), output_points.max()]).max().max()
        ymin = pd.DataFrame([input_points.min(), output_points.min()]).min().min()
        functions = algorithm.get_points_of_linsubfcts(algorithm.module, input_points)
        for function in list(functions.keys()):
            if len(functions[function]) < self.num_plots:
                ctr = 0
                for input_point in functions[function]:
                    output_point = algorithm.module(input_point)[0].detach()
                    fig = plt.figure()
                    plt.ylim(ymin, ymax)
                    plt.plot(input_point, color="blue")
                    plt.plot(output_point, color="orange")
                    self.evaluation.save_figure(
                        fig,
                        f"plot_{function}"[:20] + "_" + str(ctr),
                        subfolder="linsubfuncs",
                    )
                    plt.close("all")
                    ctr += 1
            else:
                function_points = functions[function]
                rand_ind = np.random.choice(
                    len(function_points), replace=False, size=self.num_plots
                )
                for ind in rand_ind:
                    output_point = algorithm.module(function_points[ind])[0].detach()
                    fig = plt.figure()
                    plt.ylim(ymin, ymax)
                    plt.plot(function_points[ind], color="blue")
                    plt.plot(output_point, color="orange")
                    self.evaluation.save_figure(
                        fig,
                        f"plot_{function}"[:20] + "_" + str(ind),
                        subfolder="linsubfuncs",
                    )
                    plt.close("all")
