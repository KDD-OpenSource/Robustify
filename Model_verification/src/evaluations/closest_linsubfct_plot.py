import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from .evaluation import evaluation


class closest_linsubfct_plot:
    def __init__(self, eval_inst: evaluation, name: str = "closest_linsubfct_plot"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        # sample indices
        # linSubfctDist = algorithm.lin_sub_fct_Counters
        dists = []
        import pdb

        pdb.set_trace()
        for index, point in dataset.test_data().iterrows():
            dists.append(
                algorithm.get_closest_funcBoundary(
                    algorithm.module, torch.tensor(point)
                )[0]
            )
        import pdb

        pdb.set_trace()
        dists.sort()
        fig = plt.figure()
        plt.plot(dists)
        self.evaluation.save_figure(fig, "closest_bound_dist_plot")
        plt.close("all")


#        for ind in range(len(linSubfctDist)):
#            # import pdb; pdb.set_trace()
#            fig = plt.figure()
#            fctIndices = range(len(linSubfctDist[ind]))
#            values = list(map(lambda x: x[1], linSubfctDist[ind]))
#            # import pdb; pdb.set_trace()
#            plt.bar(fctIndices, values)
#            # plt.plot(output_points.iloc[ind,:])
#            self.evaluation.save_figure(fig, "barplot_" + str(ind))
#            plt.close("all")
