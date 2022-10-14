import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import evaluation


class linSubfctBarplots:
    def __init__(self, eval_inst: evaluation, name: str = "linSubfctBarplots"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        # sample indices
        import pdb

        pdb.set_trace()
        linSubfctDist = algorithm.lin_sub_fct_Counters
        for ind in range(len(linSubfctDist)):
            fig = plt.figure()
            fctIndices = range(len(linSubfctDist[ind]))
            values = list(map(lambda x: x[1], linSubfctDist[ind]))
            plt.bar(fctIndices, values)
            self.evaluation.save_figure(fig, "barplot_" + str(ind))
            plt.close("all")
