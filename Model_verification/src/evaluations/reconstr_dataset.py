import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import evaluation


class reconstr_dataset:
    def __init__(self, eval_inst: evaluation, name: str = "reconstr_dataset"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        pass
        # sample indices


#        linsubfct_distr = algorithm.count_lin_subfcts(
#            algorithm.module, dataset.test_data()
#        )
#        # import pdb; pdb.set_trace()
#        perc = 0.9
#        fig = plt.figure(figsize=(20, 10))
#        fctIndices = range(len(linsubfct_distr))
#        values = list(map(lambda x: x[1], linsubfct_distr))
#        plt.bar(fctIndices, values)
#        self.evaluation.save_figure(fig, "model_linsubfct_distr")
#        self.evaluation.save_csv(np.array(values), name='linsubfct_distr')
#        plt.close("all")
#        res_values = self.get_top_perc(values,perc)
#        fctIndices = range(len(res_values))
#        fig = plt.figure(figsize=(20,10))
#        plt.bar(fctIndices, res_values)
#        self.evaluation.save_figure(fig, f"model_linsubfct_distr_{perc}")
#        self.evaluation.save_csv(np.array(values), name=f'linsubfct_distr_{perc}')
#        plt.close('all')
#        result_dict = {}
#        result_dict["lin_subfct"] = len(values)
#        result_dict[f"lin_subfct_{perc}"] = len(res_values)
#        self.evaluation.save_json(result_dict, "num_subfcts")
