import maraboupy
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
import time
from maraboupy import Marabou
from maraboupy import MarabouCore
from itertools import combinations

from .evaluation import evaluation

# sat means there is an 'advers attack' -> it is not robust
# unsat means there is no advers attack -> it is robust

class marabou_superv_robust:
    def __init__(
        self,
        eval_inst: evaluation,
        name: str = "marabou_superv_robust",
        eps_range=[0.0,1],
        num_steps = 50,
        num_samples = 20,
        singular_class_sample = 0,
    ):
        self.name = name
        self.evaluation = eval_inst
        self.eps_range = eps_range
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.singular_class_sample = singular_class_sample

    def evaluate(self, dataset, algorithm):
        network = self.get_network(algorithm, dataset)
        self.algorithm = algorithm

        samples = []
        if self.num_samples == 1:
            while len(samples) < self.num_samples:
                cand = dataset.test_data().sample(1)
                cand_pred = algorithm.predict(cand)
                if ((dataset.test_labels[cand_pred.index] ==
                    cand_pred['pred_label']).values and
                    cand_pred['pred_label'].values[0] ==
                    self.singular_class_sample):
                    samples.append(cand)
        else:
            while len(samples) < self.num_samples:
                cand = dataset.test_data().sample(1)
                cand_pred = algorithm.predict(cand)
                if (dataset.test_labels[cand_pred.index] ==
                    cand_pred['pred_label']).values:
                    samples.append(cand)

        result_dict = {}
        for sample_num, sample in enumerate(samples):
            pred = algorithm.predict(sample)
            solutions = self.check_sample(sample, network, pred)
            aggr_solutions = self.aggregate_sample(solutions)
            result_dict[f"sample_{sample_num}"] = aggr_solutions
            self.plot_and_save_surrounding_fcts(result_dict, sample, network,
                    algorithm, sample_num)
        self.evaluation.save_json(result_dict, "results_marabou_superv_robust")

        if self.num_samples == 1:
            self.save_singular_sample(solutions)

    def save_singular_sample(self, solutions):
        index = list(solutions.keys())
        col1 = list(map(lambda x:len(x[0])>0, list(solutions.values())))
        col2 = list(map(lambda x:x[1].getTotalTime(), list(solutions.values())))
        res_df = pd.DataFrame(np.array([col1, col2]).transpose(), index=index,
                columns = ['sat', 'runtime'])
        self.evaluation.save_csv(res_df, "singular_sample")

    def aggregate_sample(self, solutions):
        total_runtime = sum(list(map(lambda x: x[1].getTotalTime(),
            solutions.values())))
        tot_sat_sol = sum(list(map(lambda x:len(x[0])>0,
            solutions.values())))
        tot_unsat_sol = sum(list(map(lambda x:len(x[0])==0,
            solutions.values())))
        res_dict = {}
        res_dict['tot_runtime'] = total_runtime
        res_dict['tot_sat_sol'] = tot_sat_sol
        res_dict['tot_unsat_sol'] = tot_unsat_sol
        return res_dict

    def check_sample(self, sample, network, pred_tot):
        pred = pred_tot['pred_label']
        marabou_options = Marabou.createOptions(timeoutInSeconds=300)
        res_dict = {}
        for eps in np.linspace(self.eps_range[0],self.eps_range[1],self.num_steps):
            lower_bounds = (sample.values - eps).clip(-1,1)
            upper_bounds = (sample.values + eps).clip(-1,1)
            for ind in range(sample.shape[1]):
                network.setLowerBound(
                    network.inputVars[0][0][ind], lower_bounds[0][ind]
                )
                network.setUpperBound(
                    network.inputVars[0][0][ind], upper_bounds[0][ind]
                )

            pred_outputVar = network.outputVars[0][pred.values[0]]
            disj_eqs = []
            for ind, outputVar in enumerate(network.outputVars[0]):
                if outputVar != pred_outputVar:
                    eq = MarabouCore.Equation(MarabouCore.Equation.GE)
                    eq.addAddend(-1, pred_outputVar)
                    eq.addAddend(1, outputVar)
                    eq.setScalar(0)
                    disj_eqs.append([eq])

            network.disjunctionList = []
            network.addDisjunctionConstraint(disj_eqs)

            network_solution = network.solve(options=marabou_options)
            res_dict[eps] = network_solution

        return res_dict


    def plot_and_save_surrounding_fcts(
        self, result_dict, input_sample, network, algorithm, key
    ):
        # clip samples at -1,1
        # sample within largest eps range


        samples = np.random.uniform(
            low=(input_sample - self.eps_range[1]).values,
            high=(input_sample + self.eps_range[1]).values,
            size=(1000, len(input_sample.columns)),
        )
        distr = algorithm.count_lin_subfcts(algorithm.module, pd.DataFrame(samples))
        sorted_distr = sorted(list(map(lambda x: x[1], distr)), reverse=True)
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        fct_indices = range(len(sorted_distr))
        ax.bar(fct_indices, sorted_distr)
        self.evaluation.save_figure(fig, f"surrounding_fcts_distr_{key}")
        self.evaluation.save_csv(
            pd.DataFrame(sorted_distr),
            f"surrounding_fcts_distr_{key}",
            subfolder=f"results_robust_{key}",
        )
        result_dict["sample_" + str(key)].update({"surrounding_fcts": len(sorted_distr)})

    def get_network(self, algorithm, dataset):
        randomInput = torch.randn(1, algorithm.topology[0])
        run_folder = self.evaluation.run_folder[
            self.evaluation.run_folder.rfind("202") :
        ]
        onnx_folder = os.path.join(
            "./models/onnx_models/",
            str(algorithm.name) + "_" + dataset.name,
            run_folder,
        )
        marabou_folder = os.path.join(
            "./models/marabou_models/",
            str(algorithm.name) + "_" + dataset.name,
            run_folder,
        )
        os.makedirs(onnx_folder, exist_ok=True)
        os.makedirs(marabou_folder, exist_ok=True)
        torch.onnx.export(
            algorithm.module.get_neural_net(),
            randomInput.float(),
            os.path.join(onnx_folder, "saved_algorithm.onnx"),
        )
        network = Marabou.read_onnx(
            os.path.join(onnx_folder, "saved_algorithm.onnx"),
            outputName=str(2 * len(algorithm.module.get_neural_net()) + 1),
        )
        return network


def extract_solution_point(solution, network):
    solution = solution[0]
    inputpoint1 = []
    for ind1 in network.inputVars[0][0]:
        inputpoint1.append(solution[ind1])

    outputpoint1 = []
    for ind1 in network.outputVars[0]:
        outputpoint1.append(solution[ind1])

    return inputpoint1, outputpoint1


def extract_solution_stats(solution):
    res_dict = {}
    res_dict["MaxDegradation"] = solution[1].getMaxDegradation()
    res_dict["MaxStackDepth"] = solution[1].getMaxStackDepth()
    res_dict["NumConstrainFixingSteps"] = solution[1].getNumConstraintFixingSteps()
    res_dict["NumMainLoopIterations"] = solution[1].getNumMainLoopIterations()
    res_dict["NumPops"] = solution[1].getNumPops()
    res_dict["NumPrecisionRestorations"] = solution[1].getNumPrecisionRestorations()
    res_dict["NumSimplexPivotSelectionsIgnoredForStability"] = solution[
        1
    ].getNumSimplexPivotSelectionsIgnoredForStability()
    res_dict["NumSimplexUnstablePivots"] = solution[1].getNumSimplexUnstablePivots()
    res_dict["NumSplits"] = solution[1].getNumSplits()
    res_dict["NumTableauPivots"] = solution[1].getNumTableauPivots()
    res_dict["NumVisitedTreeStates"] = solution[1].getNumVisitedTreeStates()
    res_dict["TimeSimplexStepsMicro"] = solution[1].getTimeSimplexStepsMicro()
    res_dict["TotalTime"] = solution[1].getTotalTime()
    res_dict["hasTimedOut"] = solution[1].hasTimedOut()
    return res_dict


def add_solution_stats(solution1, solution2):
    if solution1 == 0:
        return solution2
    if solution2 == 0:
        return solution1
    res_solution = {}
    for key in solution1.keys():
        res_solution[key] = solution1[key] + solution2[key]
    return res_solution


def update_to_tot_key(res_dict):
    updated_dict = {}
    for key in res_dict:
        updated_dict["tot_" + key] = res_dict[key]
    return updated_dict


def test_for_collapsing(dataset, algorithm):
    pred_dataset = algorithm.predict(dataset.test_data())
    if pred_dataset.var().sum() < 0.00001:
        return True
    else:
        return False
