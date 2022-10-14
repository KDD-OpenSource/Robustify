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

# eps is the area we are searching in, delta is the largest error value


class marabou_largest_error:
    def __init__(
        self,
        eval_inst: evaluation,
        name: str = "marabou_largest_error",
        num_eps_steps=100,
        eps=0.1,
    ):
        self.name = name
        self.evaluation = eval_inst
        self.num_eps_steps = num_eps_steps
        self.eps = eps

    def evaluate(self, dataset, algorithm):
        collapsing = test_for_collapsing(dataset, algorithm)
        network = self.get_network(algorithm, dataset)
        label_means = dataset.calc_label_means(subset="test")
        result_dict = {}
        marabou_options = Marabou.createOptions(timeoutInSeconds=300)
        delta_accuracy = 0.0001
        for key in label_means.keys():
            input_sample = pd.DataFrame(label_means[key]).transpose()
            output_sample = algorithm.predict(input_sample)
            solution, tot_time, delta, tot_solution_stats = self.binary_search_delta(
                network, input_sample, output_sample, delta_accuracy, marabou_options
            )

            self.plot_and_save(
                result_dict,
                key,
                solution,
                network,
                input_sample,
                output_sample,
                delta,
                tot_time,
                tot_solution_stats,
                collapsing,
            )
            self.plot_and_save_surrounding_fcts(
                result_dict, input_sample, solution, network, algorithm, key
            )
        self.evaluation.save_json(result_dict, "results_marabou_largest")

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

    def binary_search_delta(
        self, network, input_sample, output_sample, accuracy, marabou_options
    ):
        eps = self.eps
        numInputVars = len(network.inputVars[0][0])
        for ind in range(numInputVars):
            network.setLowerBound(
                network.inputVars[0][0][ind], input_sample.values[0][ind] - eps
            )
            network.setUpperBound(
                network.inputVars[0][0][ind], input_sample.values[0][ind] + eps
            )

        found_largest_delta = False
        delta = 1
        delta_change = 0.5
        start_time = time.time()
        numOutputVars = len(network.outputVars[0])
        solution = None
        tot_solution_stats = 0

        while not found_largest_delta:
            print(delta)
            disj_eqs = []
            for ind in range(numOutputVars):
                outputVar = network.outputVars[0][ind]
                inputVar = network.inputVars[0][0][ind]
                eq1 = MarabouCore.Equation(MarabouCore.Equation.GE)
                eq1.addAddend(-1, outputVar)
                eq1.addAddend(1, inputVar)
                eq1.setScalar(delta)

                eq2 = MarabouCore.Equation(MarabouCore.Equation.GE)
                eq2.addAddend(1, outputVar)
                eq2.addAddend(-1, inputVar)
                eq2.setScalar(delta)

                disj_eqs.append([eq1])
                disj_eqs.append([eq2])

            network.disjunctionList = []
            network.addDisjunctionConstraint(disj_eqs)
            network_solution = network.solve(options=marabou_options)
            solution_stats = extract_solution_stats(network_solution)
            tot_solution_stats = add_solution_stats(tot_solution_stats, solution_stats)
            if network_solution[1].hasTimedOut():
                solution = None
                break
            if len(network_solution[0]) > 0:
                extr_solution = extract_solution_point(network_solution, network)
                diff_input = abs(
                    np.array(extr_solution[0]) - input_sample.values[0]
                ).max()
                # diff_output = abs(np.array(extr_solution[1]) -
                # output_sample.values[0]).max()
                larg_diff = abs(
                    np.array(extr_solution[1]) - np.array(extr_solution[0])
                ).max()
                solution = network_solution

                if (diff_input < eps + accuracy) and larg_diff > delta - accuracy:
                    delta = delta + delta_change
                else:
                    delte = delta - delta_change

            else:
                delta = delta - delta_change

            if delta_change <= accuracy:
                found_largest_delta = True
            delta_change = delta_change / 2

        end_time = time.time()
        tot_time = end_time - start_time
        return solution, tot_time, delta, tot_solution_stats

    def plot_and_save(
        self,
        result_dict,
        key,
        solution,
        network,
        input_sample,
        output_sample,
        delta,
        tot_time,
        tot_solution_stats,
        collapsing,
    ):
        if solution is not None:
            solution_stat_dict = extract_solution_stats(solution)
            extr_solution = extract_solution_point(solution, network)
            diff_input = abs(np.array(extr_solution[0]) - input_sample.values[0]).max()
            larg_diff = abs(
                np.array(extr_solution[1]) - np.array(extr_solution[0])
            ).max()
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[20, 20])
            fig.suptitle(f"Calculation took {tot_time}")
            ax[0].plot(input_sample.values[0], label="input_sample")
            ax[0].plot(extr_solution[0], label="input_solution")
            ax[0].set_title(
                f"""L_infty dist is at most {self.eps}. (real:
                    {diff_input})"""
            )
            ax[0].legend()
            ax[0].set_ylim(-1.25, 1.25)
            ax[1].plot(extr_solution[0], label="input_solution")
            ax[1].plot(extr_solution[1], label="output_solution")
            ax[1].set_title(
                f"""L_infty dist is at least {delta} (up to
                accuracy) real: {larg_diff}"""
            )
            ax[1].legend()
            ax[1].set_ylim(-1.25, 1.25)
            self.evaluation.save_figure(
                fig, f"marabou_largest_error_close_to_sample_{key}"
            )
            plt.close("all")

            self.evaluation.save_csv(
                input_sample, "input_sample", subfolder=f"results_largest_error_{key}"
            )
            self.evaluation.save_csv(
                output_sample, "output_sample", subfolder=f"results_largest_error_{key}"
            )
            self.evaluation.save_csv(
                pd.DataFrame(extr_solution[0]),
                "input_solution",
                subfolder=f"results_largest_error_{key}",
            )
            self.evaluation.save_csv(
                pd.DataFrame(extr_solution[1]),
                "output_solution",
                subfolder=f"results_largest_error_{key}",
            )
            result_dict["label_" + str(key)] = {
                "calc_time": tot_time,
                "dist_to_y": self.eps,
                "error": delta,
                "real_dist_to_y": diff_input,
                "real_error": larg_diff,
            }
            result_dict["label_" + str(key)].update(solution_stat_dict)
            tot_solution_stats = update_to_tot_key(tot_solution_stats)
            result_dict["label_" + str(key)].update(tot_solution_stats)
            result_dict["collapsing"] = collapsing
        else:
            result_dict["label_" + str(key)] = {
                "calc_time": tot_time,
                "dist_to_y": self.eps,
                "error": None,
                "real_dist_to_y": None,
                "real_error": None,
            }
            tot_solution_stats = update_to_tot_key(tot_solution_stats)
            result_dict["label_" + str(key)].update(tot_solution_stats)
            result_dict["collapsing"] = collapsing

    def plot_and_save_surrounding_fcts(
        self, result_dict, input_sample, solution, network, algorithm, key
    ):
        # extr_solution = extract_solution_point(solution, network)
        # diff_input = abs(np.array(extr_solution[0]) -
        # input_sample.values[0]).max()
        samples = np.random.uniform(
            low=(input_sample - self.eps).values,
            high=(input_sample + self.eps).values,
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
            subfolder=f"results_largest{key}",
        )
        result_dict["label_" + str(key)].update({"surrounding_fcts": len(sorted_distr)})


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
