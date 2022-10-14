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


class marabou_robust:
    def __init__(
        self,
        eval_inst: evaluation,
        name: str = "marabou_robust",
        num_eps_steps=100,
        delta=0.1,
    ):
        self.name = name
        self.evaluation = eval_inst
        self.num_eps_steps = num_eps_steps
        self.desired_delta = delta

    def evaluate(self, dataset, algorithm):
        collapsing = test_for_collapsing(dataset, algorithm)
        network = self.get_network(algorithm, dataset)
        label_means = dataset.calc_label_means(subset="test")
        result_dict = {}
        delta_accuracy = 0.0001
        for key in label_means.keys():
            input_sample = pd.DataFrame(label_means[key]).transpose()
            output_sample = algorithm.predict(input_sample)
            solution, tot_time, eps, tot_solution_stats = self.binary_search_delta(
                network, input_sample, output_sample, delta_accuracy
            )

            self.plot_and_save(
                result_dict,
                key,
                solution,
                network,
                input_sample,
                output_sample,
                eps,
                tot_time,
                tot_solution_stats,
                collapsing,
            )
            self.plot_and_save_surrounding_fcts(
                result_dict, input_sample, solution, network, algorithm, key
            )
        self.evaluation.save_json(result_dict, "results_marabou_robust")

    def plot_and_save(
        self,
        result_dict,
        key,
        solution,
        network,
        input_sample,
        output_sample,
        eps,
        tot_time,
        tot_solution_stats,
        collapsing,
    ):
        # self.evaluation.save_json(solution_stat_dict, f'solution_stats_{key}')
        if solution is not None:
            solution_stat_dict = extract_solution_stats(solution)
            extr_solution = extract_solution_point(solution, network)
            diff_input = abs(np.array(extr_solution[0]) - input_sample.values[0]).max()
            diff_output_max = abs(
                np.array(extr_solution[1]) - output_sample.values[0]
            ).max()
            diff_input_output_sample = (
                abs(input_sample.values[0] - output_sample.values[0])
                .max()
                .astype(np.float64)
            )
            #     fig, f"marabou_robust_different_pairs_label_{key}")
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[20, 20])
            fig.suptitle(f"Calculation took {tot_time}")
            ax[0].plot(input_sample.values[0], label="input_sample")
            ax[0].plot(extr_solution[0], label="input_solution")
            ax[0].set_title(
                f"""L_infty dist between x and y is at most {eps}
                    (up to accuracy), real: {diff_input}"""
            )
            ax[0].legend()
            ax[0].set_ylim(-1.25, 1.25)
            ax[1].plot(output_sample.values[0], label="output_sample")
            ax[1].plot(extr_solution[1], label="output_solution")
            ax[1].set_title(
                f"""L_infty dist between f(x) and f(y) is at
                    least {self.delta} (real: {diff_output_max})"""
            )
            ax[1].legend()
            ax[1].set_ylim(-1.25, 1.25)
            self.evaluation.save_figure(fig, f"marabou_robust_same_pairs_label_{key}")
            result_dict["label_" + str(key)] = {
                "calc_time": tot_time,
                "robustness": eps,
                "delta": self.delta,
                "real_robustness": diff_input,
                "real_delta": diff_output_max,
                "input_output_sample_diff": diff_input_output_sample,
            }
            result_dict["label_" + str(key)].update(solution_stat_dict)
            tot_solution_stats = update_to_tot_key(tot_solution_stats)
            result_dict["label_" + str(key)].update(tot_solution_stats)
            result_dict["collapsing"] = collapsing
            self.evaluation.save_csv(
                input_sample, "input_sample", subfolder=f"results_robust_{key}"
            )
            self.evaluation.save_csv(
                output_sample, "output_sample", subfolder=f"results_robust_{key}"
            )
            self.evaluation.save_csv(
                pd.DataFrame(extr_solution[0]),
                "input_solution",
                subfolder=f"results_robust_{key}",
            )
            self.evaluation.save_csv(
                pd.DataFrame(extr_solution[1]),
                "output_solution",
                subfolder=f"results_robust_{key}",
            )
        else:
            result_dict["label_" + str(key)] = {
                "calc_time": tot_time,
                "robustness": None,
                "delta": self.delta,
                "real_robustness": None,
                "real_delta": None,
                "input_output_sample_diff": None,
            }
            # result_dict['label_'+str(key)].update(solution_stat_dict)
            tot_solution_stats = update_to_tot_key(tot_solution_stats)
            result_dict["label_" + str(key)].update(tot_solution_stats)
            result_dict["collapsing"] = collapsing

    def binary_search_eps(
        self, eps, delta, accuracy, network, input_sample, output_sample
    ):
        solution = None
        marabou_options = Marabou.createOptions(timeoutInSeconds=300)
        found_closest_eps = False
        numInputVars = len(network.inputVars[0][0])
        eps_change = eps / 2
        summed_solution_stats = 0
        while not found_closest_eps:
            # add eps constraints
            print(eps)
            for ind in range(numInputVars):
                network.setLowerBound(
                    network.inputVars[0][0][ind], input_sample.values[0][ind] - eps
                )
                network.setUpperBound(
                    network.inputVars[0][0][ind], input_sample.values[0][ind] + eps
                )

            network_solution = network.solve(options=marabou_options)
            solution_stats = extract_solution_stats(network_solution)
            summed_solution_stats = add_solution_stats(
                summed_solution_stats, solution_stats
            )
            if network_solution[1].hasTimedOut():
                break
            if len(network_solution[0]) > 0:
                extr_solution = extract_solution_point(network_solution, network)
                solution = network_solution
                diff_input = abs(
                    np.array(extr_solution[0]) - input_sample.values[0]
                ).max()
                diff_output_max = abs(
                    np.array(extr_solution[1]) - output_sample.values[0]
                ).max()
                # found solution
                if (diff_input < eps + accuracy) and (
                    diff_output_max > delta - accuracy
                ):
                    eps = eps - eps_change
                else:
                    eps = eps + eps_change
            else:
                eps = eps + eps_change
            if eps_change <= accuracy:
                found_closest_eps = True
            eps_change = eps_change / 2
        return solution, eps, summed_solution_stats

    def binary_search_delta(self, network, input_sample, output_sample, delta_accuracy):
        self.delta = self.desired_delta
        delta_change = self.desired_delta / 2
        found_real_delta = False
        # tot_solution_stats = 0
        while not found_real_delta:
            delta = self.delta
            print(delta)
            numOutputVars = len(network.outputVars[0])
            disj_eqs = []
            for ind in range(numOutputVars):
                outputVar = network.outputVars[0][ind]
                eq1 = MarabouCore.Equation(MarabouCore.Equation.GE)
                eq1.addAddend(-1, outputVar)
                eq1.setScalar(delta - output_sample.values[0][ind])

                eq2 = MarabouCore.Equation(MarabouCore.Equation.GE)
                eq2.addAddend(1, outputVar)
                eq2.setScalar(delta + output_sample.values[0][ind])

                disj_eqs.append([eq1, eq2])

            network.disjunctionList = []
            network.addDisjunctionConstraint(disj_eqs)

            eps = 2.0
            accuracy = 0.0000005
            start_time = time.time()
            # binary search over values of eps
            solution, eps, summed_solution_stats = self.binary_search_eps(
                eps, delta, accuracy, network, input_sample, output_sample
            )
            # tot_solution_stats = add_solution_stats(
            # tot_solution_stats,summed_solution_stats)
            end_time = time.time()
            tot_time = end_time - start_time
            if solution is not None:
                extr_solution = extract_solution_point(solution, network)
                diff_output_max = abs(
                    np.array(extr_solution[1]) - output_sample.values[0]
                ).max()
                if abs(diff_output_max - self.desired_delta) < delta_accuracy:
                    found_real_delta = True
                elif diff_output_max < self.desired_delta:
                    self.delta = self.delta + delta_change
                elif diff_output_max > self.desired_delta:
                    self.delta = self.delta - delta_change
                delta_change = delta_change / 2
                if delta_change < 0.00001:
                    found_real_delta = True
            else:
                found_real_delta = True
        # return solution, tot_time, eps, tot_solution_stats
        return solution, tot_time, eps, summed_solution_stats

    def plot_and_save_surrounding_fcts(
        self, result_dict, input_sample, solution, network, algorithm, key
    ):
        # extr_solution = extract_solution_point(solution, network)
        # diff_input = abs(np.array(extr_solution[0]) -
        # input_sample.values[0]).max()
        samples = np.random.uniform(
            low=(input_sample - self.desired_delta).values,
            high=(input_sample + self.desired_delta).values,
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
        result_dict["label_" + str(key)].update({"surrounding_fcts": len(sorted_distr)})
        # sample 1000 points
        # check fct for each of them
        # count number of different functions
        # save histogram
        # return value and save in results

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
        onnx_path = os.path.join(onnx_folder, "saved_algorithm.onnx")
        onnx_outputName = str(2 * len(algorithm.module.get_neural_net()) + 1)
        network = Marabou.read_onnx(onnx_path, outputName = onnx_outputName
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
