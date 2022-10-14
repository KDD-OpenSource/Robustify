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


class deepoc_adv_marabou_borderpoint:
    def __init__(
        self,
        eval_inst: evaluation,
        name: str = "deepoc_adv_marabou_borderpoint",
        # accuracy wrt distance in input space
        accuracy=0.0001,
    ):
        self.name = name
        self.evaluation = eval_inst
        self.accuracy = accuracy

    # general plan: 
    # take random input point (-> mean? a set of? etc.)
    # calculate image
    # calculate border point (we need R for this, where to get this?)
    # find preimage

    # def calc_border_point(self, point, algorithm):
        # center = algorithm.center.numpy()
        # point = point.values[0]
        # anom_radius = algorithm.anom_radius
        # diff = point - center
        # diff_length = np.sqrt(np.square(diff).sum())
        # border_point = center + anom_radius*(diff/diff_length)
        # return pd.DataFrame(border_point).transpose()

    def get_samples(self, dataset, sampling_method: str, num_points=None):
        if sampling_method == 'random_points':
            sample_list = []
            for point in range(num_points):
                sample = dataset.test_data().sample(1, random_state = point)
                sample_list.append(sample)
            return sample_list
        else:
            raise Exception('could not return points as no method was specified')

    def find_closest_preimage(self, input_sample, border_point, network):
        # add output constraints
        for ind, output_value in enumerate(border_point.values[0]):
            outputVar = network.outputVars[0][ind]
            network.addEquality([outputVar], [1], output_value)


        # bin_search over input_sample
        marabou_options = Marabou.createOptions(timeoutInSeconds=300)
        found_closest_eps = False
        eps = 1
        eps_change = 0.5
        numInputVars = len(network.inputVars[0][0])
        solution = None
        while not found_closest_eps:
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
            if network_solution[1].hasTimedOut():
                break
            if len(network_solution[0]) > 0:
                extr_solution = extract_solution_point(network_solution, network)
                solution = network_solution
                diff_input = abs(
                    np.array(extr_solution[0]) - input_sample.values[0]
                ).max()
                diff_output_max = abs(
                    np.array(extr_solution[1]) - border_point.values[0]
                ).max()
                # found solution
                if (diff_input < eps + self.accuracy):
                    eps = eps - eps_change
                else:
                    eps = eps + eps_change
            else:
                eps = eps + eps_change
            if eps_change <= self.accuracy:
                found_closest_eps = True
            eps_change = eps_change / 2
        if solution is not None:
            closest_preimage = pd.DataFrame(extract_solution_point(solution,
                network)[0]).transpose()
            return eps, closest_preimage
        else:
            return None, None


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

    def evaluate(self, dataset, algorithm):
        collapsing = test_for_collapsing(dataset, algorithm)
        network = self.get_network(algorithm, dataset)
        samples = self.get_samples(dataset, 'random_points', num_points=100)
        result_dict = {}
        for i,input_sample in enumerate(samples):
            output_sample = algorithm.predict(input_sample)
            sample_border_point = algorithm.calc_border_point(output_sample)
            dist, closest_preimage = self.find_closest_preimage(input_sample,
                    sample_border_point, network)
            if closest_preimage is not None:
                result_dict[i] = (list(input_sample.iloc[0]),
                        list(closest_preimage.iloc[0]), dist)
            else: 
                result_dict[i] = (list(input_sample.iloc[0]), closest_preimage, dist)
        self.evaluation.save_json(result_dict, "deepoc_adv_marabou_borderpoint")



def test_for_collapsing(dataset, algorithm):
    pred_dataset = algorithm.predict(dataset.test_data())
    if pred_dataset.var().sum() < 0.00001:
        return True
    else:
        return False

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
