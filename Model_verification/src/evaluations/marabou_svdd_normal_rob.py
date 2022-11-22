import multiprocessing as mp
import json
import numpy as np
import os
import random
import time
from maraboupy import Marabou

from .evaluation import evaluation


class marabou_svdd_normal_rob(evaluation):
    def __init__(
        self,
        name: str = "marabou_svdd_normal_rob",
        eps=0.001,
        cfg=None,
    ):
        self.name = name
        self.eps = eps
        self.cfg = cfg

    def evaluate(self, dataset, algorithm, run_inst):
        test_data = dataset.test_data()
        parallel = 0.25
        result_dict = {}

        for _ in range(20):
            normal_point = test_data[dataset.test_labels == 0].sample(1)
            model_path = os.path.join(os.getcwd(),
                    self.cfg["test_models"][2:], 'models')
            # we assume multiple deepsvdd models each of which is in a
            # subfolder. If you have just one folder just put in into a folder
            # called '0' and add '_0_' somewhere in your cfg.ctx. This is to be
            # consistent with the aeens and dean model
            model_number = list(filter(str.isdigit, self.cfg.ctx))[0]
            test_model_folder = os.path.join(model_path, model_number,
                    'pmodel.onnx')
            result = self.calc_largest_error(normal_point, test_model_folder)
            sample_res_dict = {
                    "largest_error": result["largest_error"],
                    "duration": result["calc_time"],
                    "max_dist": result["max_dist"],
                    "ratio": result["error_ratio"],
                    "robustness": str(result["robustness"])
                    }

            norm_ind = str(normal_point.index[0])
            result_dict[norm_ind] = sample_res_dict
        self.save_json(run_inst, result_dict, f"results")

    def calc_largest_error(self, normal_point, onnx_path):
        # preparations
        marabou_options = Marabou.createOptions(timeoutInSeconds=300)
        model_info = self.get_model_info(onnx_path)
        network = Marabou.read_onnx(onnx_path)
        numInputVars = len(network.inputVars[0][0])
        center = model_info["c"]
        tau = model_info["tau"]
        numOutputVars = len(network.outputVars[0])
        max_dist = np.sqrt((tau**2) / numOutputVars)

        # set area in which to search for the largest error
        for ind in range(numInputVars):
            network.setLowerBound(
                network.inputVars[0][0][ind], normal_point.values[0][ind] -
                self.eps
            )
            network.setUpperBound(
                network.inputVars[0][0][ind], normal_point.values[0][ind] +
                self.eps
            )

        # set variables for binary search
        start_time = time.time()
        current_solution = None
        delta = 6
        delta_change = 3
        solution = None
        accuracy = 0.001

        # binary search
        while delta_change > accuracy:
            # ce = counterexample from SMT solver
            found_ce = False
            # we iterate over all output vars to find a counterexample
            for outputInd in range(numOutputVars):
                network.disjunctionList = []
                outputVar = network.outputVars[0][outputInd]
                network.addInequality([outputVar], [-1], -center[outputInd] - delta)
                network_solution = network.solve(options=marabou_options, verbose=False)
                network.equList = network.equList[:-1]
                if len(network_solution[0]) == 0:
                    network.addInequality([outputVar], [1], center[outputInd] - delta)
                    network_solution = network.solve(
                        options=marabou_options, verbose=False
                    )
                    network.equList = network.equList[:-1]
                if len(network_solution[0]) > 0:
                    current_solution = network_solution
                    delta = delta + delta_change
                    delta_change = delta_change / 2
                    solution = np.array(
                        extract_solution_point(current_solution, network)[1]
                    )
                    found_ce = True
                    break

            if found_ce == False:
                delta = delta - delta_change
                delta_change = delta_change / 2

        # save solution
        end_time = time.time()
        delta = delta + 2 * accuracy
        model_info["largest_error"] = delta
        model_info["calc_time"] = end_time - start_time
        model_info["max_dist"] = max_dist
        model_info["error_ratio"] = delta / max_dist
        model_info["robustness"] = delta / max_dist < 1
        try:
            # save solution if it exists
            solution_point = extract_solution_point(current_solution, network)
            model_info["solution"] = solution_point[0]
        except:
            pass

        return model_info

    def get_model_info(self, onnx_path):
        model_path = onnx_path[: onnx_path.rfind("/")]
        model_number = model_path[model_path.rfind("/") + 1 :]
        with open(os.path.join(model_path, "c.json")) as json_file:
            center = json.load(json_file)
        with open(os.path.join(model_path, "border.json")) as json_file:
            border = json.load(json_file)
        return {"id": model_number, "c": center, "tau": border}

def extract_solution_point(solution, network):
    solution = solution[0]
    inputpoint1 = []
    for ind1 in network.inputVars[0][0]:
        inputpoint1.append(solution[ind1])

    outputpoint1 = []
    for ind1 in network.outputVars[0]:
        outputpoint1.append(solution[ind1])

    return inputpoint1, outputpoint1
