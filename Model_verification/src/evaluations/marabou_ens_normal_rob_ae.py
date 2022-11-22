import multiprocessing as mp
import json
import re
import numpy as np
import os
import random
import time
from maraboupy import Marabou
from maraboupy import MarabouCore

from .evaluation import evaluation


class marabou_ens_normal_rob_ae(evaluation):
    def __init__(
        self,
        name: str = "marabou_ens_normal_rob_ae",
        eps=0.001,
        cfg=None,
    ):
        self.name = name
        self.eps = eps
        self.cfg = cfg

    def evaluate(self, dataset, algorithm, run_inst):
        test_data = dataset.test_data()
        parallel = 1
        result_dict = {}

        part_models = None
        num_folders = 100
        model_path = os.path.join(os.getcwd(), self.cfg["test_models"][2:],
                "models")

        if "submodels.json" in os.listdir(model_path) and re.match(".*_\d_.*",
                self.cfg.ctx):
            submodels_path = os.path.join(model_path, 'submodels.json')
            with open(submodels_path, 'r') as jsonfile:
                submodel_dict = json.load(jsonfile)

            model_number = list(filter(str.isdigit, self.cfg.ctx))[0]
            submodel_dict[str(model_number)] = list(
                map(lambda x: int(x), submodel_dict[str(model_number)])
            )
            part_models = submodel_dict[str(model_number)]
        else: 
            part_models = None

        with open(os.path.join(model_path, "border.json"), "r") as json_file:
            border = json.load(json_file)
        test_model_folders = self.get_test_model_folders( model_path,
                num_folders=num_folders, part_models=part_models)

        for _ in range(20):
            normal_point = test_data[dataset.test_labels == 0].sample(1)
            input_point_dict = {}
            res = []
            if parallel:
                pool = mp.Pool(int(parallel * mp.cpu_count()))
                for onnx_path in test_model_folders:
                    arg = (normal_point, onnx_path)
                    res.append(pool.apply_async(self.calc_largest_error, args=(arg)))
                pool.close()
                pool.join()
                results = [x.get() for x in res]
            else:
                for onnx_path in test_model_folders:
                    res.append(
                        self.calc_largest_error(
                            normal_point, onnx_path
                        )
                    )
                    results = res

            largest_error_dict = dict(
                map(lambda x: (x["id"], x["largest_error"] / x["ae_div"]), results)
            )
            verified_largest_error_mean_sqrt = np.sqrt(
                (np.array(list(largest_error_dict.values())) ** 2).sum()
                / len(largest_error_dict.values())
            )
            verified_largest_error_median = np.median(
                np.array(list(largest_error_dict.values()))
            )
            times = dict(map(lambda x: (x["id"], x["calc_time"]), results))

            input_point_dict[str(self.eps)] = {}
            input_point_dict[str(self.eps)]["largest_error_dict"] = largest_error_dict
            input_point_dict[str(self.eps)][
                "ver_largest_error_mean_sqrt"
            ] = verified_largest_error_mean_sqrt
            input_point_dict[str(self.eps)][
                "ver_largest_error_median"
            ] = verified_largest_error_median
            duration = sum(list(times.values()))

            norm_ind = str(normal_point.index[0])
            result_dict[norm_ind] = {
                "eps": self.eps,
                "verified_largest_error_mean_sqrt": verified_largest_error_mean_sqrt,
                "verified_largest_error_median": verified_largest_error_median,
                "ratio": verified_largest_error_median / border,
                "border": border,
                "duration": duration,
            }
            self.save_json(run_inst, input_point_dict, f"input_point_dict_{norm_ind}")
        self.save_json(run_inst, result_dict, f"results")

    def calc_largest_error(self, normal_point, onnx_path):
        # preparation
        marabou_options = Marabou.createOptions(timeoutInSeconds=300)
        model_info = self.get_model_info(onnx_path)
        network = Marabou.read_onnx(onnx_path)
        numInputVars = len(network.inputVars[0][0])

        # set input boundaries
        for ind in range(numInputVars):
            network.setLowerBound(
                network.inputVars[0][0][ind], normal_point.iloc[0, ind] - self.eps
            )
            network.setUpperBound(
                network.inputVars[0][0][ind], normal_point.iloc[0, ind] + self.eps
            )

        # set variables for binary search
        delta = 6
        delta_change = 3
        accuracy = 0.001
        numOutputVars = len(network.outputVars[0])
        found_largest_delta = False
        start_time = time.time()
        # binary search over values of delta
        while not found_largest_delta:
            disj_eqs = []
            # add equations to check whether AE error is at least delta
            for i in range(numOutputVars):
                outputVar = network.outputVars[0][i]
                inputVar = network.inputVars[0][0][i]
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
            if network_solution[1].hasTimedOut():
                solution = None
                break
            if len(network_solution[0]) > 0:
                extr_solution = extract_solution_point(network_solution, network)
                # recalculate values as marabou sometimes gives non-reliable
                # results
                diff_input = abs(
                    np.array(extr_solution[0]) - normal_point.values[0]
                ).max()
                larg_diff = abs(
                    np.array(extr_solution[1]) - np.array(extr_solution[0])
                ).max()
                solution = network_solution

                if (diff_input < self.eps + accuracy) and larg_diff > delta - accuracy:
                    delta = delta + delta_change
                else:
                    delta = delta - delta_change
            else:
                delta = delta - delta_change
            if delta_change <= accuracy:
                found_largest_delta = True
            delta_change = delta_change / 2

        # ensure that delta is an upper bound on the error
        delta = delta + 2 * accuracy
        end_time = time.time()
        model_info["largest_error"] = delta
        model_info["calc_time"] = end_time - start_time
        try:
            solution_point = extract_solution_point(solution, network)
            model_info["solution"] = solution_point[0]
        except:
            pass
        return model_info

    def get_test_model_folders(self, model_path, num_folders=None, part_models=None):
        all_models = list(filter(lambda x:x.isdigit(),os.listdir(model_path)))
        if part_models is not None:
            if not isinstance(part_models, list):
                part_models = [part_models]
            models = list(map(lambda x:str(x), part_models))
        elif num_folders is not None:
            models = random.sample(all_models, num_folders)
        else:
            models = all_models

        model_folders = list(map(lambda x:os.path.join(model_path, x,
            "conv.onnx"), models))
        return model_folders

    def get_model_info(self, onnx_path):
        model_path = onnx_path[: onnx_path.rfind("/")]
        model_number = model_path[model_path.rfind("/") + 1 :]
        ae_div_file = np.load(os.path.join(model_path, "result.npz"))
        ae_div = float(ae_div_file["div"])
        return {"id": model_number, "ae_div": ae_div}

def extract_solution_point(solution, network):
    solution = solution[0]
    inputpoint1 = []
    for ind1 in network.inputVars[0][0]:
        inputpoint1.append(solution[ind1])

    outputpoint1 = []
    for ind1 in network.outputVars[0]:
        outputpoint1.append(solution[ind1])

    return inputpoint1, outputpoint1
