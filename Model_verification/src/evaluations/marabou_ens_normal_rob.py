import multiprocessing as mp
import pandas as pd
import json
import numpy as np
import os
import re
import random
import time
from maraboupy import Marabou
from itertools import combinations

from .evaluation import evaluation


class marabou_ens_normal_rob(evaluation):
    def __init__(
        self,
        name: str = "marabou_ens_normal_rob",
        eps=0.001,
        cfg=None,
    ):
        self.name = name
        self.eps=eps
        self.cfg = cfg

    def evaluate(self, dataset, algorithm, run_inst):
        test_data = dataset.test_data()
        parallel = 0.25
        result_dict = {}
        if dataset.name == 'mnist':
            MNIST = True
        else:
            MNIST = False
        # set number of submodels, obsolete if part_models is not None
        num_folders = 100
        model_path = os.path.join(os.getcwd(), self.cfg["test_models"][2:],
                'models')

        # select particular models, makes num_folders obsolete
        # re.match required to select the right submodels for models 
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

        # read in necessary model information
        with open(os.path.join(model_path, "alltheq.json"), "r") as json_file:
            q_values = json.load(json_file)
        with open(os.path.join(model_path, "border.json"), "r") as json_file:
            border = json.load(json_file)
        test_model_folders = self.get_test_model_folders(
            model_path=model_path,
            q_values=q_values,
            num_folders=num_folders,
            part_models=part_models
        )

        for _ in range(20):
            if MNIST:
                normal_point = test_data[dataset.test_labels == 7].sample(1)
            else:
                normal_point = test_data[dataset.test_labels == 0].sample(1)

            normal_point.columns = normal_point.columns.astype(int)

            input_point_dict = {}
            res = []
            if parallel:
                pool = mp.Pool(int(parallel * mp.cpu_count()))
                for onnx_path in test_model_folders:
                    arg = (normal_point, onnx_path, q_values)
                    res.append(pool.apply_async(self.calc_largest_error, args=(arg)))
                pool.close()
                pool.join()
                results = [x.get() for x in res]
            else:
                for onnx_path in test_model_folders:
                    res.append(
                        self.calc_largest_error(
                            normal_point, onnx_path, q_values
                        )
                    )
                    results = res
            # extract largest errors and calculate verifiable robustness
            largest_error_dict = dict(
                map(lambda x: (x["id"], x["largest_error"]), results)
            )
            verified_largest_error = np.sqrt(
                (np.array(list(largest_error_dict.values())) ** 2).sum()
                / len(largest_error_dict.values())
            )

            # save results in esp_res_dict
            input_point_dict[str(self.eps)] = {}
            input_point_dict[str(self.eps)]["largest_error_dict"] = largest_error_dict
            input_point_dict[str(self.eps)]["ver_largest_error"] = verified_largest_error
            times = dict(map(lambda x: (x["id"], x["calc_time"]), results))
            duration = sum(list(times.values()))

            # calculate pseudo-adversarial
            adv_df = pd.DataFrame(columns=range(test_data.shape[1]))
            num_results = len(results)
            res_list = []
            res_list.append(adv_df)
            for result in results:
                # works if result["solution"] exists
                try:
                    res_df = pd.DataFrame(
                        [result["solution"]], columns=result["features"]
                    )
                    res_list.append(res_df)
                except:
                    pass
            adv_df = pd.concat(res_list, axis=0)
            adv_cand = pd.DataFrame(adv_df.mode().iloc[0]).transpose()
            norm_adv_pair = pd.concat([normal_point, adv_cand])

            # save results for particular input in total result_dict
            norm_ind = str(normal_point.index[0])
            self.save_csv(run_inst, norm_adv_pair, f"norm_adv_pair_{norm_ind}")
            result_dict[norm_ind] = {
                "eps": self.eps,
                "verified_largest_error": verified_largest_error,
                "duration": duration,
                "ratio": verified_largest_error / border,
                "border": border,
            }
            self.save_json(run_inst, input_point_dict, f"input_point_dict_{norm_ind}")
        self.save_json(run_inst, result_dict, f"results")

    def calc_largest_error(self, normal_point, onnx_path, q_values):
        # preparations: get network + input
        marabou_options = Marabou.createOptions(timeoutInSeconds=300)
        network = Marabou.read_onnx(onnx_path)
        numInputVars = len(network.inputVars[0][0])
        model_info = self.get_model_info(onnx_path, q_values, numInputVars,
                normal_point.shape[1])
        model_input = normal_point[model_info["features"]]
        model_info["input"] = list(model_input.values[0])
        q = model_info["q"]

        # set area in which to calculate the wce + variables for wce
        # calculation
        for ind in range(numInputVars):
            network.setLowerBound(
                network.inputVars[0][0][ind], model_input.iloc[0, ind] - self.eps
            )
            network.setUpperBound(
                network.inputVars[0][0][ind], model_input.iloc[0, ind] + self.eps
            )
        delta = 6 # value set by experience; Adjust if too small
        delta_change = 3
        outputVar = network.outputVars[0][0]
        accuracy = 0.001

        # binary search over delta
        start_time = time.time()
        while delta_change > accuracy:
            # adds outputVar <= q - delta
            network.addInequality([outputVar], [1], q - delta)
            try:
                network_solution = network.solve(options=marabou_options, verbose=False)
            except:
                pass
            # remove inequality
            network.equList = network.equList[:-1]
            if len(network_solution[0]) > 0:
                cur_best_solution = network_solution
                delta = delta + delta_change
            else:
                # try outputVar <= -q - delta
                network.addInequality([outputVar], [-1], -q - delta)
                try:
                    network_solution = network.solve(
                        options=marabou_options, verbose=False
                    )
                except:
                    pass
                # remove inequality
                network.equList = network.equList[:-1]
                if len(network_solution[0]) > 0:
                    cur_best_solution = network_solution
                    delta = delta + delta_change
                else:
                    delta = delta - delta_change
                    delta_change = delta_change / 2

        # add safety margin as we do only approximate, save results
        delta = delta + 2 * accuracy
        model_info["largest_error"] = delta
        end_time = time.time()
        model_info["calc_time"] = end_time - start_time

        # add solution_point to model_info
        try:
            solution_point = extract_solution_point(cur_best_solution, network)
            model_info["solution"] = solution_point[0]
        except:
            pass
        return model_info

    def get_test_model_folders(
        self, model_path, q_values, num_folders=None, part_models=None
    ):
        min_q = np.quantile(list(q_values.values()), 0.0)
        if part_models is not None:
            if not isinstance(part_models, list):
                part_models = [part_models]
            models = list(map(lambda x:str(x), part_models))
        elif num_folders is not None:
            rem_q_values = {k:v for (k,v) in q_values.items() if v >= min_q}
            models = random.sample(rem_q_values.keys(), num_folders)
        else:
            models = list(filter(lambda x:x.isdigit(), os.listdir(model_path)))

        model_folders = list(map(lambda x:os.path.join(model_path, x,
            "model.onnx"), models))
        return model_folders

    def get_model_info(self, onnx_path, q_values, num_modelFeatures, dataset_dim):
        model_path = onnx_path[: onnx_path.rfind("/")]
        model_number = model_path[model_path.rfind("/") + 1 :]
        model_features = self.features_of_index(
            int(model_number), dataset_dim, num_modelFeatures
        )
        q_value = q_values[model_number]
        return {"id": model_number, "features": model_features, "q": q_value}

    def features_of_index(self, index, num_feat, bag):
        np.random.seed(index)
        seed = np.random.randint(10000000)
        np.random.seed(seed)
        base = list(range(num_feat))
        np.random.shuffle(base)
        return base[:bag]

def extract_solution_point(solution, network):
    solution = solution[0]
    inputpoint1 = []
    for ind1 in network.inputVars[0][0]:
        inputpoint1.append(solution[ind1])

    outputpoint1 = []
    for ind1 in network.outputVars[0]:
        outputpoint1.append(solution[ind1])

    return inputpoint1, outputpoint1
