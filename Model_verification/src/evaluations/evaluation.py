import os
import matplotlib.pyplot as plt
import time
import json
import torch
import numpy as np
import pandas as pd
from maraboupy import Marabou


class evaluation:
    def __init__(self, base_folder=None):
        pass

    #        if base_folder:
    #            self.result_folder = os.path.join(os.getcwd(), "reports", base_folder)
    #        else:
    #            self.result_folder = os.path.join(os.getcwd(), "reports")
    #
    #    def make_run_folder(self, ctx, run_number=None):
    #        try:
    #            datetime = time.strftime("%Y-%m-%dT%H:%M:%S")
    #            if run_number:
    #                folder_name = datetime + "_" + ctx + "_run_" + str(run_number)
    #            else:
    #                folder_name = datetime + "_" + ctx
    #            self.run_folder = os.path.join(self.result_folder, folder_name)
    #            os.makedirs(self.run_folder)
    #        except:
    #            time.sleep(1)
    #            datetime = time.strftime("%Y-%m-%dT%H:%M:%S")
    #            folder_name = datetime + "_" + ctx
    #            self.run_folder = os.path.join(self.result_folder, folder_name)
    #            os.makedirs(self.run_folder)

    def get_run_folder(self):
        return self.run_folder

    def save_figure(self, run_inst, figure, name: str, subfolder=None):
        name = name + ".png"
        if subfolder:
            os.makedirs(os.path.join(run_inst.run_folder, subfolder), exist_ok=True)
            figure.savefig(os.path.join(run_inst.run_folder, subfolder, name))
        else:
            figure.savefig(os.path.join(run_inst.run_folder, name))

    def save_json(self, run_inst, res_dict: dict, name: str, subfolder=None):
        name = name + ".json"
        if subfolder:
            os.makedirs(os.path.join(run_inst.run_folder, subfolder), exist_ok=True)
            with open(
                os.path.join(run_inst.run_folder, subfolder, name), "w"
            ) as out_file:
                json.dump(res_dict, out_file, indent=4)
        else:
            with open(os.path.join(run_inst.run_folder, name), "w") as out_file:
                json.dump(res_dict, out_file, indent=4)

    def save_csv(self, run_inst, data, name: str, subfolder=None):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        name = name + ".csv"
        if subfolder:
            os.makedirs(os.path.join(run_inst.run_folder, subfolder), exist_ok=True)
            data.to_csv(os.path.join(run_inst.run_folder, subfolder, name))
        else:
            data.to_csv(os.path.join(run_inst.run_folder, name))

    def plot_and_save_surrounding_fcts(self, run_inst, result_dict, input_sample, algorithm,
            key):
        samples = np.random.uniform(
            low=(input_sample - self.surrounding_margin).values,
            high=(input_sample + self.surrounding_margin).values,
            size=(1000, len(input_sample.columns)),
        )
        distr = algorithm.count_lin_subfcts(algorithm.module, pd.DataFrame(samples))

        # distr plot
        sorted_distr = sorted(list(map(lambda x: x[1], distr)), reverse=True)
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        fct_indices = range(len(sorted_distr))
        ax.bar(fct_indices, sorted_distr)
        self.save_figure(run_inst, fig, f"surrounding_fcts_distr_{key}")

        # save exact distr of surrounding fcts as csv
        self.save_csv(
            run_inst,
            pd.DataFrame(sorted_distr),
            f"surrounding_fcts_distr_{key}",
            subfolder=f"results_largest{key}",
        )
        # update statistics in result_dict
        result_dict["label_" + str(key)].update({"surrounding_fcts": len(sorted_distr)})

    def get_marabou_network(self, algorithm, dataset, run_inst):
        randomInput = torch.randn(1, algorithm.topology[0])
        run_folder = run_inst.run_folder[
            run_inst.run_folder.rfind("202") :
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

def extract_marabou_solution_point(solution, network):
    solution = solution[0]
    inputpoint1 = []
    for ind1 in network.inputVars[0][0]:
        inputpoint1.append(solution[ind1])

    outputpoint1 = []
    for ind1 in network.outputVars[0]:
        outputpoint1.append(solution[ind1])

    return inputpoint1, outputpoint1


def extract_marabou_solution_stats(solution):
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

def add_marabou_solution_stats(solution1, solution2):
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
