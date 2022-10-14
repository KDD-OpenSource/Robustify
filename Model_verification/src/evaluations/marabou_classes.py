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


class marabou_classes:
    def __init__(
        self, eval_inst: evaluation, name: str = "marabou_classes", num_eps_steps=30
    ):
        self.name = name
        self.evaluation = eval_inst
        self.num_eps_steps = num_eps_steps

    def evaluate(self, dataset, algorithm):
        # LOAD GENERAL MODEL
        # model one network but with 2 inputs (like a small batch).
        # -> simulates two networks
        randomInput = torch.randn(2, algorithm.topology[0])
        zeros = torch.zeros(2, algorithm.topology[0])
        # '202' as short for '2021'
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
        latent_layer_key = int(len(algorithm.module.get_neural_net()) / 2)
        torch.onnx.export(
            algorithm.module.get_neural_net()[:latent_layer_key],
            randomInput.float(),
            os.path.join(onnx_folder, "saved_algorithm.onnx"),
        )
        # the below line is for debugging if the (general) outputName is wrong
        # network = Marabou.read_onnx(
        # os.path.join(onnx_folder, 'saved_algorithm.onnx'))
        network = Marabou.read_onnx(
            os.path.join(onnx_folder, "saved_algorithm.onnx"),
            outputName=str(len(algorithm.module.get_neural_net())),
        )
        # network.saveQuery('./models/marabou_models/saved_algorithm_marabou')
        network.saveQuery(os.path.join(marabou_folder, "saved_algorithm_marabou"))
        loaded_network = maraboupy.Marabou.load_query(
            os.path.join(marabou_folder, "saved_algorithm_marabou")
        )
        #'./models/marabou_models/saved_algorithm_marabou')
        dataset.test_data()
        label_means = dataset.calc_label_means(subset="test")
        if len(label_means) == 1:
            label_means[1] = pd.Series(0, range(label_means[1].shape[0]))
            if len(label_means) != 2:
                raise Exception("Something is wrong with the test_labels")

        # add equations for input and output
        # options = Marabou.createOptions(verbosity=2)
        # solutions, stats = maraboupy.MarabouCore.solve(loaded_network, options)
        numOutputVars = loaded_network.getNumOutputVariables()
        for ind1, ind2 in zip(
            range(int(numOutputVars / 2)), range(int(numOutputVars / 2), numOutputVars)
        ):
            outputInd1 = loaded_network.outputVariableByIndex(ind1)
            outputInd2 = loaded_network.outputVariableByIndex(ind2)
            equation = MarabouCore.Equation(MarabouCore.Equation.EQ)
            # test for equality
            equation.addAddend(1, outputInd1)
            equation.addAddend(-1, outputInd2)
            equation.setScalar(0)
            loaded_network.addEquation(equation)
            # network.addEquation(equation)
        # solutions, stats = maraboupy.MarabouCore.solve(loaded_network, options)

        # add lower and upper bounds + iterate over different values of epsilon
        res_dict = {}
        is_sat = False
        for label1, label2 in list(combinations(label_means.keys(), 2)):
            for epsilon in np.linspace(0.01, 0.9, self.num_eps_steps):
                print(epsilon)
                numInputVars = loaded_network.getNumInputVariables()
                for net_ind, mean_ind in zip(
                    range(int(numInputVars / 2)), range(int(numInputVars / 2))
                ):
                    cur_mean = label_means[label1][mean_ind]
                    input_ind = loaded_network.inputVariableByIndex(net_ind)
                    lower_bound = max(cur_mean - epsilon, -1)
                    upper_bound = min(cur_mean + epsilon, 1)
                    loaded_network.setLowerBound(input_ind, lower_bound)
                    loaded_network.setUpperBound(input_ind, upper_bound)

                for net_ind, mean_ind in zip(
                    range(int(numInputVars / 2), numInputVars),
                    range(int(numInputVars / 2)),
                ):
                    cur_mean = label_means[label2][mean_ind]
                    input_ind = loaded_network.inputVariableByIndex(net_ind)

                    lower_bound = max(cur_mean - epsilon, -1)
                    upper_bound = min(cur_mean + epsilon, 1)
                    loaded_network.setLowerBound(input_ind, lower_bound)
                    loaded_network.setUpperBound(input_ind, upper_bound)
                    network.setLowerBound(input_ind, lower_bound)
                    network.setUpperBound(input_ind, upper_bound)

                # THIS CODE MAKES IT SLOW AND CORRECT (A PROBLEM FOR NOT CONSISTENT
                # RESULTS WERE (MOST LIKELY) A COMBINATION OF PRECISION ERRORS AND
                # WRONG (PRECALCULATED) BOUNDS. HENCE WE CORRECT THE 'WRONG
                # PRECALCULATIONS')
                #               for net_ind in range(int(numInputVars),
                #                       loaded_network.getNumberOfVariables()):
                #                   loaded_network.setLowerBound(net_ind, -100)
                #                   loaded_network.setUpperBound(net_ind, 100)

                options = Marabou.createOptions(verbosity=2)
                solutions, stats = maraboupy.MarabouCore.solve(loaded_network, options)
                total_time = stats.getTotalTime()
                if len(solutions) > 0:
                    is_sat = True
                    curr_solution = solutions
                else:
                    # if is_sat == True:
                    is_sat = False
                res_dict[(label1, label2, epsilon)] = (total_time, is_sat)
            filtered_dict = dict(
                filter(
                    lambda x: x[0][0] == label1 and x[0][1] == label2, res_dict.items()
                )
            )
            epsilons = list(map(lambda x: x[0][2], filtered_dict.items()))
            sat_list = list(map(lambda x: x[1][1], filtered_dict.items()))
            durations = list(map(lambda x: x[1][0], filtered_dict.items()))
            times = np.cumsum(durations)

            fig, axs = plt.subplots(nrows=3, ncols=1, figsize=[20, 20])
            axs[0].plot(times, sat_list, color="blue")
            axs[1].plot(epsilons, sat_list, color="blue")
            axs[2].plot(epsilons, durations, color="blue")
            plt.title(
                f"""Total Runtime: {sum(durations)}
                    """
            )
            self.evaluation.save_figure(
                fig, "marabou_classes_" + str(label1) + "_" + str(label2)
            )
            result_df = pd.DataFrame(
                np.array([epsilons, sat_list, durations, times]).transpose(),
                columns=["epsilons", "sat_list", "durations", "times"],
            )
            self.evaluation.save_csv(
                result_df, "marabou_classes_" + str(label1) + "_" + str(label2)
            )
            plt.close("all")


def extract_solution_points(solutions, loaded_network):
    numInputVars = loaded_network.getNumInputVariables()
    inputpoint1 = []
    for ind1 in range(int(numInputVars / 2)):
        input_ind = loaded_network.inputVariableByIndex(ind1)
        inputpoint1.append(solutions[input_ind])

    inputpoint2 = []
    for ind2 in range(int(numInputVars / 2), numInputVars):
        input_ind = loaded_network.inputVariableByIndex(ind2)
        inputpoint2.append(solutions[input_ind])

    numOutputVars = loaded_network.getNumOutputVariables()
    outputpoint1 = []
    for ind1 in range(int(numOutputVars / 2)):
        output_ind = loaded_network.outputVariableByIndex(ind1)
        outputpoint1.append(solutions[output_ind])

    outputpoint2 = []
    for ind2 in range(int(numOutputVars / 2), numOutputVars):
        output_ind = loaded_network.outputVariableByIndex(ind2)
        outputpoint1.append(solutions[output_ind])

    return inputpoint1, outputpoint1, inputpoint2, outputpoint2
