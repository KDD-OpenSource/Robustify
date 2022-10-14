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

# we find a non-anomalous point that is as close to an anomaly as possible
# (anomaly is given by a randomly sampled point)


class marabou_anomalous:
    def __init__(
        self,
        eval_inst: evaluation,
        name: str = "marabou_anomalous",
        num_eps_steps=100,
        eps=0.1,
    ):
        self.name = name
        self.evaluation = eval_inst
        self.num_eps_steps = num_eps_steps
        self.eps = eps

    def evaluate(self, dataset, algorithm):
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
        # sample a random point
        result_dict = {}
        for num_anomaly in range(5):
            input_sample = pd.DataFrame(
                np.random.uniform(-1, 1, size=(1, algorithm.topology[0]))
            )
            output_sample = algorithm.predict(input_sample)
            eps = self.eps

            network.saveQuery(os.path.join(marabou_folder, "saved_algorithm_marabou"))
            loaded_network = maraboupy.Marabou.load_query(
                os.path.join(marabou_folder, "saved_algorithm_marabou")
            )

            numOutputVars = len(network.outputVars[0])
            options = Marabou.createOptions(verbosity=2)
            numInputVars = len(network.inputVars[0][0])
            for ind in range(numOutputVars):
                outputInd = loaded_network.outputVariableByIndex(ind)
                inputInd = loaded_network.inputVariableByIndex(ind)
                eq1 = MarabouCore.Equation(MarabouCore.Equation.LE)
                eq1.addAddend(1, inputInd)
                eq1.addAddend(-1, outputInd)
                eq1.setScalar(eps)
                loaded_network.addEquation(eq1)

                eq2 = MarabouCore.Equation(MarabouCore.Equation.LE)
                eq2.addAddend(1, outputInd)
                eq2.addAddend(-1, inputInd)
                eq2.setScalar(eps)
                loaded_network.addEquation(eq2)

            found_closest_non_anomaly = False
            delta = 1
            delta_change = 0.5
            accuracy = 0.1
            # binary search over values of eps
            start_time = time.time()
            numOutputVars = len(network.outputVars[0])
            solution = None
            while not found_closest_non_anomaly:
                # disj_eqs = []
                for ind in range(numInputVars):
                    loaded_network.setLowerBound(
                        ind, input_sample.values[0][ind] - delta
                    )
                    loaded_network.setUpperBound(
                        ind, input_sample.values[0][ind] + delta
                    )
                network_solution, stats = maraboupy.MarabouCore.solve(
                    loaded_network, options
                )
                if len(network_solution) > 0:
                    extr_solution = extract_solution_point(network_solution, network)
                    solution = network_solution
                    if delta_change <= accuracy:
                        found_closest_non_anomaly = True
                    else:
                        delta = delta - delta_change
                else:
                    delta = delta + delta_change
                    if delta_change <= accuracy:
                        found_closest_non_anomaly = True
                delta_change = delta_change / 2

            end_time = time.time()
            tot_time = end_time - start_time
            if solution is not None:
                extr_solution = extract_solution_point(solution, network)
                plt.ylim(-1, 1)
                fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[20, 20])
                fig.suptitle(f"Calculation took {end_time - start_time}")
                ax[0].plot(input_sample.values[0], label="input_sample")
                ax[0].plot(extr_solution[0], label="input_solution")
                ax[0].set_title(
                    f"""L_infty dist is at most {delta} (up to
                        accuracy of {accuracy}), thus this is the closest
                        non-anomaly"""
                )
                ax[0].legend()
                plt.ylim(-1, 1)
                ax[1].plot(extr_solution[0], label="input_solution")
                ax[1].plot(extr_solution[1], label="output_solution")
                ax[1].set_title(
                    f"""L_infty dist is at most {self.eps}, thus we
                        have a non_anomaly"""
                )
                ax[1].legend()
                self.evaluation.save_figure(fig, f"marabou_anomalous_{num_anomaly}")
                result_dict["anomaly"] = {
                    "calc_time": tot_time,
                    "dist_within_non_anomaly": self.eps,
                    "dist_to_anomaly": delta,
                }

                self.evaluation.save_csv(
                    input_sample,
                    "input_sample",
                    subfolder=f"results_anomaly_{num_anomaly}",
                )
                self.evaluation.save_csv(
                    output_sample,
                    "output_sample",
                    subfolder=f"results_anomaly_{num_anomaly}",
                )
                self.evaluation.save_csv(
                    pd.DataFrame(extr_solution[0]),
                    "input_solution",
                    subfolder=f"results_anomaly_{num_anomaly}",
                )
                self.evaluation.save_csv(
                    pd.DataFrame(extr_solution[1]),
                    "output_solution",
                    subfolder=f"results_anomaly_{num_anomaly}",
                )

            self.evaluation.save_json(
                result_dict, f"results_marabou_anomalous_{num_anomaly}"
            )


def extract_solution_point(solution, network):
    inputpoint1 = []
    for ind1 in network.inputVars[0][0]:
        inputpoint1.append(solution[ind1])

    outputpoint1 = []
    for ind1 in network.outputVars[0]:
        outputpoint1.append(solution[ind1])

    return inputpoint1, outputpoint1
