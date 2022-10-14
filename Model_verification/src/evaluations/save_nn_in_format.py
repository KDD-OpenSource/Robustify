import pandas as pd
import torch.nn as nn
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from maraboupy import Marabou

from .evaluation import evaluation
from ..algorithms.neural_net import smallest_k_dist_loss


class save_nn_in_format:
    def __init__(self, eval_inst: evaluation, name: str =
            "save_nn_in_format"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        network = self.get_network(algorithm, dataset)
        self.save_marabouQuery(network)
        self.save_nn_txt(algorithm)

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
        os.makedirs(onnx_folder, exist_ok=True)
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

    def save_marabouQuery(self, network):
        network.saveQuery(os.path.join(self.evaluation.run_folder, 'alg_query'))

    def save_nn_txt(self, algorithm):
        nnet = algorithm.module.get_neural_net()
        architecture = list(map(lambda x: x.strip(),
            str(nnet).split('\n')))[1:-1]
        nnet_file = os.path.join(self.evaluation.run_folder,
                'neural_net_file.txt')
        with open(nnet_file, 'w') as file:
            for line in architecture:
                file.write(line)
                file.write('\n')

            for layer in nnet:
                if isinstance(layer, nn.Linear):
                    weights = layer.weight.detach().numpy()
                    bias = layer.bias.detach().numpy()
                    file.write(str(weights))
                    file.write('\n')
                    file.write(str(bias))
                    file.write('\n')

