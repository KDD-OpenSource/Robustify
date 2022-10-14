import pandas as pd
from maraboupy import Marabou
from maraboupy import MarabouCore
import torch.nn as nn
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from .evaluation import evaluation
from ..algorithms.neural_net import smallest_k_dist_loss


import sys
sys.path.append('/home/bboeing/NNLinSubfct/Code/eran_files/ELINA/python_interface')
sys.path.append('/home/bboeing/NNLinSubfct/Code/eran_files/deepg/code')
#sys.path.append('/home/bboeing/NNLinSubfct/Code/eran_files/deepg/ERAN/tf_verify')
sys.path.append('/home/bboeing/NNLinSubfct/Code/eran_files/tf_verify')
from read_net_file import *
from eran import ERAN

# LiRPA
#from auto_LiRPA import BoundedModule, BoundedTensor, PertubationLpNorm



class code_test_on_trained_model:
    def __init__(self, eval_inst: evaluation, name: str =
            "code_test_on_trained_model"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        sample = dataset.test_data().sample(1)
        sample = torch.tensor(sample.values)
        import pdb; pdb.set_trace()
        lol = algorithm.module(sample)
        test = nn.Bilinear(30,30, 5)
        #test = nn.Sequential(algorithm.module, nn.Bilinear(30,30, 5))
        #lol2 = test(sample)

        algorithm.module.add_linfty_layer()
        lol2 = algorithm.module(sample)

        import pdb; pdb.set_trace()


        network_path = self.get_network(algorithm, dataset)
        import pdb; pdb.set_trace()
        #filename, file_extension = os.path.splitext(network_path)
        model, is_conv = read_onnx_net(network_path)
        eran = ERAN(model, is_onnx=True)
        box  = [[-1,1] for _ in range(algorithm.topology[0])]
        specLB = [interval[0] for interval in box]
        specUB = [interval[1] for interval in box]
        domain = 'deeppoly'
        import pdb; pdb.set_trace()
        hold, inn, nlb, nub, _, _ = eran.analyze_box(specLB, specUB, domain,
                #config.timeout_lp, config.timeout_milp,
                #config.use_default_heuristic, constraints)
                30.0, 30.0,
                True, None)
        import pdb; pdb.set_trace()


        #self.evaluation.save_json(result_dict, 'avg_min_fctborder_dist')


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
        import pdb; pdb.set_trace()
        torch.onnx.export(
            algorithm.module.get_neural_net(),
            randomInput.float(),
            os.path.join(onnx_folder, "saved_algorithm.onnx"),
        )
        network = Marabou.read_onnx(
            os.path.join(onnx_folder, "saved_algorithm.onnx"),
            outputName=str(2 * len(algorithm.module.get_neural_net()) + 1),
        )
        import pdb; pdb.set_trace()
        return os.path.join(onnx_folder, 'saved_algorithm.onnx')
