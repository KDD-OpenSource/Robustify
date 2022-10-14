import maraboupy
from pprint import pprint
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
import os
import random
import tensorflow as tf
import torch
import time
from maraboupy import Marabou
from maraboupy import MarabouCore
from itertools import combinations

from .evaluation import evaluation


class marabou_svdd_normal_rob:
    def __init__(
        self,
        eval_inst: evaluation,
        name: str = "marabou_svdd_normal_rob",
        num_eps_steps=100,
        delta=0.1,
        cfg = None,
    ):
        self.name = name
        self.evaluation = eval_inst
        self.num_eps_steps = num_eps_steps
        self.desired_delta = delta
        self.cfg = cfg

    def evaluate(self, dataset, algorithm):
        test_data = dataset.test_data()
        parallel = 0.25
        result_dict = {}
        #robustness_sum = 0
        #truly_verif_sum = 0
        for _ in range(20):
            normal_point = test_data[dataset.test_labels ==
                0].sample(1)
            #normal_point = pd.DataFrame(np.random.uniform(
             #   0,1, size=(36)), index=range(36)).transpose().set_index([
             #       np.random.randint(1,1000,1)])

            simon_folder = os.path.join(os.getcwd(),
                    self.cfg['multiple_models'][2:])

            marabou_options = Marabou.createOptions(timeoutInSeconds=600,
                    verbosity=2, numWorkers=1)

            eps = 0.01

            start_time = time.time()
            res_dict = {}
            test_model_folders = self.get_test_model_folders(
                    simon_folder)

            model_info = []
            res = []
            start_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            if parallel:
                pool = mp.Pool(int(parallel * mp.cpu_count()))
                for onnx_path in test_model_folders:
                    arg = (normal_point, onnx_path, eps, model_info)
                    res.append(pool.apply_async(self.calc_largest_error, args=(arg
                        )))
                pool.close()
                pool.join()
                results = [x.get()[0] for x in res]
            else:
                for onnx_path in test_model_folders:
                    res.append(self.calc_largest_error(normal_point, onnx_path,
                        eps, model_info))
                    results = res[0]

            largest_error_vals = dict(map(lambda
                x:(x['id'],x['largest_error']), results))
            times = dict(map(lambda
                x:(x['id'],x['calc_time']), results))
            max_dists = dict(map(lambda
                x:(x['id'],x['max_dist']), results))
            ratios = dict(map(lambda
                x:(x['id'],x['error_ratio']), results))
            robustness_vals = dict(map(lambda
                x:(x['id'],x['robustness']), results))
            end_time = time.time()
            # add res_dict
            sample_res_dict = {}
            for key in largest_error_vals.keys():
                sample_res_dict[key] = {
                        'largest_error': largest_error_vals[key],
                        'duration': times[key],
                        'max_dist': max_dists[key],
                        'ratio': ratios[key],
                        'robustness': str(robustness_vals[key]),
                        }


            norm_ind = str(normal_point.index[0])
            result_dict[norm_ind] = sample_res_dict
        self.evaluation.save_json(result_dict, f'results')

    def calc_largest_error(self, normal_point, onnx_path, eps, model_info):
        marabou_options = Marabou.createOptions(timeoutInSeconds=300)
        model_info.append(self.get_model_info(onnx_path))
        # add line to calculate \tau
        network = Marabou.read_onnx(onnx_path)
        numInputVars = len(network.inputVars[0][0])
        center = model_info[-1]['c']
        tau = model_info[-1]['tau']
        numOutputVars = len(network.outputVars[0])
        max_dist = np.sqrt((tau**2)/numOutputVars)
        for ind in range(numInputVars):
            network.setLowerBound(
                network.inputVars[0][0][ind], normal_point.values[0][ind] - eps
            )
            network.setUpperBound(
                network.inputVars[0][0][ind], normal_point.values[0][ind] + eps
            )
        start_time = time.time()
        #is_robust = True
        current_solution = None
        delta = 6
        delta_change = 3
        delta_seq = []
        l_infty_seq = []
        l_2_seq = []
        sol_seq = []
        l_infty_error = None
        l_2_error = None
        solution = None
        accuracy = 0.001
        while delta_change > accuracy:
            found_ce = False
            for outputInd in range(numOutputVars):
                network.disjunctionList = []
                outputVar = network.outputVars[0][outputInd]
                eq1 = MarabouCore.Equation(MarabouCore.Equation.GE)
                eq1.addAddend(1, outputVar)
                eq1.setScalar(delta + center[outputInd])

                eq2 = MarabouCore.Equation(MarabouCore.Equation.GE)
                eq2.addAddend(-1, outputVar)
                eq2.setScalar(delta - center[outputInd])

                network.addInequality([outputVar],[-1], -center[outputInd]-delta)
                network_solution = network.solve(options=marabou_options,
                    verbose=False)
                network.equList = network.equList[:-1]
                if len(network_solution[0]) == 0:
                    network.addInequality([outputVar],[1], center[outputInd]-delta)
                    network_solution = network.solve(options=marabou_options,
                        verbose=False)
                    network.equList = network.equList[:-1]



                delta_old = delta
                if len(network_solution[0]) > 0:
                    current_solution = network_solution
                    delta = delta + delta_change
                    delta_change = delta_change/2
                    solution = np.array(extract_solution_point(
                        current_solution, network)[1])
                    l_infty_error = abs(np.array(extract_solution_point(
                        current_solution, network)[1])-np.array(center)).max()
                    l_2_error = np.sqrt(((np.array(extract_solution_point(
                        current_solution,
                        network)[1])-np.array(center))**2).sum())
                    #print(l_infty_error)

                    #print(delta_old)
                    found_ce = True
                    break
            if found_ce == False:
                delta = delta - delta_change
                delta_change = delta_change/2
                l_infty_error = None

                #print(delta_old)
            print(delta)
            print(delta_change)
            l_infty_seq.append(l_infty_error)
            delta_seq.append(delta_old)
            l_2_seq.append(l_2_error)
            sol_seq.append(solution)

        #truly_verified = True
        end_time = time.time()

        delta = delta + 2*accuracy
        model_info[-1]['largest_error'] = delta
        model_info[-1]['calc_time'] = end_time - start_time
        model_info[-1]['max_dist'] = max_dist
        model_info[-1]['error_ratio'] = delta/max_dist
        model_info[-1]['robustness'] = (delta/max_dist < 1)

        #model_info[-1]['truly_verified'] = truly_verified 
        try:
            solution_point = extract_solution_point(current_solution, network)
            #model_info[-1] += (solution_point[0], )
            model_info[-1]['solution'] = solution_point[0]
        except:
            pass
        # asses if true
        return model_info

    def get_test_model_folders(self, simon_folder, num_folders=None,
            part_model=None):
        simon_folder = os.path.join(simon_folder, 'models')
        all_models = os.listdir(simon_folder)
        all_models.remove('dataset')
        try:
            all_models.remove('readme.md')
        except:
            pass
        try:
            all_models.remove('alltheq.json')
        except:
            pass
        if num_folders is not None:
            rand_models = []
            while len(rand_models) < num_folders:
                rand_model = random.sample(all_models, 1)[0]
                if rand_model not in rand_models:
                    rand_models.append(rand_model)
        else:
            rand_models = all_models
            #if '0' in rand_models:
                #rand_models.remove('0')
        if part_model is not None:
            rand_models=[part_model]
        model_folders = []

        for model in rand_models:
            model_folders.append(os.path.join(simon_folder, model, 'pmodel.onnx'))
        return model_folders

    def get_model_info(self, onnx_path):
        model_path = onnx_path[:onnx_path.rfind('/')]
        model_number = model_path[model_path.rfind('/')+1:]
        with open(os.path.join(model_path, 'c.json')) as json_file:
            center = json.load(json_file)
        with open(os.path.join(model_path, 'border.json')) as json_file:
            border = json.load(json_file)
        return {'id': model_number, 'c': center, 'tau': border}


    def features_of_index(self, index, num_feat, bag):
        np.random.seed(index)
        seed = np.random.randint(10000000)
        np.random.seed(seed)
        base = list(range(num_feat))
        np.random.shuffle(base)
        return base[:bag]

    def cust_scaler(self, sample):
        # sample is a df with one row
        return (sample - sample.mean().mean())/255

def extract_solution_point(solution, network):
    #import pdb; pdb.set_trace()
    solution = solution[0]
    inputpoint1 = []
    for ind1 in network.inputVars[0][0]:
        inputpoint1.append(solution[ind1])

    outputpoint1 = []
    for ind1 in network.outputVars[0]:
        outputpoint1.append(solution[ind1])

    return inputpoint1, outputpoint1

# TODO:  richtige features (mit Simons skript), parallelisieren, MNIST mit
# normalisierung reintun (was gibt x - mean(x)/ 255 -> sollte zwischen [-1,1]
# liegen -> damit macht eine eps skala sinn...) 
