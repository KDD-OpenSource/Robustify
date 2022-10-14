import maraboupy
from pprint import pprint
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import json
import re
import numpy as np
import os
import random
import tensorflow as tf
import torch
import time
from maraboupy import Marabou
from maraboupy import MarabouCore
from itertools import combinations
from NNet.converters.onnx2nnet import onnx2nnet

from .evaluation import evaluation


class marabou_ens_normal_rob_ae:
    def __init__(
        self,
        eval_inst: evaluation,
        name: str = "marabou_ens_normal_rob_ae",
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
        approx_search = False
        #pre_test_num_folders = 100
        part_model = None
        num_folders = 100
        simon_folder = os.path.join(os.getcwd(),
                self.cfg['multiple_models'][2:])
        if 'submodels.json' in os.listdir(os.path.join(simon_folder,
            'models')) and re.match('.*_\d_.*', self.cfg.ctx):
            with open(os.path.join(simon_folder, 'models',
                'submodels.json'), 'r') as jsonfile:
                submodel_dict = json.load(jsonfile)
            model_number = list(filter(str.isdigit, self.cfg.ctx
                ))[0]
            submodel_dict[str(model_number)] = list(map(lambda x:int(x),
                submodel_dict[str(model_number)]))
            part_model = submodel_dict[str(model_number)]


        with open(os.path.join(simon_folder, 'models', 'border.json'), 'r') as json_file:
            border = json.load(json_file)
        test_model_folders = self.get_test_model_folders(
                simon_folder, num_folders=num_folders,part_model=part_model)
        test_model_folders = test_model_folders[1:]
        for _ in range(20):
            normal_point = test_data[dataset.test_labels ==
                0].sample(1)


            marabou_options = Marabou.createOptions(timeoutInSeconds=600,
                    verbosity=2, numWorkers=1)

            # model info is list of tuples with (model_number, features, q_value) 

            eps = 0.001
            eps_res_dict = {}
          #  test_model_folders = self.get_test_model_folders(
          #          simon_folder, num_folders=num_folders,part_model=part_model)
          #  test_model_folders = test_model_folders[1:]

            model_info = []
            largest_errors = []
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
            results = list(filter(lambda x:x['test']==True, results))
            largest_error_dict = dict(map(lambda
            x:(x['id'],x['largest_error']/x['ae_div']), results))
            verified_largest_error_mean_sqrt = np.sqrt((np.array(
                list(largest_error_dict.values()))**2).sum()/len(largest_error_dict.values()))
            verified_largest_error_median = np.median(
                    np.array(list(largest_error_dict.values())))
            times = dict(map(lambda
                x:(x['id'],x['calc_time']), results))

            eps_res_dict[str(eps)] = {}
            # calc_mean_normal_point
            #eps_res_dict[str(eps)]['mean_input_dict'] = dict(map(lambda x:
                #(x['id'], float(np.mean(x['input']))), results))
            eps_res_dict[str(eps)]['largest_error_dict'] = largest_error_dict
            eps_res_dict[str(eps)]['ver_largest_error_mean_sqrt'] = verified_largest_error_mean_sqrt
            eps_res_dict[str(eps)]['ver_largest_error_median'] = verified_largest_error_median
            duration = sum(list(times.values()))

            res_counter = 0
            num_results = len(results)
            res_list = []
            for result in results:
                print(res_counter/num_results)
                res_counter += 1
                try:
                    res_df = pd.DataFrame([result['solution']])
                    res_list.append(res_df)
                except:
                    pass
            adv_df = pd.concat(res_list, axis=0)
            adv_cand = pd.DataFrame(adv_df.mode().iloc[0]).transpose()
            norm_adv_pair = pd.concat([normal_point, adv_cand])


            norm_ind = str(normal_point.index[0])
            self.evaluation.save_csv(norm_adv_pair, f'norm_adv_pair_{norm_ind}')
            result_dict[norm_ind] = {
                        'eps': eps,
                        'verified_largest_error_mean_sqrt': verified_largest_error_mean_sqrt,
                        'verified_largest_error_median': verified_largest_error_median,
                        'ratio': verified_largest_error_median/border,
                        'border': border,
                        'duration': duration
                        }
            self.evaluation.save_json(largest_error_dict,
                    f'largest_error_dict_{norm_ind}')
            self.evaluation.save_json(eps_res_dict,
                    f'eps_res_dict_{norm_ind}')
        self.evaluation.save_json(result_dict, f'results')

    def calc_largest_error(self, normal_point, onnx_path, eps, model_info):
        marabou_options = Marabou.createOptions(timeoutInSeconds=300)
        model_info.append(self.get_model_info(onnx_path))

        # tf_model_path = os.path.join(onnx_path[:onnx_path.rfind('/')], 'saved'
                # )
        # tf_model = tf.keras.models.load_model(tf_model_path)
        # tf_model.layers[1].get_weights()[0][0]
        network = Marabou.read_onnx(onnx_path)
        #onnx2nnet(onnx_path)
        #nnet_path = onnx_path[:-4] + 'nnet'

        #tf_path = onnx_path[:onnx_path.rfind('/')]+'/saved/saved_model.pb'
        ##tf_path = onnx_path[:onnx_path.rfind('/')]+'/saved/'
        #network = Marabou.read_tf(tf_path)
        #network = Marabou.read_nnet(nnet_path)
        numInputVars = len(network.inputVars[0][0])
        #q = model_info[-1]['q']
        for ind in range(numInputVars):
            network.setLowerBound(
                network.inputVars[0][0][ind], normal_point.iloc[0, ind] - eps
            )
            network.setUpperBound(
                network.inputVars[0][0][ind], normal_point.iloc[0, ind] + eps
            )
        # set outputVar
        # delta is to be in the binary search
        # q is supposed to be fixed by the training procedure
        delta = 0.5
        delta_change = 0.25
        accuracy = 0.001

        numOutputVars = len(network.outputVars[0])
        found_largest_delta = False
        start_time = time.time()
        while not found_largest_delta:
        #while delta_change > 0.00001:
            test = False
            disj_eqs = []
            #for ind in range(1):
            #ind_saver = []
            for i in range(numOutputVars):
                outputVar = network.outputVars[0][i]
                inputVar = network.inputVars[0][0][i]
                #outputVar = i + 21
                #inputVar = i
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

            try:
                network_solution = network.solve(options=marabou_options)
                if network_solution[1].getNumVisitedTreeStates() > 1:
                    test = True
                    #ind_saver.append([inputVar, outputVar])
            except:
                pass

            #solution_stats = extract_solution_stats(network_solution)
            #tot_solution_stats = add_solution_stats(tot_solution_stats, solution_stats)
            if network_solution[1].hasTimedOut():
                solution = None
                break
            if len(network_solution[0]) > 0:
                extr_solution = extract_solution_point(network_solution, network)
                diff_input = abs(
                    np.array(extr_solution[0]) - normal_point.values[0]
                ).max()
                # diff_output = abs(np.array(extr_solution[1]) -
                # output_sample.values[0]).max()
                larg_diff = abs(
                    np.array(extr_solution[1]) - np.array(extr_solution[0])
                ).max()
                solution = network_solution

                if (diff_input < eps + accuracy) and larg_diff > delta - accuracy:
                    delta = delta + delta_change
                else:
                    delte = delta - delta_change

            else:
                delta = delta - delta_change

            if delta_change <= accuracy:
                found_largest_delta = True
            delta_change = delta_change / 2

        delta = delta + 2*accuracy
        end_time = time.time()
        model_info[-1]['largest_error'] = delta
        model_info[-1]['test'] = test
        model_info[-1]['calc_time'] = end_time - start_time
        try:
            solution_point = extract_solution_point(solution, network)
            model_info[-1]['solution'] = solution_point[0]
        except:
            pass
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
            all_models.remove('border.json')
        except:
            pass
        try:
            all_models.remove('submodels.json')
        except:
            pass
        if num_folders is not None:
            rand_models = []
            while len(rand_models) < num_folders:
                rand_model = random.sample(all_models, 1)[0]
                if (rand_model not in rand_models):
                    rand_models.append(rand_model)
        else:
            rand_models = all_models
            #if '0' in rand_models:
                #rand_models.remove('0')
        if part_model is not None and not isinstance(part_model, list):
            rand_models=[part_model]
        if part_model is not None and isinstance(part_model, list):
            part_model = list(map(lambda x:str(x), part_model))
        if part_model is not None:
            rand_models = part_model
        model_folders = []
        # filter out too small q_values
        #for model in rand_models:
            #if q_values[model] < 0.1:
                #rand_models.remove(model)

        for model in rand_models:
            model_folders.append(os.path.join(simon_folder, model, 'conv.onnx'))
        return model_folders

    def get_model_info(self, onnx_path):
        model_path = onnx_path[:onnx_path.rfind('/')]
        model_number = model_path[model_path.rfind('/')+1:]
        ae_div_file = np.load(os.path.join(model_path, 'result.npz'))
        ae_div = float(ae_div_file['div'])
        #model_features = self.features_of_index(int(model_number), 28*28, 32)
        #return {'id': model_number, 'features': model_features}
        return {'id': model_number, 'ae_div': ae_div}


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


def add_solution_stats(solution1, solution2):
    if solution1 == 0:
        return solution2
    if solution2 == 0:
        return solution1
    res_solution = {}
    for key in solution1.keys():
        res_solution[key] = solution1[key] + solution2[key]
    return res_solution

# TODO:  richtige features (mit Simons skript), parallelisieren, MNIST mit
# normalisierung reintun (was gibt x - mean(x)/ 255 -> sollte zwischen [-1,1]
# liegen -> damit macht eine eps skala sinn...) 
