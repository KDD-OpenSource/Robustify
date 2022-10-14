import maraboupy
from pprint import pprint
import multiprocessing as mp
import re
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


class marabou_ens_normal_rob_submodels:
    def __init__(
        self,
        eval_inst: evaluation,
        name: str = "marabou_ens_normal_rob_submodels",
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
        parallel = True
        approx_search = False
        pre_test_num_folders = 100
        part_model = None
        num_folders = 1000
        simon_folder = os.path.join(os.getcwd(),
                self.cfg['multiple_models'][2:])
        with open(os.path.join(simon_folder, 'models', 'alltheq.json'), 'r') as json_file:
            q_values = json.load(json_file)
        marabou_options = Marabou.createOptions(timeoutInSeconds=60,
                verbosity=2, initialTimeout=1, numWorkers=1)
        sample_submodel_filenames = os.listdir(os.path.join(simon_folder,
            'remaining_models'))



        for file_name in sample_submodel_filenames:
            result_dict = {}
            sample_index = re.findall(r'\d+', file_name)[0]
            file_path = os.path.join(simon_folder, 'remaining_models',
                    file_name)
            with open(file_path, 'r') as json_file:
                remaining_models = json.load(json_file)['remaining_models']
            test_model_folders = self.get_test_model_folders(
                    simon_folder, q_values,
                    part_models=remaining_models)

            points = []
            orig_point = pd.DataFrame(
                    test_data.loc[int(sample_index)]).transpose()
            points.append(orig_point)
            for _ in range(20):
                points.append(test_data[dataset.test_labels ==
                    7].sample(1))


            for i, normal_point in enumerate(points):
            # model info is list of tuples with (model_number, features, q_value) 
            #with open(os.path.join(simon_folder, 'models', 'alltheq.json'), 'r') as json_file:
                #q_values = json.load(json_file)

                eps = 0.001
                start_time = time.time()
                eps_res_dict = {}
                model_info = []
                largest_errors = []
                res = []
                start_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
                if parallel:
                    pool = mp.Pool(int(1 * mp.cpu_count()))
                    for onnx_path in test_model_folders:
                        arg = (normal_point, onnx_path, eps, model_info, q_values)
                        res.append(pool.apply_async(self.calc_largest_error, args=(arg
                            )))
                    pool.close()
                    pool.join()
                    results = [x.get()[0] for x in res]
                else:
                    for onnx_path in test_model_folders:
                        res.append(self.calc_largest_error(normal_point, onnx_path,
                            eps, model_info, q_values))
                        results = res[0]
                largest_error_dict = dict(map(lambda
                x:(x['id'],x['largest_error']), results))
                verified_largest_error = np.sqrt((np.array(
                    list(largest_error_dict.values()))**2).sum()/len(largest_error_dict.values()))

            #eps_res_dict[str(eps)] = {}
            # calc_mean_normal_point
            #eps_res_dict[str(eps)]['mean_input_dict'] = dict(map(lambda x:
                #(x['id'], float(np.mean(x['input']))), results))
            #eps_res_dict[str(eps)]['largest_error_dict'] = largest_error_dict
            #eps_res_dict[str(eps)]['ver_largest_error'] = verified_largest_error
                end_time = time.time()

            #adv_df = pd.DataFrame(columns=range(784))
            #res_counter = 0
            #num_results = len(results)
            #res_list = []
            #res_list.append(adv_df)
            #for result in results:
                #print(res_counter/num_results)
                #res_counter += 1
                #try:
                    #res_df = pd.DataFrame([result['solution']],
                            #columns=result['features'])
                    #res_list.append(res_df)
                #except:
                    #pass
            #adv_df = pd.concat(res_list, axis=0)
            #adv_cand = pd.DataFrame(adv_df.mode().iloc[0]).transpose()
            #norm_adv_pair = pd.concat([normal_point, adv_cand])


                norm_ind = str(normal_point.index[0])
            #self.evaluation.save_csv(norm_adv_pair, f'norm_adv_pair_{norm_ind}')
                result_dict[norm_ind] = {
                            'eps': eps,
                            'verified_largest_error': verified_largest_error,
                            'duration': end_time - start_time
                            }
                result_dict_interm = {}
                result_dict_interm[norm_ind] = result_dict[norm_ind]
                self.evaluation.save_json(result_dict_interm,
                        f'results_{sample_index}_{i}')
            #self.evaluation.save_json(largest_error_dict,
                    #f'largest_error_dict_{norm_ind}')
            #self.evaluation.save_json(eps_res_dict,
                    #f'eps_res_dict_{norm_ind}')
            self.evaluation.save_json(result_dict, f'results_{sample_index}')

    def calc_largest_error(self, normal_point, onnx_path, eps, model_info, q_values):
        marabou_options = Marabou.createOptions(timeoutInSeconds=300)
        model_info.append(self.get_model_info(onnx_path, q_values))
        model_input = normal_point[model_info[-1]['features']]
        model_info[-1]['input'] = list(model_input.values[0])

        #tf_model_path = os.path.join(simon_folder, folder, 'saved'
        #        )
        #tf_model = tf.keras.models.load_model(tf_model_path)
        # tf_model.layers[1].get_weights()[0][0]
        network = Marabou.read_onnx(onnx_path)
        numInputVars = len(network.inputVars[0][0])
        q = model_info[-1]['q']
        if abs(model_input).max().max() == 0:
            model_info[-1]['largest_error'] = q
            return model_info
        for ind in range(numInputVars):
            network.setLowerBound(
                network.inputVars[0][0][ind], model_input.iloc[0, ind] - eps
            )
            network.setUpperBound(
                network.inputVars[0][0][ind], model_input.iloc[0, ind] + eps
            )
        # set outputVar
        # delta is to be in the binary search
        # q is supposed to be fixed by the training procedure
        delta = 6
        delta_change = 3
        outputVar = network.outputVars[0][0]
        while delta_change > 0.00001:
            # eq1 = MarabouCore.Equation(MarabouCore.Equation.GE)
            # eq1.addAddend(-1, outputVar)
            # eq1.setScalar(delta-q)

            # eq2 = MarabouCore.Equation(MarabouCore.Equation.GE)
            # eq2.addAddend(1, outputVar)
            # eq2.setScalar(delta+q)



            # eq1 = maraboucore.equation(maraboucore.equation.le)
            # eq1.addaddend(1, outputvar)
            # eq1.setscalar(q-delta)

            # eq2 = maraboucore.equation(maraboucore.equation.ge)
            # eq2.addaddend(1, outputvar)
            # eq2.setscalar(q+delta)
            # disjunction = [[eq1], [eq2]]
            # network.disjunctionList = []
            # network.addDisjunctionConstraint(disjunction)
            network.addInequality([outputVar], [1], q-delta)
            print(model_info[-1]['id'])
            try:
                network_solution = network.solve(options=marabou_options,
                        verbose=False)
            except:
                pass
            print(model_info[-1]['id'])
            network.equList = network.equList[:-1]
            if len(network_solution[0]) > 0:
                cur_best_solution = network_solution
                print(network_solution[0][outputVar])
                print(q-delta)
                delta = delta + delta_change
            else:
                # remove inequality
                network.addInequality([outputVar], [-1], -q-delta)
                try:
                    network_solution = network.solve(options=marabou_options,
                            verbose=False)
                except:
                    pass
                network.equList = network.equList[:-1]
                if len(network_solution[0]) > 0:
                    cur_best_solution = network_solution
                    print(-network_solution[0][outputVar])
                    print(-q-delta)
                    delta = delta + delta_change
                else:
                    delta = delta - delta_change
                    delta_change = delta_change/2
            print(delta)
            print(model_info[-1]['id'])



            # if len(network_solution[0]) > 0:
                # print(network_solution[0][outputVar])
                # print(q+delta)
                # print(q-delta)
                # print(delta)
                # if (network_solution[0][outputVar] > q+delta or
                        # network_solution[0][outputVar] < q-delta):
                    # cur_best_solution = network_solution
                    # delta = delta + delta_change
                # else:
                    # delta = delta - delta_change
            # else:
                # delta = delta - delta_change
            # delta_change = delta_change/2
            # print(delta)

        #model_info[-1] = model_info[-1] + (delta, )
        model_info[-1]['largest_error'] = delta
        try:
            solution_point = extract_solution_point(cur_best_solution, network)
            #model_info[-1] += (solution_point[0], )
            model_info[-1]['solution'] = solution_point[0]
        except:
            pass
        return model_info

    def get_test_model_folders(self, simon_folder, q_values, num_folders=None,
            part_models=None):
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
                if (rand_model not in rand_models and q_values[rand_model] >=
                    0.1):
                    rand_models.append(rand_model)
        else:
            rand_models = all_models
            #if '0' in rand_models:
                #rand_models.remove('0')
        if part_models is not None:
            if not isinstance(part_models, list):
                part_models = [part_models]
            rand_models=part_models
        model_folders = []
        # filter out too small q_values
        for model in rand_models:
            if q_values[model] < 0.1:
                rand_models.remove(model)

        for model in rand_models:
            model_folders.append(os.path.join(simon_folder, model, 'model.onnx'))
        return model_folders

    def get_model_info(self, onnx_path, q_values):
        model_path = onnx_path[:onnx_path.rfind('/')]
        model_number = model_path[model_path.rfind('/')+1:]
        model_features = self.features_of_index(int(model_number), 28*28, 32)
        q_value = q_values[model_number]
        #return (model_number, model_features, q_value)
        return {'id': model_number, 'features': model_features, 'q': q_value}


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

# TODO:  richtige features (mit Simons skript), parallelisieren, MNIST mit
# normalisierung reintun (was gibt x - mean(x)/ 255 -> sollte zwischen [-1,1]
# liegen -> damit macht eine eps skala sinn...) 
