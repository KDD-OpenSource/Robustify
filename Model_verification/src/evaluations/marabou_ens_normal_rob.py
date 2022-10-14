import maraboupy
from pprint import pprint
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
import os
import re
import random
import tensorflow as tf
import torch
import time
from maraboupy import Marabou
from maraboupy import MarabouCore
from itertools import combinations

from .evaluation import evaluation


class marabou_ens_normal_rob:
    def __init__(
        self,
        eval_inst: evaluation,
        name: str = "marabou_ens_normal_rob",
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
        pre_test_num_folders = 100
        part_model = None
        MNIST = True
        num_folders = 1000
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


        with open(os.path.join(simon_folder, 'models', 'alltheq.json'), 'r') as json_file:
            q_values = json.load(json_file)
        with open(os.path.join(simon_folder, 'models', 'border.json'), 'r') as json_file:
            border = json.load(json_file)
        test_model_folders = self.get_test_model_folders(
                simon_folder, q_values,
                num_folders=num_folders,part_model=part_model)
        for _ in range(20):
            if MNIST:
                normal_point = test_data[dataset.test_labels ==
                    7].sample(1)
            else: 
                normal_point = test_data[dataset.test_labels ==
                    0].sample(1)
            normal_point.columns = normal_point.columns.astype(int)

   #     sample_list = [292,
   #             732,22,1097,226,190,215,263,827,860,1110,552,214,
   #             737,934,554,1031,59,144,771
   #             ]
   #     for sample_num in sample_list:
   #         normal_point = pd.DataFrame(test_data.loc[sample_num]).transpose()

            marabou_options = Marabou.createOptions(timeoutInSeconds=60,
                    verbosity=2, initialTimeout=1, numWorkers=1)

            # model info is list of tuples with (model_number, features, q_value) 

            eps = 0.001
            eps_res_dict = {}
         #   test_model_folders = self.get_test_model_folders(
         #           simon_folder, q_values,
         #           num_folders=num_folders,part_model=part_model)

            model_info = []
            largest_errors = []
            res = []
            start_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            if parallel:
                pool = mp.Pool(int(parallel * mp.cpu_count()))
                for onnx_path in test_model_folders:
                    arg = (normal_point, onnx_path, eps, model_info, q_values
                            )
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

            eps_res_dict[str(eps)] = {}
            # calc_mean_normal_point
            #if bagging_MNIST:
            #    eps_res_dict[str(eps)]['mean_input_dict'] = dict(map(lambda x:
            #        (x['id'], float(np.mean(x['input']))), results))
            eps_res_dict[str(eps)]['largest_error_dict'] = largest_error_dict
            eps_res_dict[str(eps)]['ver_largest_error'] = verified_largest_error
            times = dict(map(lambda
                x:(x['id'],x['calc_time']), results))
            duration = sum(list(times.values()))

            adv_df = pd.DataFrame(columns=range(test_data.shape[1]))
            res_counter = 0
            num_results = len(results)
            res_list = []
            res_list.append(adv_df)
            for result in results:
                print(res_counter/num_results)
                res_counter += 1
                try:
                    res_df = pd.DataFrame([result['solution']],
                            columns=result['features'])
                    res_list.append(res_df)
                except:
                    pass
            adv_df = pd.concat(res_list, axis=0)
            adv_cand = pd.DataFrame(adv_df.mode().iloc[0]).transpose()
            norm_adv_pair = pd.concat([normal_point, adv_cand])

            mode_dict = {}
            adv_df_mode = adv_df.mode()
            tot_models = adv_df.shape[0]
            for i in adv_df.columns:
                mode_models = adv_df[adv_df[i] == adv_df_mode[i][0]].shape[0]
                other_models = (tot_models - adv_df[i].isnull().sum() -
                        mode_models)
                mode_dict[i] = mode_models/(mode_models + other_models)
                #null_models = adv_df[i].isnull().sum()

            import pdb; pdb.set_trace()

            norm_ind = str(normal_point.index[0])
            self.evaluation.save_csv(norm_adv_pair, f'norm_adv_pair_{norm_ind}')
            result_dict[norm_ind] = {
                        'eps': eps,
                        'verified_largest_error': verified_largest_error,
                        'duration': duration,
                        'ratio': verified_largest_error/border,
                        'border': border,
                        }
            self.evaluation.save_json(largest_error_dict,
                    f'largest_error_dict_{norm_ind}')
            self.evaluation.save_json(eps_res_dict,
                    f'eps_res_dict_{norm_ind}')
            self.evaluation.save_json(mode_dict,
                    f'mode_dict_{norm_ind}')
        self.evaluation.save_json(result_dict, f'results')

    def calc_largest_error(self, normal_point, onnx_path, eps, model_info,
            q_values):
        marabou_options = Marabou.createOptions(timeoutInSeconds=300)
        network = Marabou.read_onnx(onnx_path)
        numInputVars = len(network.inputVars[0][0])
        model_info.append(self.get_model_info(onnx_path, q_values,
            numInputVars, normal_point.shape[1]))
        model_input = normal_point[model_info[-1]['features']]
        model_info[-1]['input'] = list(model_input.values[0])

        #tf_model_path = os.path.join(simon_folder, folder, 'saved'
        #        )
        #tf_model = tf.keras.models.load_model(tf_model_path)
        # tf_model.layers[1].get_weights()[0][0]
        q = model_info[-1]['q']
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
        start_time = time.time()
        delta = 6
        delta_change = 3
        outputVar = network.outputVars[0][0]
        accuracy = 0.001
        while delta_change > accuracy:
            network.addInequality([outputVar], [1], q-delta)
            print('before first')
            print(model_info[-1]['id'])
            try:
                network_solution = network.solve(options=marabou_options,
                        verbose=False)
            except:
                pass
            print(model_info[-1]['id'])
            print('after first')
            network.equList = network.equList[:-1]
            if len(network_solution[0]) > 0:
                cur_best_solution = network_solution
                print(network_solution[0][outputVar])
                print(q-delta)
                delta = delta + delta_change
            else:
                # remove inequality
                network.addInequality([outputVar], [-1], -q-delta)
                print('before second')
                try:
                    network_solution = network.solve(options=marabou_options,
                            verbose=False)
                except:
                    pass
                print('after second')
                network.equList = network.equList[:-1]
                if len(network_solution[0]) > 0:
                    cur_best_solution = network_solution
                    print(-network_solution[0][outputVar])
                    print(-q-delta)
                    delta = delta + delta_change
                else:
                    delta = delta - delta_change
                    delta_change = delta_change/2

            ## add safety margin as we do only approximate

            print(delta)
            print(delta_change)
            print(q-delta)
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
        delta = delta + 2*accuracy
        model_info[-1]['largest_error'] = delta
        end_time = time.time()
        model_info[-1]['calc_time'] = end_time - start_time
        try:
            solution_point = extract_solution_point(cur_best_solution, network)
            #model_info[-1] += (solution_point[0], )
            model_info[-1]['solution'] = solution_point[0]
        except:
            pass
        return model_info

    def get_test_model_folders(self, simon_folder, q_values, num_folders=None,
            part_model=None):
        simon_folder = os.path.join(simon_folder, 'models')
        all_models = os.listdir(simon_folder)
        all_models.remove('dataset')
        #min_q = np.quantile(list(q_values.values()), 0.5)
        min_q = np.quantile(list(q_values.values()), 0.0)
        try:
            all_models.remove('readme.md')
        except:
            pass
        try:
            all_models.remove('alltheq.json')
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
                if (rand_model not in rand_models and q_values[rand_model] >=
                    min_q):
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

        models_for_removal = []
        for model in rand_models:
            if q_values[model] <= min_q:
                models_for_removal.append(model)
        # TMP
        rand_models = list(set(rand_models) - set(models_for_removal))


        for model in rand_models:
            model_folders.append(os.path.join(simon_folder, model, 'model.onnx'))
        return model_folders

    def get_model_info(self, onnx_path, q_values, num_modelFeatures,
            dataset_dim):
        model_path = onnx_path[:onnx_path.rfind('/')]
        model_number = model_path[model_path.rfind('/')+1:]
        # MNIST
        model_features = self.features_of_index(int(model_number), dataset_dim,
                num_modelFeatures)
        q_value = q_values[model_number]
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
