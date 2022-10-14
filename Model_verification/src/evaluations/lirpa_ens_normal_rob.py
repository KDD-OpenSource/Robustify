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
from itertools import combinations
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from onnx2pytorch import ConvertModel
import onnx

from .evaluation import evaluation


class lirpa_ens_normal_rob:
    def __init__(
        self,
        eval_inst: evaluation,
        name: str = "lirpa_ens_normal_rob",
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
        import pdb; pdb.set_trace()
        simon_folder = os.path.join(os.getcwd(),
                self.cfg['multiple_models'][2:])
        merged_model = onnx.load(os.path.join(simon_folder, 'models',
            'merged.onnx'))
        torch_model = ConvertModel(merged_model)
        import pdb; pdb.set_trace()
        input_tensor = torch.zeros(1,784)

        test_data = dataset.test_data()
        normal_point = torch.tensor(test_data[dataset.test_labels ==
            7].sample(1).values)
        import pdb; pdb.set_trace()
        ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
        my_input = BoundedTensor(normal_point, ptb)

        lirpa_model = BoundedModule(torch_model, my_input)
        pred = lirpa_model(normal_point)
        lb, ub = lirpa_model.compute_bounds(x=(my_input,), method="CROWN")
        import pdb; pdb.set_trace()



#        test_data = dataset.test_data()
#        parallel = True
#        result_dict = {}
#        approx_search = True
#        pre_test_num_folders = 100
#        for _ in range(1):
#            #normal_point = self.cust_scaler(test_data[dataset.test_labels ==
#                #7].sample(1))
#            normal_point = test_data[dataset.test_labels ==
#                7].sample(1)
#
#            simon_folder = os.path.join(os.getcwd(),
#                    self.cfg['multiple_models'][2:])
#            test_model_folders = self.get_test_model_folders(simon_folder,
#                    num_folders = 100)
#            # test_model_folders = self.get_test_model_folders(simon_folder
#                    # )
#
#
#            # tf_model_path = os.path.join(test_model_folders[0], 'saved'
#                   # )
#            # print(tf_model_path)
#            # tf_model_path = '/home/bboeing/NNLinSubfct/Code/models/trained_models/simon2/models/4366/saved'
#            # print(tf_model_path)
#            # tf_model = tf.keras.models.load_model(tf_model_path)
#            # tf_model.layers[1].get_weights()[0][0]
#
#
#            #test_model_folders = self.get_test_model_folders(simon_folder)
#            marabou_options = Marabou.createOptions(timeoutInSeconds=60)
#
#            # model info is list of tuples with (model_number, features, q_value) 
#            with open(os.path.join(simon_folder, 'models', 'alltheq.json'), 'r') as json_file:
#                q_values = json.load(json_file)
#
#            score_threshold = 0.15
#            eps = 0.05
#            eps_change = 0.025
#            start_time = time.time()
#            eps_res_dict = {}
#            eps_acc = 0.00001
#            while eps_change > eps_acc:
#                if approx_search and eps_change > 2*eps_acc:
#                    test_model_folders = self.get_test_model_folders(simon_folder,
#                            num_folders = pre_test_num_folders)
#                elif approx_search:
#                    test_model_folders = self.get_test_model_folders(
#                            simon_folder)
#                model_info = []
#                largest_errors = []
#                pool = mp.Pool(int(1/2 * mp.cpu_count()))
#                res = []
#                start_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
#                if parallel:
#                    for onnx_path in test_model_folders:
#                        arg = (normal_point, onnx_path, eps, model_info, q_values)
#                        res.append(pool.apply_async(self.calc_largest_error, args=(arg
#                            )))
#                    pool.close()
#                    pool.join()
#                    results = [x.get()[0] for x in res]
#                else:
#                    for onnx_path in test_model_folders:
#                        res.append(self.calc_largest_error(normal_point, onnx_path,
#                            eps, model_info, q_values))
#                        results = res[0]
#                end_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
#                #results = [x.get()[0] for x in res]
#                largest_errors = list(map(lambda x:x['largest_error'], results))
#                largest_error_dict = dict(map(lambda
#                    x:(x['id'],x['largest_error']), results))
#                verified_largest_error = np.sqrt(
#                        (np.array(largest_errors)**2).sum()/len(largest_errors))
#                eps_res_dict[str(eps)] = {}
#                eps_res_dict[str(eps)]['mean_input_dict'] = dict(map(lambda x:
#                    (x['id'], float(np.mean(x['input']))), results))
#                # calc_mean_normal_point
#                eps_res_dict[str(eps)]['largest_error_dict'] = largest_error_dict
#                eps_res_dict[str(eps)]['ver_largest_error'] = verified_largest_error
#
#                print(eps)
#                print(verified_largest_error)
#                print(score_threshold)
#                if verified_largest_error > score_threshold:
#                    eps = eps - eps_change
#                else:
#                    eps = eps + eps_change
#                eps_change = eps_change / 2
#            end_time = time.time()
#            # once we have the verified largest error we can increase/decrease
#            # eps
#
#
#
#            adv_df = pd.DataFrame(columns=range(784))
#            for result in results:
#                try:
#                    res_df = pd.DataFrame([result['solution']],
#                            columns=result['features'])
#                    adv_df = pd.concat([adv_df, res_df], axis=0)
#                except:
#                    pass
#
#            adv_df_mode = adv_df.mode()
#            adv_cand = pd.DataFrame(adv_df.mode().iloc[0]).transpose()
#            norm_adv_pair = pd.concat([normal_point, adv_cand])
#            norm_ind = str(normal_point.index[0])
#            self.evaluation.save_csv(norm_adv_pair, f'norm_adv_pair_{norm_ind}')
#            result_dict[norm_ind] = {
#                        'eps': eps,
#                        'verified_largest_error': verified_largest_error,
#                        'duration': end_time - start_time
#                        }
#            self.evaluation.save_json(eps_res_dict,
#                    f'eps_res_dict_{norm_ind}')
#        self.evaluation.save_json(result_dict, f'results')
#
#    def calc_largest_error(self, normal_point, onnx_path, eps, model_info, q_values):
#        marabou_options = Marabou.createOptions(timeoutInSeconds=300)
#        model_info.append(self.get_model_info(onnx_path, q_values))
#        model_input = normal_point[model_info[-1]['features']]
#        model_info[-1]['input'] = list(model_input.values[0])
#        #tf_model_path = os.path.join(simon_folder, folder, 'saved'
#        #        )
#        #tf_model = tf.keras.models.load_model(tf_model_path)
#        # tf_model.layers[1].get_weights()[0][0]
#        network = Marabou.read_onnx(onnx_path)
#        numInputVars = len(network.inputVars[0][0])
#        q = model_info[-1]['q']
#        for ind in range(numInputVars):
#            network.setLowerBound(
#                network.inputVars[0][0][ind], model_input.iloc[0, ind] - eps
#            )
#            network.setUpperBound(
#                network.inputVars[0][0][ind], model_input.iloc[0, ind] + eps
#            )
#        # set outputVar
#        # delta is to be in the binary search
#        # q is supposed to be fixed by the training procedure
#        delta = 6
#        delta_change = 3
#        outputVar = network.outputVars[0][0]
#        while delta_change > 0.0000001:
#            # eq1 = MarabouCore.Equation(MarabouCore.Equation.GE)
#            # eq1.addAddend(-1, outputVar)
#            # eq1.setScalar(delta-q)
#
#            # eq2 = MarabouCore.Equation(MarabouCore.Equation.GE)
#            # eq2.addAddend(1, outputVar)
#            # eq2.setScalar(delta+q)
#
#
#
#            # eq1 = maraboucore.equation(maraboucore.equation.le)
#            # eq1.addaddend(1, outputvar)
#            # eq1.setscalar(q-delta)
#
#            # eq2 = maraboucore.equation(maraboucore.equation.ge)
#            # eq2.addaddend(1, outputvar)
#            # eq2.setscalar(q+delta)
#            # disjunction = [[eq1], [eq2]]
#            # network.disjunctionList = []
#            # network.addDisjunctionConstraint(disjunction)
#            network.addInequality([outputVar], [1], q-delta)
#            network_solution = network.solve(options=marabou_options)
#            network.equList = network.equList[:-1]
#            if len(network_solution[0]) > 0:
#                cur_best_solution = network_solution
#                print(network_solution[0][outputVar])
#                print(q-delta)
#                delta = delta + delta_change
#            else:
#                # remove inequality
#                network.addInequality([outputVar], [-1], -q-delta)
#                network_solution = network.solve(options=marabou_options)
#                network.equList = network.equList[:-1]
#                if len(network_solution[0]) > 0:
#                    cur_best_solution = network_solution
#                    print(-network_solution[0][outputVar])
#                    print(-q-delta)
#                    delta = delta + delta_change
#                else:
#                    delta = delta - delta_change
#                    delta_change = delta_change/2
#            print(delta)
#            print(model_info[-1]['id'])
#
#
#
#            # if len(network_solution[0]) > 0:
#                # print(network_solution[0][outputVar])
#                # print(q+delta)
#                # print(q-delta)
#                # print(delta)
#                # if (network_solution[0][outputVar] > q+delta or
#                        # network_solution[0][outputVar] < q-delta):
#                    # cur_best_solution = network_solution
#                    # delta = delta + delta_change
#                # else:
#                    # delta = delta - delta_change
#            # else:
#                # delta = delta - delta_change
#            # delta_change = delta_change/2
#            # print(delta)
#
#        #model_info[-1] = model_info[-1] + (delta, )
#        model_info[-1]['largest_error'] = delta
#        try:
#            solution_point = extract_solution_point(cur_best_solution, network)
#            #model_info[-1] += (solution_point[0], )
#            model_info[-1]['solution'] = solution_point[0]
#        except:
#            pass
#        return model_info
#
#    def get_test_model_folders(self, simon_folder, num_folders=None,
#            part_model=None):
#        simon_folder = os.path.join(simon_folder, 'models')
#        all_models = os.listdir(simon_folder)
#        all_models.remove('dataset')
#        try:
#            all_models.remove('readme.md')
#        except:
#            pass
#        try:
#            all_models.remove('alltheq.json')
#        except:
#            pass
#        if num_folders is not None:
#            rand_models = random.sample(all_models, num_folders)
#            #if '0' in rand_models:
#                #rand_models.remove('0')
#        else:
#            rand_models = all_models
#            #if '0' in rand_models:
#                #rand_models.remove('0')
#        if part_model is not None:
#            rand_models=[part_model]
#        model_folders = []
#        for model in rand_models:
#            model_folders.append(os.path.join(simon_folder, model, 'model.onnx'))
#        return model_folders
#
#    def get_model_info(self, onnx_path, q_values):
#        model_path = onnx_path[:onnx_path.rfind('/')]
#        model_number = model_path[model_path.rfind('/')+1:]
#        model_features = self.features_of_index(int(model_number), 28*28, 32)
#        q_value = q_values[model_number]
#        #return (model_number, model_features, q_value)
#        return {'id': model_number, 'features': model_features, 'q': q_value}
#
#
#    def features_of_index(self, index, num_feat, bag):
#        np.random.seed(index)
#        seed = np.random.randint(10000000)
#        np.random.seed(seed)
#        base = list(range(num_feat))
#        np.random.shuffle(base)
#        return base[:bag]
#
#    def cust_scaler(self, sample):
#        # sample is a df with one row
#        return (sample - sample.mean().mean())/255
#
#def extract_solution_point(solution, network):
#    solution = solution[0]
#    inputpoint1 = []
#    for ind1 in network.inputVars[0][0]:
#        inputpoint1.append(solution[ind1])
#
#    outputpoint1 = []
#    for ind1 in network.outputVars[0]:
#        outputpoint1.append(solution[ind1])
#
#    return inputpoint1, outputpoint1
#
## TODO:  richtige features (mit Simons skript), parallelisieren, MNIST mit
## normalisierung reintun (was gibt x - mean(x)/ 255 -> sollte zwischen [-1,1]
## liegen -> damit macht eine eps skala sinn...) 
