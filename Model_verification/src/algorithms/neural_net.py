from pprint import pprint

import os
import abc
from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull
import gc
import math
import copy
import hashlib
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from functools import reduce



class neural_net:
    def __init__(
        self,
        name: str,
        num_epochs: int = 100,
        dynamic_epochs: bool = False,
        batch_size: int = 20,
        lr: float = 1e-3,
        seed: int = None,
        save_interm_models = False,
    ):
        self.name = name
        self.num_epochs = num_epochs
        self.dynamic_epochs = dynamic_epochs
        self.batch_size = batch_size
        self.lr = float(lr)
        self.seed = seed
        self.save_interm_models = save_interm_models

    def __str__(self) -> str:
        return self.name

    @abc.abstractmethod
    def load(self):
        pass

    def calc_min_avg_border_dist(self, dataset, subsample = None):
        if subsample is not None:
            dataset = dataset.sample(subsample)
        if subsample == 0:
            return None
        data_loader = DataLoader(
            dataset=dataset.values,
            batch_size=20,
            drop_last=False,
            pin_memory=True,
        )
        closest_dists = []
        for inst_batch in data_loader:
            for instance in inst_batch:
                subfcts = self.get_neuron_border_subFcts(self.module,
                        instance)
                dists = sorted(
                        self.get_dists_from_border_subFcts(instance,
                    subfcts))
                closest_dists.append(
                        dists[0],
                            )
        avg_min_fctborder_dist = np.mean(closest_dists)
        return avg_min_fctborder_dist

    def get_top_k_closest_neurons(self, instance, k=1, dists = True):
        # calculate dists with respective neurons return neurons only
        # a neuron will be a pair of (i,j) where i is the network layer and j
        # is the neuron index. The network layer corresponds to the layer in
        # neural_not_mod
        subfcts = self.get_neuron_border_subFcts(self.module,
                instance, subfct_inds = True)
        dists = self.get_dists_from_border_subFcts(instance, subfcts,
                signed_dists = True)
        sorted_dists = dict(sorted(dists.items(), key=lambda x:abs(x[1])))
        return list(sorted_dists.items())[:k]

    #def push_closest_fctborders_set(self, X: pd.DataFrame, dist_fct,
    def push_closest_fctborders_set(self, X: pd.DataFrame, cond_fct,
            bias_shift, subset = None):
        if subset is not None:
            X = X.sample(subset)
        if subset == 0:
            return 0
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        ctr = 0
        data_len = len(data_loader)
        pushed_samples = 0
        for inst_batch in data_loader:
            for inst in inst_batch:
                #condition = self.get_push_condition(condition_type)
                condition = cond_fct(self.k_dist_model, inst)
                pushed_samples += self.push_closest_fctborder_inst(inst,
                        bias_shift, condition)
                ctr += 1
        return pushed_samples

    def push_closest_fctborder_inst(self, instance, bias_shift, condition =
            None, advanced = True):
        push_neurons_dists = self.get_top_k_closest_neurons(instance, k=1, dists =
                True)
        neural_net = self.module.get_neural_net()
        for neuron_ind, dist in push_neurons_dists:
            with torch.no_grad():
                # increase distance by 10 percent through change of bias
                if self.check_push_condition(condition, instance, dist):
                #if max_dist is not None and abs(dist) < max_dist:
                    normalized_weight = np.linalg.norm(
                            neural_net[neuron_ind[0]].weight[neuron_ind[1]])
                    neural_net[
                            neuron_ind[0]].bias[neuron_ind[1]] += (
                                    (bias_shift - 1) *
                                    (dist*normalized_weight))
                    if advanced:
                        self.adjust_followup_bias(neuron_ind, dist, bias_shift,
                                normalized_weight, neural_net)
                    return 1
                else:
                    return 0

    def adjust_followup_bias(self, neuron_ind, dist, bias_shift,
            normalized_weight, neural_net):
        with torch.no_grad():
            shift_val = (bias_shift - 1) * (dist*normalized_weight)
            # we are sure to have a relu layer inbetween, hence + 2
            spread_layer = neuron_ind[0] + 2
            # now we get the column as we are interested in weight
            # going from the part. neuron
            spread_layer_weights = neural_net[
                    spread_layer].weight[:,neuron_ind[1]]
            spread_layer_weights_sum = sum(abs(spread_layer_weights))
            for tar_bias_ind, tar_bias in enumerate(neural_net[
                spread_layer].bias):
                cur_weight = neural_net[
                        spread_layer].weight[tar_bias_ind,
                                neuron_ind[1]]
                neural_net[
                        spread_layer].bias[tar_bias_ind] = (tar_bias
                        - (cur_weight/spread_layer_weights_sum)*shift_val)

    def check_push_condition(self, condition, instance, dist):
        if condition is None:
            return False
        if isinstance(condition, np.float32):
            # max_dist condition: if the dist to the next border is smaller
            # than some threshold we push
            if abs(dist) < condition:
                return True
            else:
                return False
        elif type(condition) == type(instance):
            inst_fct = self.get_lin_subfct(self.module, instance)
            for sample_ind in range(condition.shape[0]):
                cond_fct = self.get_lin_subfct(self.module,
                        condition[sample_ind])
                if cond_fct != inst_fct:
                    return True
            return False


    def save(self, path, subfolder = None):
        os.makedirs(os.path.join("./models/trained_models", self.name,
            'subfolder'), exist_ok=True)
        torch.save(
            self.module.state_dict(),
            os.path.join("./models/trained_models", self.name, "subfolder/model.pth"),
        )

        torch.save(
            self.__dict__,
            os.path.join("./models/trained_models", self.name, "subfolder/model_detailed.pth"),
        )

        if subfolder:
            path = path + '/' + subfolder
            os.makedirs(path, exist_ok = True)
        torch.save(self.module.state_dict(), os.path.join(path, "model.pth"))
        torch.save(
            self.__dict__,
            os.path.join(path, "model_detailed.pth"),
        )

    def get_data_loader(self, X: pd.DataFrame):
        if self.dynamic_epochs:
            data_sampler = torch.utils.data.RandomSampler(
                data_source=X, replacement=True, num_samples=500
            )
            self.num_epochs = self.num_epochs * int(X.shape[0] / 500)

            data_loader = DataLoader(
                num_workers=0,
                dataset=X.values,
                batch_size=self.batch_size,
                drop_last=True,
                pin_memory=True,
                shuffle=True,
                sampler=data_sampler,
            )
        else:
            data_loader = DataLoader(
                num_workers=0,
                dataset=X.values,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
                pin_memory=True,
            )
        return data_loader

    def calc_layer_stats_weights(self):
        neural_net = self.module.get_neural_net()
        stat_dict = {}
        for ind, layer in enumerate(neural_net):
            if isinstance(layer, nn.Linear):
                weights = layer.weight.detach().numpy()
                abs_weights = abs(weights)
                stat_dict[ind] = {}
                stat_dict[ind]['mean'] = abs_weights.mean()
                stat_dict[ind]['std'] = abs_weights.std()
                stat_dict[ind]['min'] = weights.min()
                stat_dict[ind]['max'] = weights.max()
        return stat_dict


    def calc_layer_stats_biases(self):
        neural_net = self.module.get_neural_net()
        stat_dict = {}
        for ind, layer in enumerate(neural_net):
            if isinstance(layer, nn.Linear):
                if layer.bias is not None:
                    biases = layer.bias.detach().numpy()
                    abs_biases = abs(biases)
                    stat_dict[ind] = {}
                    stat_dict[ind]['mean'] = abs_biases.mean()
                    stat_dict[ind]['std'] = abs_biases.std()
                    stat_dict[ind]['min'] = biases.min()
                    stat_dict[ind]['max'] = biases.max()
        return stat_dict


    # def calc_linfct_volume(self, instances):
#        halfspaces = np.array([[-1,0,0],
#            [0,-1,0],
#            [1,1,-1],
#            ])
#        feasible_point = np.array([0.3,0.3])
#        hs = HalfspaceIntersection(halfspaces, feasible_point)
#        ch = ConvexHull(hs.intersections)


        # test functionality with 100 equations in 50 dimensions (randomly
        # THE TEST ESSENTIALLY SHOWS THAT WE CANNOT CALCULATE IT PRECISELY
        # (EVEN THOUGH IT IS THEORETICALLY CLEAR HOW TO DO IT/ THERE EVEN
        # EXISTS THE PYTHON FUNCTIONALITY FOR IT) -> TOO MANY FACETS ON THE
        # POLYTOPE
        # space_dim = 50
        # num_eq = 52
        # inner_point = np.random.uniform(low=-1,high=1, size=(space_dim))
        # equations = []
        # while len(equations) < num_eq:
            # linfct = np.random.uniform(low=-1, high=1, size=space_dim + 1)
            # test_inside = (np.dot(linfct[:space_dim], inner_point) +
                    # linfct[space_dim] < 0)
            # if test_inside:
                # equations.append(linfct)
        # halfspaces = np.array(equations)
        # hs = HalfspaceIntersection(halfspaces, inner_point)
        # print(len(hs.intersections))
        # convhull = ConvexHull(hs.intersections)

        # generate them and always let a given point be in there...)
        # calc borders, calc convex hull of borders ()


    def get_lin_subfct(self, neural_net_mod, instance, max_layer=None):
        # if max_layer is added, we calcuate it until and including the layer
        forward_help_fct = smallest_k_dist_loss(1)
        neural_net = copy.deepcopy(neural_net_mod.get_neural_net())
        #calculate_inst calculates function until but excluding max_layer
        mat, bias, relus, _ = forward_help_fct.calculate_inst(
            neural_net, instance, max_layer=max_layer
        )

        nn_layers = list(neural_net_mod.get_neural_net())
        relu_key_list = []
        for layer_ind in range(len(nn_layers)):
            if isinstance(nn_layers[layer_ind], nn.ReLU):
                relu_key_list.append(str(layer_ind))
        relu_dict = {}
        for layer_ind, relu_vals in zip(relu_key_list, relus):
            relu_dict[layer_ind] = relu_vals.numpy().flatten()
        lin_subfct = linearSubfunction(
            mat.detach().numpy(), bias.detach().numpy(), relu_dict
        )

        return lin_subfct


    def check_final_linFct(self, neural_net_mod, instance):
        _, intermed_results = neural_net_mod(instance)
        lin_subfct = self.get_lin_subfct(neural_net_mod, instance)
        mat = lin_subfct.matrix
        bias = lin_subfct.bias
        linFctRes = np.matmul(mat, instance) + bias
        linFctRes = linFctRes.float()
        output_key = list(intermed_results.keys())[-1]
        acc = 1e-6
        if (
            (abs((linFctRes - intermed_results[output_key]) / linFctRes)) > acc
        ).sum() == 0:
            return 0
        else:
            return 1

    def check_interm_linFcts(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        _, intermed_results = neural_net_mod(instance)

        linSubFcts = self.get_interm_linFct(neural_net_mod, instance)
        linLayers = self.get_linLayers(neural_net_mod)
        mats = list(map(lambda x: x.matrix, linSubFcts))
        biases = list(map(lambda x: x.bias, linSubFcts))
        num_inacc = 0
        for fct_ind in range(len(mats)):
            linFctRes = np.matmul(mats[fct_ind], instance) + biases[fct_ind]
            linFctRes = linFctRes.float()
            output_key = str(linLayers[fct_ind])
            acc = 1e-6
            if (
                (abs((linFctRes - intermed_results[output_key]) / linFctRes)) > acc
            ).sum() != 0:
                num_inacc += 1

        return num_inacc

    def count_lin_subfcts(self, neural_net_mod, X):
        functions = []
        neural_net_mod = copy.deepcopy(neural_net_mod)
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        ctr = 0
        for inst_batch in data_loader:
            for inst in inst_batch:
                functions.append(self.get_lin_subfct(neural_net_mod, inst.float()))
                ctr += 1
                # print(ctr)
        unique_functions = []
        functions_counter = []
        # there is probably a smarter way to do this?
        for function in functions:
            if function not in unique_functions:
                unique_functions.append(function)
                functions_counter.append(1)
            else:
                index = unique_functions.index(function)
                functions_counter[index] += 1
        return list(zip(unique_functions, functions_counter))

    def assign_lin_subfcts(self, neural_net_mod, X):
        functions = []
        neural_net_mod = copy.deepcopy(neural_net_mod)
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        inst_func_pairs = []
        for inst_batch in data_loader:
            for inst in inst_batch:
                inst_func_pairs.append(
                    (inst.float(), self.get_lin_subfct(neural_net_mod, inst.float()))
                )
        return inst_func_pairs

    def assign_bias_feature_imps(self, neural_net_mod, X):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        inst_imp_pairs = []
        for inst_batch in data_loader:
            for inst in inst_batch:
                feature_imp = self.apply_lin_func_without_bias(neural_net_mod, inst)
                bias_imp = self.lin_func_bias_imp(neural_net_mod, inst)
                feature_imp_abs_sum = abs(feature_imp).sum()
                bias_imp_abs_sum = abs(bias_imp).sum()
                imp_sum = feature_imp_abs_sum + bias_imp_abs_sum
                inst_imp_pairs.append(
                    (inst.float(), imp_sum, feature_imp_abs_sum, bias_imp_abs_sum)
                )
        return inst_imp_pairs

    def assign_lin_subfcts_ind(self, neural_net_mod, X):
        inst_func_pairs = self.assign_lin_subfcts(neural_net_mod, X)
        unique_functions = []
        functions_counter = []
        inst_ind_pairs = []
        # there is probably a smarter way to do this?
        for inst, function in inst_func_pairs:
            if function not in unique_functions:
                unique_functions.append(function)
                functions_counter.append(1)
                inst_ind_pairs.append((inst, len(functions_counter)))
            else:
                index = unique_functions.index(function)
                inst_ind_pairs.append((inst, index))
        return inst_ind_pairs


    def assign_top_k_border_dists(self, neural_net_mod, X, k):
        top_k_dists = []
        neural_net_mod = copy.deepcopy(neural_net_mod)
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        sample_dist_pairs = []
        ctr = 0
        for inst_batch in data_loader:
            for inst in inst_batch:
                try:
                    top_k_dists_sum = (
                        smallest_k_dist_loss(k, border_dist=True, fct_dist=False)(
                            inst.unsqueeze(0), neural_net_mod
                        )[0]
                        .detach()
                        .numpy()
                    )
                except:
                    top_k_dists_sum = np.array([0])
                if top_k_dists_sum is not None:
                    sample_dist_pairs.append((inst, top_k_dists_sum))
        return sample_dist_pairs

    def assign_border_dists(self, neural_net_mod, X):
        return self.assign_top_k_border_dists(neural_net_mod, X, 1)

    def assign_most_far_border_dists(self, neural_net_mod, X):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        sample_dist_pairs = []
        ctr = 0
        for inst_batch in data_loader:
            for inst in inst_batch:
                closest_func_Boundary = self.get_most_far_funcBoundary(
                    neural_net_mod, inst.float()
                )
                if closest_func_Boundary is not None:
                    sample_dist_pairs.append((inst.float(), closest_func_Boundary[1]))
                ctr += 1
                print(ctr)
        return sample_dist_pairs

    def get_points_of_linsubfcts(self, neural_net_mod, X):
        functions = {}
        neural_net_mod = copy.deepcopy(neural_net_mod)
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        for inst_batch in data_loader:
            for inst in inst_batch:
                function = self.get_lin_subfct(neural_net_mod, inst.float())
                if function not in functions.keys():
                    functions[function] = []
                    functions[function].append(inst.float())
                else:
                    functions[function].append(inst.float())
        return functions

    def get_signature(self, neural_net_mod, instance):
        intermed_res = neural_net_mod(instance)[1]
        layer_key_list = list(zip(list(neural_net_mod.get_neural_net()), intermed_res))
        relu_key_list = []
        for layer in layer_key_list:
            if isinstance(layer[0], nn.ReLU):
                relu_key_list.append(layer[1])
        signature_dict = {}
        for key in relu_key_list:
            signature_dict[key] = intermed_res[key]
        for key, value in signature_dict.items():
            signature_dict[key] = self.binarize(value)
        return signature_dict

    def binarize(self, tensor):
        nparray = tensor.detach().numpy()
        npmask = nparray > 0
        nparray[npmask] = 1
        return nparray

    def get_interm_linFct(self, neural_net_mod, instance):
        # sees each layer including relu
        neural_net_mod = copy.deepcopy(neural_net_mod)
        linSubFcts = []
        linLayers = self.get_linLayers(neural_net_mod)
        for layer_ind in linLayers:
            linSubFcts.append(
                self.get_lin_subfct(neural_net_mod, instance, layer_ind + 1)
            )
        return linSubFcts

    def get_interm_pre_relu_linFct(self, neural_net_mod, instance):
        # calculates lin_subfct without relus
        neural_net_mod = copy.deepcopy(neural_net_mod)
        linSubFcts = []
        linLayers = self.get_linLayers(neural_net_mod)
        for layer_ind in linLayers[:-1]:
            linSubFcts.append(
                self.get_lin_subfct(neural_net_mod, instance, layer_ind)
            )
        return linSubFcts

    def get_linLayers(self, neural_net_mod):
        linLayers = []
        for layer_ind in range(len(list(neural_net_mod.get_neural_net()))):
            if isinstance(neural_net_mod.get_neural_net()[layer_ind], nn.Linear):
                linLayers.append(layer_ind)
        return linLayers

    def get_closest_funcBoundary(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        instance_sig = self.get_signature(neural_net_mod, instance)
        # calculate crosspoints
        cross_points = []
        neuron_subFcts = self.get_neuron_border_subFcts(neural_net_mod, instance)
        for neuron_subFct in neuron_subFcts:
            cross_point = self.get_cross_point(neuron_subFct, instance)
            if cross_point is not None:
                cross_points.append(cross_point)
        if None in cross_points:
            print("NoneType in cross_points")
            return None
        if len(cross_points) == 0:
            print("No cross_points found")
            return None
        diffs = torch.stack(cross_points) - instance
        dists = np.linalg.norm(np.array(diffs), axis=1)
        try:
            cross_points_sorted = sorted(
                list(zip(dists, cross_points)), key=lambda x: x[0]
            )
        except:
            import pdb

            pdb.set_trace()
        for dist_cross_point_pair in cross_points_sorted:
            if self.check_true_cross_point(
                neural_net_mod, instance, dist_cross_point_pair[1]
            ):
                return (dist_cross_point_pair[1], dist_cross_point_pair[0])

    def get_top_k_funcBoundDists(self, neural_net_mod, instance, k):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        # calculate crosspoints
        cross_points = []
        neuron_subFcts = self.get_neuron_border_subFcts(neural_net_mod, instance)
        for neuron_subFct in neuron_subFcts:
            cross_point = self.get_cross_point(neuron_subFct, instance)
            if cross_point is not None:
                cross_points.append(cross_point)
        if None in cross_points:
            print("NoneType in cross_points")
            return None
        if len(cross_points) == 0:
            print("No cross_points found")
            return None
        diffs = torch.stack(cross_points) - instance
        dists = np.linalg.norm(np.array(diffs), axis=1)
        cross_points_sorted = sorted(list(zip(dists, cross_points)), key=lambda x: x[0])
        return cross_points_sorted[:k]

    def get_top_k_dists_sum(self, neural_net_mod, instance, k):
        cross_points = self.get_top_k_funcBoundDists(neural_net_mod, instance, k)
        dists = list(map(lambda x: x[0], cross_points))
        return sum(dists)

    def get_most_far_funcBoundary(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        # calculate crosspoints
        cross_points = []
        neuron_subFcts = self.get_neuron_border_subFcts(neural_net_mod, instance)
        for neuron_subFct in neuron_subFcts:
            cross_point = self.get_cross_point(neuron_subFct, instance)
            if cross_point is not None:
                cross_points.append(cross_point)
        if None in cross_points:
            print("NoneType in cross_points")
            return None
        if len(cross_points) == 0:
            print("No cross_points found")
            return None
        diffs = torch.stack(cross_points) - instance
        dists = np.linalg.norm(np.array(diffs), axis=1)
        cross_points_sorted = sorted(list(zip(dists, cross_points)), reverse=True)
        for dist_cross_point_pair in cross_points_sorted:
            if self.check_true_cross_point(
                neural_net_mod, instance, dist_cross_point_pair[1]
            ):
                return (dist_cross_point_pair[1], dist_cross_point_pair[0])

    def get_closest_afterCross_fct(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        instance_sig = self.get_signature(neural_net_mod, instance)
        # calculate crosspoints
        cross_points = []
        neuron_subFcts = self.get_neuron_border_subFcts(neural_net_mod, instance)
        for neuron_subFct in neuron_subFcts:
            cross_point = self.get_cross_point(neuron_subFct, instance)
            if cross_point is not None:
                cross_points.append(cross_point)
        if None in cross_points:
            print("NoneType in cross_points")
            return None
        if len(cross_points) == 0:
            print("No cross_points found")
            return None
        diffs = torch.stack(cross_points) - instance
        dists = np.linalg.norm(np.array(diffs), axis=1)
        cross_points_sorted = sorted(list(zip(dists, cross_points)))
        for dist_cross_point_pair in cross_points_sorted:
            if self.check_true_cross_point(
                neural_net_mod, instance, dist_cross_point_pair[1]
            ):
                cross_point = dist_cross_point_pair[1]
                after_cross = instance + 1.01 * (cross_point - instance)
                return (after_cross, self.get_lin_subfct(neural_net_mod, after_cross))

    def get_most_far_afterCross_fct(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        instance_sig = self.get_signature(neural_net_mod, instance)
        # calculate crosspoints
        cross_points = []
        neuron_subFcts = self.get_neuron_border_subFcts(neural_net_mod, instance)
        for neuron_subFct in neuron_subFcts:
            cross_point = self.get_cross_point(neuron_subFct, instance)
            if cross_point is not None:
                cross_points.append(cross_point)
        if None in cross_points:
            print("NoneType in cross_points")
            return None
        if len(cross_points) == 0:
            print("No cross_points found")
            return None
        diffs = torch.stack(cross_points) - instance
        dists = np.linalg.norm(np.array(diffs), axis=1)
        cross_points_sorted = sorted(list(zip(dists, cross_points)), reverse=True)
        for dist_cross_point_pair in cross_points_sorted:
            if self.check_true_cross_point(
                neural_net_mod, instance, dist_cross_point_pair[1]
            ):
                cross_point = dist_cross_point_pair[1]
                after_cross = instance + 1.01 * (cross_point - instance)
                return (after_cross, self.get_lin_subfct(neural_net_mod, after_cross))

    def check_true_cross_point(self, neural_net_mod, instance, cross_point):
        inst_linFct = self.get_lin_subfct(neural_net_mod, instance)
        cross_point_linFct = self.get_lin_subfct(neural_net_mod, cross_point)
        before_cross = instance + 0.99 * (cross_point - instance)
        after_cross = instance + 1.01 * (cross_point - instance)
        before_cross_linFct = self.get_lin_subfct(neural_net_mod, before_cross)
        after_cross_linFct = self.get_lin_subfct(neural_net_mod, after_cross)
        if inst_linFct == before_cross_linFct and inst_linFct != after_cross_linFct:
            return True
        else:
            # for most far only
            return True
            if inst_linFct != before_cross_linFct:
                print("CrossPoint has different function")
            if inst_linFct == after_cross_linFct:
                print("After crossing it has still the same function")
            print("False")
            return False

    def get_closest_funcBoundaries(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        instance_sig = self.get_signature(neural_net_mod, instance)
        linSubFcts = self.get_interm_pre_relu_linFct(neural_net_mod, instance)
        # calculate crosspoints
        cross_points = []
        neuron_subFcts = self.get_neuron_border_subFcts(neural_net_mod, instance)
        for neuron_subFct in neuron_subFcts:
            cross_points.append(self.get_cross_point(neuron_subFct, instance))

        closest_cross_points = []
        for point in cross_points:
            point_signatures.append(self.get_signature(neural_net_mod, point))
            if isequal_sig_dict(instance_sig, point_signatures[-1]):
                sig_counter += 1
                closest_cross_points.append(point)
        return closest_cross_points

    def get_all_funcBoundaries(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        instance_sig = self.get_signature(neural_net_mod, instance)
        # linSubFcts = self.get_interm_linFct(neural_net_mod, instance)
        # calculate crosspoints
        cross_points = []
        neuron_subFcts = self.get_neuron_border_subFcts(neural_net_mod, instance)
        for neuron_subFct in neuron_subFcts:
            cross_points.append(self.get_cross_point(neuron_subFct, instance))
        return cross_points

    def erase_ReLU(self, neural_net_mod):
        new_layers = []
        for layer in neural_net_mod.get_neural_net():
            if not isinstance(layer, nn.ReLU):
                new_layers.append(layer)
        return IntermediateSequential(*new_layers)

    def interpolate(point_from: torch.tensor, point_to: torch.tensor, num_steps):
        inter_points = [
            point_from + int_var / (num_steps - 1) * point_to
            for int_var in range(num_steps)
        ]
        return torch.stack(inter_points)

    def get_neuron_subFcts(self, neural_net_mod, instance):
        linSubFcts = self.get_interm_linFct(neural_net_mod, instance)
        neuron_subFcts = []
        for layer in linSubFcts:
            neuron_subFcts.extend(self.get_neuron_subFcts_from_layer(layer))
        return neuron_subFcts

    def get_neuron_border_subFcts(self, neural_net_mod, instance, subfct_inds =
            False):
        linSubFcts = self.get_interm_pre_relu_linFct(neural_net_mod, instance)
        lin_layers = self.get_linLayers(neural_net_mod)
        if isinstance(neural_net_mod.get_neural_net()[-1], nn.Linear):
            lin_layers = lin_layers[:-1]
        neuron_subFcts = []
        for layer_ind, layer in zip(lin_layers, linSubFcts):
            if subfct_inds:
                subfcts = self.get_neuron_subFcts_from_layer(layer, neuron_inds
                        = True)
                subfcts_layer = list(zip(len(subfcts)*[layer_ind], subfcts))
                subfcts_format = [((x[0], x[1][0]), x[1][1]) for x in
                        subfcts_layer]
                neuron_subFcts.extend(subfcts_format)
            else:
                neuron_subFcts.extend(self.get_neuron_subFcts_from_layer(layer
                    ))
        if subfct_inds:
            return dict(neuron_subFcts)
        else:
            return neuron_subFcts

    def get_dists_from_border_subFcts(self, instance, border_subfcts,
            signed_dists = False):
        # signed_dists indicates whether the information on which side of the
        # border (given by a linear function) the instance is.
        if isinstance(border_subfcts, dict):
            dists = {}
            for key, neuron in border_subfcts.items():
                normal_vector = neuron[0] / np.linalg.norm(neuron[0])
                dist_to_border = (np.matmul(neuron[0], instance) + neuron[1]) / np.linalg.norm(
                    neuron[0]
                )
                if signed_dists:
                    dists[key]= dist_to_border.tolist()
                else:
                    dists[key]= abs(dist_to_border.tolist())
            return dists
        elif isinstance(border_subfcts, list):
            dists = []
            for neuron in border_subfcts:
                normal_vector = neuron[0] / np.linalg.norm(neuron[0])
                dist_to_border = (np.matmul(neuron[0], instance) + neuron[1]) / np.linalg.norm(
                    neuron[0]
                )
                if signed_dists:
                    dists.append(dist_to_border.tolist())
                else:
                    dists.append(abs(dist_to_border.tolist()))
            return dists
        else:
            raise Exception('Not defined what to do with input')


    def get_neuron_subFcts_from_layer(self, layer_subfunction, neuron_inds =
            False):
        neuron_subFcts = []
        if neuron_inds:
            for neuron_ind in range(len(layer_subfunction.matrix)):
                neuron_subFcts.append((neuron_ind,
                    (
                        layer_subfunction.matrix[neuron_ind],
                        layer_subfunction.bias[neuron_ind]
                    ))
                )
        else:
            for neuron_ind in range(len(layer_subfunction.matrix)):
                neuron_subFcts.append(
                    (
                        layer_subfunction.matrix[neuron_ind],
                        layer_subfunction.bias[neuron_ind],
                    )
                )
        return neuron_subFcts

    def get_neuron_subFct_cross_point(self, neural_net_mod, instance):
        # maybe this function is irrelevant...
        neural_net_mod = copy.deepcopy(neural_net_mod)
        linSubFcts = self.get_interm_pre_relu_linFct(neural_net_mod, instance)
        subFct_cross_points = []
        for layer in linSubFcts:
            for neuron_ind in range(len(layer.matrix)):
                neuron_sub_fct = (layer.matrix[neuron_ind], layer.bias[neuron_ind])
                cross_point = self.get_cross_point(neuron_sub_fct, instance)
                subFct_cross_points.append((neuron_sub_fct, cross_point))
        # result has the followin structure:
        # ((fct_vector, bias), cross_point)
        return subFct_cross_points

    def get_cross_point(self, neuron, instance):
        # neuron[0] is the weight_vector
        # neuron[1] is the bias
        if np.linalg.norm(neuron[0]) == 0:
            return None
        normal_vector = neuron[0] / np.linalg.norm(neuron[0])
        dist_to_border = (np.matmul(neuron[0], instance) + neuron[1]) / np.linalg.norm(
            neuron[0]
        )
        cross_point = instance - dist_to_border * normal_vector
        if cross_point.isnan().sum() > 0:
            return None
        else:
            return cross_point

    def check_interp_signatures(self, point_from, point_to, num_steps, neural_net_mod):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        inter_points = neural_net.interpolate(point_from, point_to, num_steps)
        point_from_sig = self.get_signature(neural_net_mod, point_from)
        for point in inter_points:
            point_inter_sig = self.get_signature(neural_net_mod, point)
            if isequal_sig_dict(point_from_sig, point_inter_sig) == False:
                return False
        return True

    def get_fctBoundary(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        closest_boundaries = self.get_closest_funcBoundaries(
            self, neural_net_mod, instance
        )
        # find counterexample in points
        # interpolate between inst and counterexample to find another boundary
        # (binary search)
        # iterate until with 10k sample we find no inlier
        pass

    def get_fct_area(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        # boundaries = self.get_all_funcBoundaries(self, neural_net_mod, instance)
        # linSubFcts = self.get_interm_linFct(neural_net_mod, instance)
        # neuron_subFcts = self.get_neuron_subFcts(neural_net_mod, instance)
        neuron_subFcts_cross_points = self.get_neuron_subFct_cross_point(
            neural_net_mod, instance
        )
        # for layer in linSubFcts:
        # neuron_subFcts.extend(self.get_neuron_subFcts(layer))
        limiting_neurons = []
        inst_lin_subFct = self.get_lin_subfct(neural_net_mod, instance)
        for neuron_sub_fct, cross_point in neuron_subFcts_cross_points:
            # check if the following equation will be right
            crossed_inst = instance + 1.1 * (cross_point - instance)
            crossed_inst_lin_subFct = self.get_lin_subfct(neural_net_mod, crossed_inst)
            if inst_lin_subFct != crossed_inst_lin_subFct:
                limiting_neurons.append(neuron_sub_fct)
        return limiting_neurons

    def lin_func_feature_imp(self, neural_net_mod, instance):
        lin_func = self.get_lin_subfct(neural_net_mod, instance)
        feature_imps = lin_func.matrix.sum(axis=0)
        return feature_imps

    def apply_lin_func_without_bias(self, neural_net_mod, instance):
        lin_func = self.get_lin_subfct(neural_net_mod, instance)
        result = np.matmul(lin_func.matrix, instance)
        return result

    def lin_func_bias_imp(self, neural_net_mod, instance):
        lin_func = self.get_lin_subfct(neural_net_mod, instance)
        bias_imps = lin_func.bias
        return bias_imps

    def lrp_ae(self, neural_net_mod, instance, gamma=1):
        input_relevance = np.zeros(instance.size())
        neural_net_mod = copy.deepcopy(neural_net_mod)
        output, intermed_res = neural_net_mod(instance)
        layer_inds = range(len(neural_net_mod.get_neural_net()))
        gamma = gamma
        error = nn.MSELoss()(instance, output)
        relevance = torch.tensor((np.array(instance) - output.detach().numpy()) ** 2)

        relevance_bias = 0
        for layer_ind in layer_inds[::-1]:
            layer = neural_net_mod.get_neural_net()[layer_ind]
            if layer_ind != 0:
                activation = intermed_res[str(layer_ind - 1)]
            else:
                activation = instance
            if isinstance(layer, nn.Linear):
                # first if: not last layer
                # second if: right layer for relu
                if layer_ind != (len(layer_inds) - 1) and isinstance(
                    neural_net_mod.get_neural_net()[layer_ind + 1], nn.ReLU
                ):
                    relevance = self.lrp_linear_relu(
                        activation, layer, relevance, gamma
                    )
                    relevance_bias += relevance[-1]
                    relevance = relevance[:-1]
                else:
                    relevance = self.lrp_linear(activation, layer, relevance)
                    relevance_bias += relevance[-1]
                    relevance = relevance[:-1]
        return relevance.detach().numpy(), relevance_bias

    def lrp_linear(self, activation, layer, relevance):
        act_bias = torch.cat((activation, torch.tensor([1])))
        layer_weight = torch.cat((layer.weight, layer.bias.reshape(-1, 1)), dim=1)
        relevance = (
            (
                (act_bias * layer_weight).transpose(0, 1)
                / ((act_bias * layer_weight).sum(axis=1))
            )
            * relevance
        ).sum(axis=1)
        return relevance

    def lrp_linear_relu(self, activation, layer, relevance, gamma):
        pos_weights = copy.deepcopy(layer.weight)
        pos_weights.detach()[pos_weights < 0] = 0
        act_bias = torch.cat((activation, torch.tensor([1])))
        layer_weight = torch.cat((layer.weight, layer.bias.reshape(-1, 1)), dim=1)
        pos_layer_weight = torch.cat((pos_weights, layer.bias.reshape(-1, 1)), dim=1)
        relevance = (
            (
                (act_bias * (layer_weight + gamma * pos_layer_weight)).transpose(0, 1)
                / ((act_bias * (layer_weight + gamma * pos_layer_weight)).sum(axis=1))
            )
            * relevance
        ).sum(axis=1)
        return relevance


class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs

class LinftyIntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)
        # test with batch...
        #inp_out = torch.cat([input, output]).reshape(1,-1)
        #linear = nn.Linear(input.shape[1],input.shape[1], bias=False)
        #res = linear(input)

        #return inp_out, intermediate_outputs
        return output
        #return intermediate_outputs


class linearSubfunction:
    def __init__(self, matrix, bias, signature):
        self.matrix = matrix
        self.bias = bias
        self.signature = signature
        self.name = self.__hash__()

    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        return (
            (abs(self.matrix - other.matrix)).sum()
            + (abs(self.bias - other.bias)).sum()
        ) < 000000000.1

    def __key(self):
        return (self.matrix.tobytes(), self.bias.tobytes())

    def __hash__(self):
        return hash(self.__key())

    def dist(self, other):
        return self.dist_bias(other) + self.dist_mat(other)

    def dist_bias(self, other):
        diff = self.bias - other.bias
        squared = diff ** 2
        summed = squared.sum()
        result = math.sqrt(summed)
        return result

    def dist_mat(self, other):
        diff = self.matrix - other.matrix
        scalar_prod = np.trace(np.matmul(diff, diff.transpose()))
        result = math.sqrt(scalar_prod)
        return result

    def dist_sign(self, other):
        return signature_dist(self.signature, other.signature)

    def dist_sign_weighted(self, other):
        if self.signature.keys() != other.signature.keys():
            print("Cannot compare these signatures")
            return None
        distance = 0
        for key in self.signature.keys():
            num_changes = int(abs(self.signature[key] - other.signature[key]).sum())
            # calculates /2 and round up  (mapping [1,3,5,7] to [1,2,3,4])
            # factor_inverse = 1/(int((int(key)/2)+1))
            factor = int((int(key) / 2) + 1)
            distance += factor * num_changes
        return distance




def isequal_sig_dict(signature1, signature2):
    if signature1.keys() != signature2.keys():
        return False
    keys = signature1.keys()
    return all(np.array_equal(signature1[key], signature2[key]) for key in keys)


def signature_dist(signature1, signature2):
    if signature1.keys() != signature2.keys():
        print("Cannot compare these signatures")
        return None
    distance = 0
    for key in signature1.keys():
        distance += int(abs(signature1[key] - signature2[key]).sum())
    return distance


class smallest_k_dist_loss(nn.Module):
    def __init__(
        self,
        k,
        border_dist=False,
        penal_dist=None,
        fct_dist=False,
        cross_point_sampling=False,
        max_layer=None,
    ):
        self.k = k
        self.border_dist = border_dist
        self.penal_dist = penal_dist
        self.fct_dist = fct_dist
        self.cross_point_sampling = cross_point_sampling
        self.max_layer = max_layer
        super(smallest_k_dist_loss, self).__init__()

    def forward(self, inputs, module):
        border_dist_sum = 0
        fct_dist_sum = 0
        for inst in inputs:
            border_dist, fct_dist = self.smallest_k_dists(module, inst)
            border_dist_sum += border_dist
            fct_dist_sum += fct_dist
        return border_dist_sum, fct_dist_sum

    def smallest_k_dists(self, neural_net_mod, instance):
        border_result = 0
        fct_result = 0
        if not self.border_dist and not self.fct_dist:
            return 0, 0
        neural_net = neural_net_mod.get_neural_net()

        fct_dists = torch.tensor([])
        if self.cross_point_sampling:
            inst_mat, inst_bias, relus, relu_acts = self.calculate_inst(
                neural_net, instance, dists=False, max_layer=self.max_layer
            )
        else:
            inst_mat, inst_bias, dists, cross_points, relu_acts = self.calculate_inst(
                neural_net, instance, dists=True, max_layer=self.max_layer
            )
            if dists.isnan().sum() > 0:
                return 0, 0
            if dists.isinf().sum() > 0:
                return 0, 0
            top_k_dists = torch.topk(input=dists, k=self.k, largest=False)
            if self.border_dist:
                # penalize every border that is closer than self.penal_dist (version from paper)
                if self.penal_dist:
                    top_k_maxed = torch.max(
                        torch.zeros(self.k), 1 - (top_k_dists[0] / self.penal_dist)
                    )
                else:
                    # simply return the (true) distance (for evaluation purposes)
                    top_k_maxed = top_k_dists[0]

                border_result += top_k_maxed.sum() / self.k

        if self.fct_dist:
            comp_points = []
            if self.cross_point_sampling:
                for _ in range(self.k):
                    comp_points.append(
                        instance
                        + np.random.normal(
                            0, self.penal_dist, size=instance.shape
                        ).astype(np.float32)
                    )
            else:
                for top_k_ind in top_k_dists[1]:
                    comp_points.append(cross_points[top_k_ind])

            # if self.cross_point_sampling append sample
            # else append cross_points[top_k_ind]
            # then iterate over comp_points
            # for top_k_ind in top_k_dists[1]:
            for comp_point in comp_points:
                # sample new points (5)
                # cross_point = cross_points[top_k_ind]
                (
                    cross_point_mat,
                    cross_point_bias,
                    _,
                    cross_point_relu_acts,
                ) = self.calculate_inst(
                    neural_net, comp_point, dists=False, max_layer=self.max_layer
                )

                fct_dist = 0
                greater_0_flg = True

                if "relu" in self.fct_dist:
                    relu_dist = self.calc_relu_diff(relu_acts, cross_point_relu_acts)
                    fct_dist += relu_dist
                    if relu_dist <= 0:
                        greater_0_flg = False
                if "mat" in self.fct_dist:
                    mat_dist = self.calc_mat_diff(inst_mat, cross_point_mat)
                    fct_dist += mat_dist
                    if mat_dist <= 0:
                        greater_0_flg = False
                if "bias" in self.fct_dist:
                    bias_dist = self.calc_bias_diff(inst_bias, cross_point_bias)
                    fct_dist += bias_dist
                    if bias_dist <= 0:
                        greater_0_flg = False
                if greater_0_flg:
                    fct_dist = fct_dist.unsqueeze(0)
                else:
                    fct_dist = fct_dist.unsqueeze(0)
                    fct_dist = fct_dist.detach()

                fct_dists = torch.cat((fct_dists, fct_dist))
            fct_result += fct_dists.sum() / self.k
        return border_result, fct_result

    def calculate_inst(self, neural_net, instance, dists: bool = False, max_layer=None):
        if max_layer is not None:
            # calculate until and including max_layer
            # if max layer ==0, then we will get until layer 0
            neural_net = copy.deepcopy(neural_net[:max_layer+1])
        _, intermed_results = neural_net(instance)
        relus, weights, biases, relu_activations = self.get_neural_net_info(
            neural_net, intermed_results, activations=True
        )

        if dists:
            distances = torch.tensor([])
            cross_points = torch.tensor([])

        V = torch.eye(instance.shape[0])
        a = torch.zeros(instance.shape[0])

        for ind in range(0, len(weights)):
            V = torch.matmul(weights[ind], V)
            if biases[0] is not None:
                a = biases[ind] + torch.matmul(weights[ind], a)
                if dists:
                    intermed_res_ind = int(ind * 2)
                    dist = (torch.matmul(V, instance) + a) / torch.norm(V, dim=1)
                    normals = V.transpose(0, 1) / torch.norm(V, dim=1)
                    dist_normals = normals * 1.05 * dist
                    cross_point = instance - torch.transpose(dist_normals, 0, 1)
                    cross_points = torch.cat((cross_points, cross_point))
                    dist = abs(dist)
                    distances = torch.cat((distances, dist))
                a = a * relus[ind][:, 0]
            else:
                if dists:
                    distances = 0 # adjust dimension()
            V = V * relus[ind]

        if isinstance(neural_net[-1], nn.Linear):
            inst_mat = torch.matmul(neural_net[len(neural_net) - 1].weight, V)
            if neural_net[-1].bias is not None:
                inst_bias = neural_net[len(neural_net) - 1].bias + torch.matmul(
                    neural_net[len(neural_net) - 1].weight, a
                    )
            else:
                inst_bias = torch.zeros(inst_mat.shape[0])
        else:
            inst_mat = V
            if biases[0] is not None:
                inst_bias = a
            else:
                inst_bias = torch.zeros(inst_mat.shape[0])

        if dists:
            return inst_mat, inst_bias, distances, cross_points, relu_activations
        else:
            return inst_mat, inst_bias, relus, relu_activations

    def get_neural_net_info(self, neural_net, intermed_results, activations=False):
        relus = []
        weights = []
        biases = []
        relu_activations = []
        for key, value in intermed_results.items():
            if (
                int(key) < len(neural_net) - 1
                and isinstance(neural_net[int(key)], nn.Linear)
                and isinstance(neural_net[int(key) + 1], nn.ReLU)
            ):
                reluKey = str(int(key) + 1)
                relu = intermed_results[reluKey]
                relu_activations.append(torch.unsqueeze(relu, dim=1))
                relu = torch.unsqueeze(
                    torch.greater(relu, 0).type(torch.FloatTensor), dim=1
                )
                relus.append(relu)
                weights.append(neural_net[int(key)].weight)
                biases.append(neural_net[int(key)].bias)
        if activations:
            return relus, weights, biases, relu_activations
        else:
            return relus, weights, biases

    def calc_fct_diff(self, inst_mat, inst_bias, other_mat, other_bias):
        res_mat = self.calc_mat_diff(inst_mat, other_mat)
        res_bias = self.calc_bias_diff(inst_bias, other_bias)
        return (res_mat + res_bias).unsqueeze(0)

    def calc_mat_diff(self, inst_mat, other_mat):
        diff_mat = other_mat - inst_mat
        scalar_prod = torch.trace(
            torch.matmul(diff_mat, torch.transpose(diff_mat, 0, 1))
        )
        res_mat = torch.sqrt(scalar_prod)
        return res_mat

    def calc_relu_diff(self, inst_relu, other_relu):
        relu_diff = 0
        for ind in range(len(inst_relu)):
            relu_diff += ((inst_relu[ind] - other_relu[ind]) ** 2).sum()
        return relu_diff

    def calc_bias_diff(self, inst_bias, other_bias):
        diff_bias = other_bias - inst_bias
        squared = diff_bias ** 2
        summed = squared.sum()
        res_bias = torch.sqrt(summed)
        return res_bias


def reset_weights(layer):
    if isinstance(layer, nn.Linear):
        layer.reset_parameters()
