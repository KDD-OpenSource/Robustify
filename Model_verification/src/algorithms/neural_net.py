from pprint import pprint

import os
import abc
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
        batch_size: int = 20,
        lr: float = 1e-3,
        seed: int = None,
    ):
        self.name = name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = float(lr)
        self.seed = seed

    def __str__(self) -> str:
        return self.name

    @abc.abstractmethod
    def load(self):
        pass

    def save(self, path, subfolder=None):
        os.makedirs(
            os.path.join("./models/trained_models", self.name, "subfolder"),
            exist_ok=True,
        )
        torch.save(
            self.module.state_dict(),
            os.path.join("./models/trained_models", self.name, "subfolder/model.pth"),
        )

        torch.save(
            self.__dict__,
            os.path.join(
                "./models/trained_models", self.name, "subfolder/model_detailed.pth"
            ),
        )

        if subfolder:
            path = path + "/" + subfolder
            os.makedirs(path, exist_ok=True)
        torch.save(self.module.state_dict(), os.path.join(path, "model.pth"))
        torch.save(
            self.__dict__,
            os.path.join(path, "model_detailed.pth"),
        )

    def get_data_loader(self, X: pd.DataFrame):
        data_loader = DataLoader(
            num_workers=0,
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
        )
        return data_loader

    def get_lin_subfct(self, neural_net_mod, instance, max_layer=None):
        # if max_layer is added, we calcuate it until and including the layer
        forward_help_fct = smallest_k_dist_loss(1)
        neural_net = copy.deepcopy(neural_net_mod.get_neural_net())
        # calculate_inst calculates function until but excluding max_layer
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
        squared = diff**2
        summed = squared.sum()
        result = math.sqrt(summed)
        return result

    def dist_mat(self, other):
        diff = self.matrix - other.matrix
        scalar_prod = np.trace(np.matmul(diff, diff.transpose()))
        result = math.sqrt(scalar_prod)
        return result

class smallest_k_dist_loss(nn.Module):
    def __init__(
        self,
        k,
        border_dist=False,
        penal_dist=None,
        fct_dist=False,
        max_layer=None,
    ):
        self.k = k
        self.border_dist = border_dist
        self.penal_dist = penal_dist
        self.fct_dist = fct_dist
        self.max_layer = max_layer
        super(smallest_k_dist_loss, self).__init__()

    def forward(self, inputs, module):
        border_dist_sum = 0
        fct_dist_sum = 0
        for inst in inputs:
            if self.fct_dist:
                fct_dist = self.smallest_k_fctdist(module, inst)
            else:
                fct_dist = 0
            if self.border_dist:
                border_dist = self.smallest_k_borderdist(module, inst)
            else:
                border_dist = 0
            border_dist_sum += border_dist
            fct_dist_sum += fct_dist
        return border_dist_sum, fct_dist_sum

    def smallest_k_fctdist(self, neural_net_mod, instance):
        fct_result = 0
        neural_net = neural_net_mod.get_neural_net()
        fct_dists = torch.tensor([])
        inst_mat, inst_bias, relus, relu_acts, dists, cross_points = self.calculate_inst(
            neural_net, instance, dists=True, max_layer=self.max_layer
        )

        # capture errors if occuring
        if dists.isnan().sum() > 0:
            print('Return 0 as some dists are NaN')
            return 0, 0
        if dists.isinf().sum() > 0:
            print('Return 0 as some dists are Infty')
            return 0, 0

        top_k_dists = torch.topk(input=dists, k=self.k, largest=False)
        comp_points = []
        for top_k_ind in top_k_dists[1]:
            comp_points.append(cross_points[top_k_ind])

        for comp_point in comp_points:
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
                # either bias or fct are exactly the same -> gradient would be
                # zero
                fct_dist = fct_dist.unsqueeze(0)
                fct_dist = fct_dist.detach()

            fct_dists = torch.cat((fct_dists, fct_dist))
        fct_result += fct_dists.sum() / self.k

        return fct_result

    def smallest_k_borderdist(self, neural_net_mod, instance):
        border_result = 0
        neural_net = neural_net_mod.get_neural_net()
        fct_dists = torch.tensor([])
        inst_mat, inst_bias, relus, relu_acts, dists, cross_points = self.calculate_inst(
            neural_net, instance, dists=True, max_layer=self.max_layer
        )

        # capture errors if occuring
        if dists.isnan().sum() > 0:
            print('Return 0 as some dists are NaN')
            return 0, 0
        if dists.isinf().sum() > 0:
            print('Return 0 as some dists are Infty')
            return 0, 0

        top_k_dists = torch.topk(input=dists, k=self.k, largest=False)
        if self.border_dist:
            # penalize every border that is closer than self.penal_dist
            if self.penal_dist:
                top_k_maxed = torch.max(
                    torch.zeros(self.k), 1 - (top_k_dists[0] / self.penal_dist)
                )
            else:
                raise Exception('You should specify gamma (variable called penal_dist)')

            border_result += top_k_maxed.sum() / self.k
        return border_result

    def calculate_inst(self, neural_net, instance, dists: bool = False, max_layer=None):
        # calculates the linear function the current instance is on including
        # its matrix, its bias, distances to other fcts, cross points and relu
        # activations
        if max_layer is not None:
            # calculate until and including max_layer
            # if max layer ==0, then we will get until layer 0
            neural_net = copy.deepcopy(neural_net[: max_layer + 1])

        _, intermed_results = neural_net(instance)
        relu_sig, weights, biases, relu_vals = self.get_neural_net_info(
            neural_net, intermed_results, activations=True
        )

        # setup
        if biases[0] is not None:
            has_bias = True
        else:
            has_bias = False
        if dists:
            distances = torch.tensor([])
            cross_points = torch.tensor([])
        V = torch.eye(instance.shape[0])
        a = torch.zeros(instance.shape[0])

        # iterate over layers
        for ind in range(0, len(weights)):
            V = torch.matmul(weights[ind], V)
            if has_bias:
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
                a = a * relu_sig[ind][:, 0]
            else:
                raise Exception('The case of no bias has not been implemented')
            V = V * relu_sig[ind]

        # last layer calculation
        if isinstance(neural_net[-1], nn.Linear):
            inst_mat = torch.matmul(neural_net[len(neural_net) - 1].weight, V)
            if has_bias:
                inst_bias = neural_net[len(neural_net) - 1].bias + torch.matmul(
                    neural_net[len(neural_net) - 1].weight, a
                )
            else:
                inst_bias = torch.zeros(inst_mat.shape[0])
        else:
            inst_mat = V
            if has_bias:
                inst_bias = a
            else:
                inst_bias = torch.zeros(inst_mat.shape[0])

        if dists:
            return inst_mat, inst_bias, relu_sig, relu_vals, distances, cross_points
        else:
            return inst_mat, inst_bias, relu_sig, relu_vals

    def get_neural_net_info(self, neural_net, intermed_results, activations=False):
        relu_sig = []
        weights = []
        biases = []
        relu_vals = []
        for key, value in intermed_results.items():
            # check if we should extract relus, weights, biases; the current
            # key must refer to a linear layer, the next to ReLU
            if (
                int(key) < len(neural_net) - 1
                and isinstance(neural_net[int(key)], nn.Linear)
                and isinstance(neural_net[int(key) + 1], nn.ReLU)
            ):
                reluKey = str(int(key) + 1)
                relu = intermed_results[reluKey]
                relu_vals.append(torch.unsqueeze(relu, dim=1))
                relu = torch.unsqueeze(
                    torch.greater(relu, 0).type(torch.FloatTensor), dim=1
                )
                relu_sig.append(relu)
                weights.append(neural_net[int(key)].weight)
                biases.append(neural_net[int(key)].bias)
        if activations:
            return relu_sig, weights, biases, relu_vals
        else:
            return relu_sig, weights, biases

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
        squared = diff_bias**2
        summed = squared.sum()
        res_bias = torch.sqrt(summed)
        return res_bias


def reset_weights(layer):
    if isinstance(layer, nn.Linear):
        layer.reset_parameters()
