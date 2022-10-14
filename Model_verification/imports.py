import os
import csv
import copy
import time
import sys

sys.path.append('/home/NNLinSubfct/Code/eran_files/ELINA/python_interface')
sys.path.append('/home/NNLinSubfct/Code/eran_files/deepg/code')
#sys.path.append('/home/NNLinSubfct/Code/eran_files/deepg/ERAN/tf_verify')
sys.path.append('/home/NNLinSubfct/Code/eran_files/tf_verify')
sys.path.append('/home/NNLinSubfct/Code/NNet')


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
#import onnx
#from onnx2pytorch import ConvertModel
#from pytorch2keras.converter import pytorch_to_keras
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from box import Box

from src.datasets.gaussianClouds import gaussianClouds
from src.datasets.uniformClouds import uniformClouds
from src.datasets.sineNoise import sineNoise
from src.datasets.sineClasses import sineClasses
from src.datasets.moons_2d import moons_2d
from src.datasets.parabola import parabola
from src.datasets.mnist import mnist
from src.datasets.cardio import cardio
from src.datasets.pc3 import pc3
from src.datasets.wbc import wbc
from src.datasets.spambase import spambase
from src.datasets.steel_plates_fault import steel_plates_fault
from src.datasets.qsar_biodeg import qsar_biodeg
from src.datasets.page_blocks import page_blocks
from src.datasets.ozone_level_8hr import ozone_level_8hr
from src.datasets.gas_drift import gas_drift
from src.datasets.har import har
from src.datasets.ionosphere import ionosphere
from src.datasets.satellite import satellite
from src.datasets.segment import segment
from src.datasets.satimage import satimage
from src.datasets.creditcardFraud import creditcardFraud
from src.datasets.predictiveMaintenance import predictiveMaintenance
from src.datasets.ecg5000 import ecg5000
from src.datasets.electricDevices import electricDevices
from src.datasets.italyPowerDemand import italyPowerDemand
from src.datasets.proximalPhalanxOutlineCorrect import proximalPhalanxOutlineCorrect
from src.datasets.sonyAIBORobotSurface1 import sonyAIBORobotSurface1
from src.datasets.sonyAIBORobotSurface2 import sonyAIBORobotSurface2
from src.datasets.syntheticControl import syntheticControl
from src.datasets.twoLeadEcg import twoLeadEcg
from src.datasets.chinatown import chinatown
from src.datasets.crop import crop
from src.datasets.moteStrain import moteStrain
from src.datasets.wafer import wafer
from src.datasets.insectWbs import insectWbs
from src.datasets.chlorineConcentration import chlorineConcentration
from src.datasets.melbournePedestrian import melbournePedestrian
from src.algorithms.autoencoder import autoencoder
from src.algorithms.deepOcc import deepOcc
from src.algorithms.fcnnClassifier import fcnnClassifier
from src.algorithms.neural_net import reset_weights
from src.evaluations.parallelQualplots import parallelQualplots
from src.evaluations.downstream_kmeans import downstream_kmeans
from src.evaluations.downstream_naiveBayes import downstream_naiveBayes
from src.evaluations.downstream_knn import downstream_knn
from src.evaluations.downstream_rf import downstream_rf
from src.evaluations.tsne_latent import tsne_latent
from src.evaluations.linSubfctBarplots import linSubfctBarplots
from src.evaluations.linSub_unifPoints import linSub_unifPoints
from src.evaluations.subfunc_distmat import subfunc_distmat
from src.evaluations.linsubfct_parallelPlots import linsubfct_parallelPlots
from src.evaluations.linsubfct_distr import linsubfct_distr
from src.evaluations.calc_linfct_volume import calc_linfct_volume
from src.evaluations.reconstr_dataset import reconstr_dataset
from src.evaluations.label_info import label_info
from src.evaluations.mse_test import mse_test
from src.evaluations.anomaly_score_hist import anomaly_score_hist
from src.evaluations.anomaly_score_samples import anomaly_score_samples
from src.evaluations.anomaly_quantile_radius import anomaly_quantile_radius
from src.evaluations.ad_box_creator import ad_box_creator
from src.evaluations.singularValuePlots import singularValuePlots
from src.evaluations.closest_linsubfct_plot import closest_linsubfct_plot
from src.evaluations.boundary_2d_plot import boundary_2d_plot
from src.evaluations.inst_area_2d_plot import inst_area_2d_plot
from src.evaluations.border_dist_2d import border_dist_2d
from src.evaluations.border_dist_sort_plot import border_dist_sort_plot
from src.evaluations.error_border_dist_plot import error_border_dist_plot
from src.evaluations.superv_accuracy import superv_accuracy
from src.evaluations.orig_latent_dist_ratios import orig_latent_dist_ratios
from src.evaluations.error_label_mean_dist_plot import error_label_mean_dist_plot
from src.evaluations.error_border_dist_plot_colored import (
    error_border_dist_plot_colored,
)
from src.evaluations.error_border_dist_plot_anomalies import (
    error_border_dist_plot_anomalies,
)
from src.evaluations.plot_mnist_samples import plot_mnist_samples
from src.evaluations.image_lrp import image_lrp
from src.evaluations.image_feature_imp import image_feature_imp
from src.evaluations.qual_by_border_dist_plot import qual_by_border_dist_plot
from src.evaluations.marabou_classes import marabou_classes
from src.evaluations.marabou_robust import marabou_robust
from src.evaluations.marabou_anomalous import marabou_anomalous
from src.evaluations.marabou_largest_error import marabou_largest_error
from src.evaluations.marabou_superv_robust import marabou_superv_robust
from src.evaluations.deepoc_adv_marabou_borderpoint import (
    deepoc_adv_marabou_borderpoint)
from src.evaluations.deepoc_adv_marabou_borderplane import (
    deepoc_adv_marabou_borderplane)
from src.evaluations.deepoc_adv_derivative import (
    deepoc_adv_derivative)
from src.evaluations.fct_change_by_border_dist_qual import (
    fct_change_by_border_dist_qual,
)
from src.evaluations.bias_feature_imp import bias_feature_imp
from src.evaluations.interpolation_func_diffs_pairs import (
    interpolation_func_diffs_pairs,
)
from src.evaluations.interpolation_error_plot import interpolation_error_plot
from src.evaluations.mnist_interpolation_func_diffs_pairs import (
    mnist_interpolation_func_diffs_pairs,
)
from src.evaluations.num_anomalies_sanity import num_anomalies_sanity
from src.evaluations.deepOcc_2d_implot import deepOcc_2d_implot
from src.evaluations.latent_avg_cos_sim import latent_avg_cos_sim
from src.evaluations.nn_eval_avg_layer_weights import nn_eval_avg_layer_weights
from src.evaluations.plot_closest_border_dist import plot_closest_border_dist
from src.evaluations.avg_min_fctborder_dist import avg_min_fctborder_dist
from src.evaluations.code_test_on_trained_model import code_test_on_trained_model
from src.evaluations.num_subfcts_per_class import num_subfcts_per_class
from src.evaluations.parallel_plot_same_subfcts import parallel_plot_same_subfcts
from src.evaluations.plot_entire_train_set import plot_entire_train_set
from src.evaluations.save_nn_in_format import save_nn_in_format
from src.evaluations.marabou_ens_largErr import marabou_ens_largErr
from src.evaluations.marabou_ens_normal_rob import marabou_ens_normal_rob
from src.evaluations.marabou_svdd_normal_rob import marabou_svdd_normal_rob
from src.evaluations.marabou_ens_normal_rob_ae import marabou_ens_normal_rob_ae
from src.evaluations.marabou_ens_normal_rob_submodels import (
    marabou_ens_normal_rob_submodels)
from src.evaluations.marabou_ens_anom_rob import marabou_ens_anom_rob
from src.evaluations.lirpa_ens_normal_rob import lirpa_ens_normal_rob



from src.evaluations.evaluation import evaluation
from src.utils.config import config, init_logging

