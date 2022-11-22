"""Util file for main"""
from src.utils.imports import *
import sys
import logging
import copy
import os
import time
import json


def exec_cfg(cfg, start_timestamp):
    cur_time_str = time.strftime("%Y-%m-%dT%H:%M:%S")
    base_folder = None
    if cfg.repeat_experiment > 1:
        base_folder = cur_time_str
    if cfg.test_models != None:
        base_folder = (
            cur_time_str + "/" + cfg.test_models[cfg.test_models.rfind("/") + 1 :]
        )

    for repetition in range(cfg.repeat_experiment):
        if cfg.repeat_experiment > 1:
            run_inst, dataset, algorithm, evals = load_objects_cfgs(
                cfg, base_folder=base_folder, run_number=str(repetition)
            )
        else:
            run_inst, dataset, algorithm, evals = load_objects_cfgs(
                cfg, base_folder=base_folder
            )

        if "train" in cfg.mode:
            init_logging(run_inst.get_run_folder())
            logger = logging.getLogger(__name__)
            algorithm.fit(dataset.train_data(), run_inst.get_run_folder(), logger)
            algorithm.save(run_inst.get_run_folder())
            dataset.save(os.path.join(run_inst.get_run_folder(), "dataset"))

            dataset.save(
                os.path.join(
                    "./models/trained_models/", algorithm.name, "subfolder/dataset"
                )
            )
        if "test" in cfg.mode:
            for evaluation in evals:
                evaluation.evaluate(dataset, algorithm, run_inst)
        print(f'Repetition {repetition} is done.')
    cfg.to_json(filename=os.path.join(run_inst.run_folder, "cfg.json"))
    print(f"Config {cfg.ctx} is done")


def load_cfgs():
    cfgs = []
    for arg in sys.argv[1:]:
        if arg[-1] == "/":
            for cfg in os.listdir(arg):
                if cfg[-4:] == "yaml":
                    cfgs.extend(read_cfg(arg + cfg))
        elif arg[-4:] == "yaml":
            cfgs.extend(read_cfg(arg))
        else:
            raise Exception("could not read argument")
    return cfgs


def read_cfg(cfg):
    cfgs = []
    cfgs.append(Box(config(os.path.join(os.getcwd(), cfg)).config_dict))
    if cfgs[-1].test_models is not None:
        # allows to load multiple models for testing
        model_containing_folder = cfgs[-1].test_models
        model_list = os.listdir(model_containing_folder)
        for _ in range(len(model_list) - 1):
            cfgs.append(copy.deepcopy(cfgs[0]))
        for cfg, model_folder in zip(cfgs, model_list):
            model_path = model_containing_folder + "/" + model_folder
            cfg.algorithm = model_path
            cfg.ctx = cfg.ctx + "_" + model_path[model_path.rfind("/") + 1 :]
            dataset_path = model_path + "/dataset"
            try:
                data_properties = list(
                    filter(lambda x: "properties" in x, os.listdir(dataset_path))
                )[0]
                with open(os.path.join(dataset_path, data_properties)) as file:
                    ds_dict = json.load(file)
                    cfg.dataset = ds_dict["name"]
                dataset_type = cfg.dataset
                cfg.datasets = {dataset_type: ds_dict}
                for conf_key, conf_value in cfg.datasets[dataset_type].items():
                    cfg.datasets[dataset_type][conf_key] = ds_dict[conf_key]
                cfg.datasets[dataset_type].file_path = dataset_path
            except:
                pass
    return cfgs


def load_objects_cfgs(cfg, base_folder, run_number=None):
    run_inst = exp_run(base_folder)
    run_inst.make_run_folder(ctx=cfg.ctx, run_number=run_number)
    try:
        dataset = load_dataset(cfg)
    except:
        dataset = None
    try:
        algorithm = load_algorithm(cfg)
    except:
        algorithm = None
    try:
        evals = load_evals(cfg, base_folder, run_number)
    except:
        evals = None
    return run_inst, dataset, algorithm, evals


def load_dataset(cfg):
    if cfg.test_models is not None:
        mod = __import__('src.datasets.'+cfg.dataset)
        mod = getattr(mod, 'datasets')
        data_class = getattr(mod, cfg.dataset)
        dataset = data_class(file_path = cfg.datasets[cfg.dataset].file_path)
    elif cfg.dataset == "sineNoise":
        dataset = sineNoise(
            file_path=cfg.datasets.sineNoise.file_path,
            subsample=cfg.datasets.subsample,
            scale=False,
            spacedim=cfg.datasets.sineNoise.spacedim,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.sineNoise.num_anomalies,
            num_testpoints=cfg.datasets.synthetic_test_samples,
        )
    elif cfg.dataset == "gaussianClouds":
        dataset = gaussianClouds(
            file_path=cfg.datasets.gaussianClouds.file_path,
            subsample=cfg.datasets.subsample,
            spacedim=cfg.datasets.gaussianClouds.spacedim,
            clouddim=cfg.datasets.gaussianClouds.clouddim,
            num_clouds=cfg.datasets.gaussianClouds.num_clouds,
            num_samples=cfg.datasets.num_samples,
            scale=cfg.datasets.scale,
            num_anomalies=cfg.datasets.gaussianClouds.num_anomalies,
            num_testpoints=cfg.datasets.synthetic_test_samples,
        )
    elif cfg.dataset == "mnist":
        dataset = mnist(
            file_path=cfg.datasets.mnist.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.mnist.num_anomalies,
            normal_class=cfg.datasets.mnist.normal_class,
        )
    elif cfg.dataset == "wbc":
        dataset = wbc(
            file_path=cfg.datasets.wbc.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.wbc.num_anomalies,
        )
    elif cfg.dataset == "steel_plates_fault":
        dataset = steel_plates_fault(
            file_path=cfg.datasets.steel_plates_fault.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.steel_plates_fault.num_anomalies,
        )
    elif cfg.dataset == "qsar_biodeg":
        dataset = qsar_biodeg(
            file_path=cfg.datasets.qsar_biodeg.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.qsar_biodeg.num_anomalies,
        )
    elif cfg.dataset == "page_blocks":
        dataset = page_blocks(
            file_path=cfg.datasets.page_blocks.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.page_blocks.num_anomalies,
        )
    elif cfg.dataset == "gas_drift":
        dataset = gas_drift(
            file_path=cfg.datasets.gas_drift.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.gas_drift.num_anomalies,
        )
    elif cfg.dataset == "har":
        dataset = har(
            file_path=cfg.datasets.har.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.har.num_anomalies,
        )
    elif cfg.dataset == "satellite":
        dataset = satellite(
            file_path=cfg.datasets.satellite.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.satellite.num_anomalies,
        )
    elif cfg.dataset == "segment":
        dataset = segment(
            file_path=cfg.datasets.segment.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.segment.num_anomalies,
        )
    elif cfg.dataset == "moteStrain":
        dataset = moteStrain(
            file_path=cfg.datasets.moteStrain.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "proximalPhalanxOutlineCorrect":
        dataset = proximalPhalanxOutlineCorrect(
            file_path=cfg.datasets.proximalPhalanxOutlineCorrect.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "phalangesOutlinesCorrect":
        dataset = phalangesOutlinesCorrect(
            file_path=cfg.datasets.phalangesOutlinesCorrect.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "twoLeadEcg":
        dataset = twoLeadEcg(
            file_path=cfg.datasets.twoLeadEcg.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "sonyAIBORobotSurface1":
        dataset = sonyAIBORobotSurface1(
            file_path=cfg.datasets.sonyAIBORobotSurface1.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "sonyAIBORobotSurface2":
        dataset = sonyAIBORobotSurface2(
            file_path=cfg.datasets.sonyAIBORobotSurface2.file_path,
            subsample=cfg.datasets.subsample,
        )
    else:
        raise Exception("Could not create dataset.")
    return dataset


def load_algorithm(cfg):
    if "/" in cfg.algorithm:
        # means that it is a path to an already trained one
        if "autoencoder" in cfg.algorithm:
            algorithm = autoencoder()
            algorithm.load(cfg.algorithm)
    else:
        if cfg.algorithm == "autoencoder":
            # create autoencoder according to specifications in config file
            algorithm = autoencoder(
                topology=cfg.algorithms.autoencoder.topology,
                bias=cfg.algorithms.autoencoder.bias,
                fct_dist=cfg.algorithms.autoencoder.fct_dist,
                lambda_fct=cfg.algorithms.autoencoder.lambda_fct,
                robust_ae=cfg.algorithms.autoencoder.robust_ae,
                denoising=cfg.algorithms.autoencoder.denoising,
                border_dist=cfg.algorithms.autoencoder.border_dist,
                lambda_border=cfg.algorithms.autoencoder.lambda_border,
                penal_dist=cfg.algorithms.autoencoder.penal_dist,
                num_border_points=cfg.algorithms.autoencoder.num_border_points,
                dropout=cfg.algorithms.autoencoder.dropout,
                L2Reg=cfg.algorithms.autoencoder.L2Reg,
                num_epochs=cfg.algorithms.num_epochs,
                lr=cfg.algorithms.lr,
            )
        else:
            raise Exception("Could not load algorithm")
    return algorithm


def load_evals(cfg, base_folder=None, run_number=None):
    evals = []
    if "marabou_ens_normal_rob" in cfg.evaluations:
        evals.append(marabou_ens_normal_rob(cfg = cfg))
    if "marabou_svdd_normal_rob" in cfg.evaluations:
        evals.append(marabou_svdd_normal_rob(cfg = cfg))
    if "marabou_ens_normal_rob_ae" in cfg.evaluations:
        evals.append(marabou_ens_normal_rob_ae(cfg = cfg))
    if "marabou_robust" in cfg.evaluations:
        evals.append(marabou_robust())
    if "marabou_largest_error" in cfg.evaluations:
        evals.append(marabou_largest_error())
    if "inst_area_2d_plot" in cfg.evaluations:
        evals.append(inst_area_2d_plot())
    if "downstream_naiveBayes" in cfg.evaluations:
        evals.append(downstream_naiveBayes())
    if "downstream_knn" in cfg.evaluations:
        evals.append(downstream_knn())
    if "downstream_rf" in cfg.evaluations:
        evals.append(downstream_rf())
    return evals


if __name__ == "__main__":
    main()
