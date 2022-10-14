"""Util file for main"""
from imports import *


def exec_cfg(cfg, start_timestamp):
    cur_time_str = time.strftime("%Y-%m-%dT%H:%M:%S")
    if cfg.repeat_experiments > 1:
        base_folder = cur_time_str
    else:
        base_folder = None
    if cfg.multiple_models != None:
        base_folder = (
            cur_time_str
            + "/"
            + cfg.ctx[: cfg.ctx.find("2021")]
            + cfg.multiple_models[cfg.multiple_models.rfind("/") + 1 :]
        )

    for repetition in range(cfg.repeat_experiments):
        check_cfg_consistency(cfg)
        if cfg.repeat_experiments > 1:
            dataset, algorithm, eval_inst, evals = load_objects_cfgs(
                cfg, base_folder=base_folder, exp_run=str(repetition)
            )
        else:
            dataset, algorithm, eval_inst, evals = load_objects_cfgs(
                cfg, base_folder=base_folder
            )
        if "train" in cfg.mode:
            init_logging(eval_inst.get_run_folder())
            logger = logging.getLogger(__name__)
            algorithm.fit(dataset, eval_inst.get_run_folder(), logger)
            algorithm.save(eval_inst.get_run_folder())
            dataset.save(os.path.join(eval_inst.get_run_folder(), "dataset"))

            dataset.save(
                os.path.join(
                    "./models/trained_models/", algorithm.name, "subfolder/dataset"
                )
            )
        if "opt_param" in cfg.mode:
            best_param, best_mean, best_gaussian, best_var, result_df = param_cross_val(
                cfg
            )
            res_dict = {}
            res_dict["param"] = cfg.algorithms.validation.parameter
            res_dict["value"] = best_param
            res_dict["error_mean"] = best_mean.astype(np.float64)
            res_dict["gaussian_mean"] = best_gaussian.astype(np.float64)
            res_dict["gaussian_var"] = best_gaussian.astype(np.float64)
            eval_inst.save_json(res_dict, "parameter_opt")
            eval_inst.save_csv(result_df, "parameter_all")
        if "test" in cfg.mode:
            for evaluation in evals:
                evaluation.evaluate(dataset, algorithm)
    cfg.to_json(filename=os.path.join(eval_inst.run_folder, "cfg.json"))
    print(f"Config {cfg.ctx} is done")


## has to be reimplemented (made general; the loss function is for AEs only)
#def param_cross_val(cfg, num_times=20, test_split=0.3):
#    parallel = True
#    results = []
#    param_lower_bound = cfg.algorithms.validation.range[0]
#    param_upper_bound = cfg.algorithms.validation.range[1]
#    if parallel:
#        pool = mp.Pool(int(mp.cpu_count() / 2))
#        for param_value in np.linspace(param_lower_bound, param_upper_bound, 20):
#            results.append(
#                (
#                    param_value,
#                    pool.apply_async(
#                        calc_fixed_param,
#                        args=((param_value, cfg, num_times, test_split)),
#                    ),
#                )
#            )
#        pool.close()
#        pool.join()
#        results_df = pd.DataFrame(
#            map(lambda x: (x[0], x[1].get()[0], x[1].get()[1], x[1].get()[2]), results),
#            columns=["param", "error_mean", "gaussian_mean", "gaussian_var"],
#        )
#    else:
#        for param_value in np.linspace(param_lower_bound, param_upper_bound, 2):
#            results.append(
#                (param_value, calc_fixed_param(param_value, cfg, num_times, test_split))
#            )
#        results_df = pd.DataFrame(
#            map(lambda x: (x[0], x[1][0], x[1][1], x[1][2]), results),
#            columns=["param", "error_mean", "gaussian_mean", "gaussian_var"],
#        )
#
#    # find largest mean with smallest var (not just largest mean)
#    max_gaussian_mean = results_df.loc[results_df["gaussian_mean"].idxmax()]
#    return (
#        max_gaussian_mean["param"],
#        max_gaussian_mean["error_mean"],
#        max_gaussian_mean["gaussian_mean"],
#        max_gaussian_mean["gaussian_var"],
#        results_df,
#    )
#
#
#def calc_fixed_param(param_value, cfg, num_times, test_split):
#    parameter = cfg.algorithms.validation.parameter
#    dataset = load_dataset(cfg)
#    X = dataset.train_data()
#
#    algorithm = load_algorithm(cfg)
#    algorithm.__dict__[parameter] = param_value
#    split = ShuffleSplit(n_splits=num_times, train_size=1 - test_split)
#    # self.__dict__[parameter] = param_value
#    mean_error_list = []
#    gaussian_nb_list = []
#    for train_ds, test_ds in split.split(X):
#        algorithm.fit(X.loc[train_ds])
#        res = algorithm.predict(X.loc[test_ds])
#        mean_error = ((X.loc[test_ds] - res) ** 2).sum(axis=1).mean()
#        gnb_acc_latent = algorithm.calc_naive_bayes(
#            X.loc[train_ds],
#            X.loc[test_ds],
#            dataset.train_labels[train_ds],
#            dataset.train_labels[test_ds],
#        )
#        gaussian_nb_list.append(gnb_acc_latent)
#        mean_error_list.append(mean_error)
#        algorithm.module.apply(reset_weights)
#    param_error_mean = np.array(mean_error_list).mean()
#    gaussian_acc_mean = np.array(gaussian_nb_list).mean()
#    gaussian_acc_var = np.var(np.array(gaussian_nb_list))
#    print(f"Param_value: {param_value}")
#    print(f"Gaussian: {gaussian_acc_mean}")
#    print(f"GaussianVar: {gaussian_acc_var}")
#    return param_error_mean, gaussian_acc_mean, gaussian_acc_var


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
    if cfgs[-1].multiple_models is not None:
        subfolder_blocklist = ['remaining_models']

        model_containing_folder = cfgs[-1].multiple_models
        model_list = os.listdir(model_containing_folder)
        for blocked in subfolder_blocklist:
            try:
                model_list.remove(blocked)
            except:
                pass
        
        for _ in range(len(model_list) - 1):
            cfgs.append(copy.deepcopy(cfgs[0]))
        for cfg, model_folder in zip(cfgs, model_list):
            model_path = model_containing_folder + "/" + model_folder
            cfg.algorithm = model_path
            cfg.ctx = cfg.ctx + "_" + model_path[model_path.rfind("/") + 1 :]
            dataset_path = model_path + "/dataset"
            try:
                data_properties = list(
                    filter(lambda x: "Properties" in x, os.listdir(dataset_path))
                )[0]
                with open(os.path.join(dataset_path, data_properties)) as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        if row[0] == "name":
                            cfg.dataset = row[1]
                dataset_type = cfg.dataset
                for cfg_dataset in cfg.datasets.items():
                    if cfg_dataset[0] in dataset_type:
                        cfg_dataset[1].file_path = dataset_path
            except:
                pass
    return cfgs


def check_cfg_consistency(cfg):
    pass
    # check data dim = input dim
    # check if for each evaluation linsubfctcount is required
    # if no train, then you must have a load field


def load_objects_cfgs(cfg, base_folder, exp_run=None):
    try:
        dataset = load_dataset(cfg)
    except:
        dataset = None
    try:
        algorithm = load_algorithm(cfg)
    except:
        algorithm = None
    try:
        eval_inst, evals = load_evals(cfg, base_folder, exp_run)
    except:
        eval_inst, evals = None, None
    return dataset, algorithm, eval_inst, evals


def load_dataset(cfg):
    if cfg.dataset == "uniformClouds":
        dataset = uniformClouds(
            file_path=cfg.datasets.uniformClouds.file_path,
            subsample=cfg.datasets.subsample,
            spacedim=cfg.datasets.uniformClouds.spacedim,
            clouddim=cfg.datasets.uniformClouds.clouddim,
            num_clouds=cfg.datasets.uniformClouds.num_clouds,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.uniformClouds.num_anomalies,
            noise=cfg.datasets.uniformClouds.noise,
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
    elif cfg.dataset == "sineClasses":
        dataset = sineClasses(
            file_path=cfg.datasets.sineClasses.file_path,
            subsample=cfg.datasets.subsample,
            scale=False,
            spacedim=cfg.datasets.sineClasses.spacedim,
            num_samples=cfg.datasets.num_samples,
            num_train_anomalies=cfg.datasets.sineClasses.num_train_anomalies,
            num_test_anomalies=cfg.datasets.sineClasses.num_test_anomalies,
            num_classes=cfg.datasets.sineClasses.num_classes,
            num_testpoints=cfg.datasets.synthetic_test_samples,
        )
    elif cfg.dataset == "moons_2d":
        dataset = moons_2d(
            file_path=cfg.datasets.moons_2d.file_path,
            subsample=cfg.datasets.subsample,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.moons_2d.num_anomalies,
            noise=cfg.datasets.moons_2d.noise,
        )
    elif cfg.dataset == "parabola":
        dataset = parabola(
            file_path=cfg.datasets.parabola.file_path,
            subsample=cfg.datasets.subsample,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.parabola.num_anomalies,
            noise=cfg.datasets.parabola.noise,
            spacedim=cfg.datasets.parabola.spacedim,
        )
    elif cfg.dataset == "mnist":
        dataset = mnist(
            file_path=cfg.datasets.mnist.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.mnist.num_anomalies,
        )
    elif cfg.dataset == "cardio":
        dataset = cardio(
            file_path=cfg.datasets.cardio.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.cardio.num_anomalies,
        )
    elif cfg.dataset == "pc3":
        dataset = pc3(
            file_path=cfg.datasets.pc3.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.pc3.num_anomalies,
        )
    elif cfg.dataset == "wbc":
        dataset = wbc(
            file_path=cfg.datasets.wbc.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.wbc.num_anomalies,
        )
    elif cfg.dataset == "spambase":
        dataset = spambase(
            file_path=cfg.datasets.spambase.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.spambase.num_anomalies,
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
    elif cfg.dataset == "ozone_level_8hr":
        dataset = ozone_level_8hr(
            file_path=cfg.datasets.ozone_level_8hr.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.ozone_level_8hr.num_anomalies,
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
    elif cfg.dataset == "ionosphere":
        dataset = ionosphere(
            file_path=cfg.datasets.ionosphere.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.ionosphere.num_anomalies,
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
    elif cfg.dataset == "satimage":
        dataset = satimage(
            file_path=cfg.datasets.satimage.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.satimage.num_anomalies,
        )
    elif cfg.dataset == "creditcardFraud":
        dataset = creditcardFraud(
            file_path=cfg.datasets.creditcardFraud.file_path,
            subsample=cfg.datasets.subsample,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.creditcardFraud.num_anomalies,
        )
    elif cfg.dataset == "electricDevices":
        dataset = electricDevices(
            file_path=cfg.datasets.electricDevices.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "italyPowerDemand":
        dataset = italyPowerDemand(
            file_path=cfg.datasets.italyPowerDemand.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "ecg5000":
        dataset = ecg5000(
            file_path=cfg.datasets.ecg5000.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "chinatown":
        dataset = chinatown(
            file_path=cfg.datasets.chinatown.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif "crop" in cfg.dataset:
        dataset = crop(
            name=cfg.dataset,
            class1=cfg.datasets.crop.class1,
            class2=cfg.datasets.crop.class2,
            file_path=cfg.datasets.crop.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "moteStrain":
        dataset = moteStrain(
            file_path=cfg.datasets.moteStrain.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "wafer":
        dataset = wafer(
            file_path=cfg.datasets.wafer.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "insectWbs":
        dataset = insectWbs(
            file_path=cfg.datasets.insectWbs.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "chlorineConcentration":
        dataset = chlorineConcentration(
            file_path=cfg.datasets.chlorineConcentration.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "melbournePedestrian":
        dataset = melbournePedestrian(
            file_path=cfg.datasets.melbournePedestrian.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "proximalPhalanxOutlineCorrect":
        dataset = proximalPhalanxOutlineCorrect(
            file_path=cfg.datasets.proximalPhalanxOutlineCorrect.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "syntheticControl":
        dataset = syntheticControl(
            file_path=cfg.datasets.syntheticControl.file_path,
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
    elif cfg.dataset == "predictiveMaintenance":
        dataset = predictiveMaintenance(
            file_path=cfg.datasets.predictiveMaintenance.file_path,
            subsample=cfg.datasets.subsample,
            num_samples=cfg.datasets.predictiveMaintenance.num_samples,
            num_anomalies=cfg.datasets.predictiveMaintenance.num_anomalies,
            window_size=cfg.datasets.predictiveMaintenance.window_size,
        )
    else:
        raise Exception("Could not create dataset.")
    return dataset


def load_algorithm(cfg):
    if "/" in cfg.algorithm:
        if "autoencoder" in cfg.algorithm:
            algorithm = autoencoder(
                topology=[2, 1, 2],
                fct_dist=cfg.algorithms.autoencoder.fct_dist,
                border_dist=cfg.algorithms.autoencoder.border_dist,
                num_border_points=cfg.algorithms.autoencoder.num_border_points,
            )
            algorithm.load(cfg.algorithm)
        if "deepOcc" in cfg.algorithm:
            algorithm = deepOcc(
                topology=[3,2,1],
            )
            algorithm.load(cfg.algorithm)
        if "fcnnClassifier" in cfg.algorithm:
            algorithm = fcnnClassifier(
                topology=[3,2,1],
            )
            algorithm.load(cfg.algorithm)
    else:
        if cfg.algorithm == "autoencoder":
            algorithm = autoencoder(
                train_robust_ae=cfg.algorithms.autoencoder.train_robust_ae,
                denoising=cfg.algorithms.autoencoder.denoising,
                fct_dist=cfg.algorithms.autoencoder.fct_dist,
                fct_dist_layer=cfg.algorithms.autoencoder.fct_dist_layer,
                border_dist=cfg.algorithms.autoencoder.border_dist,
                cross_point_sampling=cfg.algorithms.autoencoder.cross_point_sampling,
                lambda_border=cfg.algorithms.autoencoder.lambda_border,
                lambda_fct=cfg.algorithms.autoencoder.lambda_fct,
                topology=cfg.algorithms.autoencoder.topology,
                num_epochs=cfg.algorithms.num_epochs,
                dynamic_epochs=cfg.algorithms.dynamic_epochs,
                lr=cfg.algorithms.lr,
                collect_subfcts=cfg.algorithms.collect_subfcts,
                bias_shift=cfg.algorithms.autoencoder.bias_shift,
                push_factor=cfg.algorithms.push_factor,
                dropout=cfg.algorithms.autoencoder.dropout,
                L2Reg=cfg.algorithms.autoencoder.L2Reg,
                num_border_points=cfg.algorithms.autoencoder.num_border_points,
                save_interm_models=cfg.algorithms.save_interm_models,
            )
        elif cfg.algorithm == "deepOcc":
            algorithm = deepOcc(
                topology = cfg.algorithms.deepOcc.topology,
                L2Reg=cfg.algorithms.deepOcc.L2Reg,
                anom_quantile=cfg.algorithms.deepOcc.anom_quantile,
                dropout =cfg.algorithms.deepOcc.dropout ,
                num_epochs=cfg.algorithms.num_epochs,
                lr=cfg.algorithms.lr,
                save_interm_models=cfg.algorithms.save_interm_models,
            )
        elif cfg.algorithm == "fcnnClassifier":
            algorithm = fcnnClassifier(
                topology = cfg.algorithms.fcnnClassifier.topology,
                L2Reg=cfg.algorithms.fcnnClassifier.L2Reg,
                dropout =cfg.algorithms.fcnnClassifier.dropout ,
                num_epochs=cfg.algorithms.num_epochs,
                lr=cfg.algorithms.lr,
                save_interm_models=cfg.algorithms.save_interm_models,
            )
        else:
            raise Exception("Could not load algorithm")
    return algorithm


def load_evals(cfg, base_folder=None, exp_run=None):
    eval_inst = evaluation(base_folder)
    eval_inst.make_run_folder(ctx=cfg.ctx, exp_run=exp_run)
    evals = []
    if "parallelQualplots" in cfg.evaluations:
        evals.append(parallelQualplots(eval_inst=eval_inst))
    if "downstream_kmeans" in cfg.evaluations:
        evals.append(downstream_kmeans(eval_inst=eval_inst))
    if "downstream_naiveBayes" in cfg.evaluations:
        evals.append(downstream_naiveBayes(eval_inst=eval_inst))
    if "downstream_knn" in cfg.evaluations:
        evals.append(downstream_knn(eval_inst=eval_inst))
    if "downstream_rf" in cfg.evaluations:
        evals.append(downstream_rf(eval_inst=eval_inst))
    if "linSubfctBarplots" in cfg.evaluations:
        evals.append(linSubfctBarplots(eval_inst=eval_inst))
    if "linSub_unifPoints" in cfg.evaluations:
        evals.append(linSub_unifPoints(eval_inst=eval_inst))
    if "subfunc_distmat" in cfg.evaluations:
        evals.append(subfunc_distmat(eval_inst=eval_inst))
    if "linsubfct_parallelPlots" in cfg.evaluations:
        evals.append(linsubfct_parallelPlots(eval_inst=eval_inst))
    if "linsubfct_distr" in cfg.evaluations:
        evals.append(linsubfct_distr(eval_inst=eval_inst))
    if "reconstr_dataset" in cfg.evaluations:
        evals.append(reconstr_dataset(eval_inst=eval_inst))
    if "label_info" in cfg.evaluations:
        evals.append(label_info(eval_inst=eval_inst))
    if "mse_test" in cfg.evaluations:
        evals.append(mse_test(eval_inst=eval_inst))
    if "anomaly_score_hist" in cfg.evaluations:
        evals.append(anomaly_score_hist(eval_inst=eval_inst))
    if "anomaly_score_samples" in cfg.evaluations:
        evals.append(anomaly_score_samples(eval_inst=eval_inst))
    if "anomaly_quantile_radius" in cfg.evaluations:
       evals.append(anomaly_quantile_radius(eval_inst=eval_inst))
    if "ad_box_creator" in cfg.evaluations:
       evals.append(ad_box_creator(eval_inst=eval_inst))
    if "singularValuePlots" in cfg.evaluations:
        evals.append(singularValuePlots(eval_inst=eval_inst))
    if "closest_linsubfct_plot" in cfg.evaluations:
        evals.append(closest_linsubfct_plot(eval_inst=eval_inst))
    if "boundary_2d_plot" in cfg.evaluations:
        evals.append(boundary_2d_plot(eval_inst=eval_inst))
    if "inst_area_2d_plot" in cfg.evaluations:
        evals.append(inst_area_2d_plot(eval_inst=eval_inst))
    if "border_dist_2d" in cfg.evaluations:
        evals.append(border_dist_2d(eval_inst=eval_inst))
    if "border_dist_sort_plot" in cfg.evaluations:
        evals.append(border_dist_sort_plot(eval_inst=eval_inst))
    if "error_border_dist_plot" in cfg.evaluations:
        evals.append(error_border_dist_plot(eval_inst=eval_inst))
    if "error_label_mean_dist_plot" in cfg.evaluations:
        evals.append(error_label_mean_dist_plot(eval_inst=eval_inst))
    if "error_border_dist_plot_colored" in cfg.evaluations:
        evals.append(error_border_dist_plot_colored(eval_inst=eval_inst))
    if "error_border_dist_plot_anomalies" in cfg.evaluations:
        evals.append(error_border_dist_plot_anomalies(eval_inst=eval_inst))
    if "plot_mnist_samples" in cfg.evaluations:
        evals.append(plot_mnist_samples(eval_inst=eval_inst))
    if "image_lrp" in cfg.evaluations:
        evals.append(image_lrp(eval_inst=eval_inst))
    if "image_feature_imp" in cfg.evaluations:
        evals.append(image_feature_imp(eval_inst=eval_inst))
    if "qual_by_border_dist_plot" in cfg.evaluations:
        evals.append(qual_by_border_dist_plot(eval_inst=eval_inst))
    if "fct_change_by_border_dist_qual" in cfg.evaluations:
        evals.append(fct_change_by_border_dist_qual(eval_inst=eval_inst))
    if "marabou_classes" in cfg.evaluations:
        evals.append(marabou_classes(eval_inst=eval_inst))
    if "marabou_robust" in cfg.evaluations:
        evals.append(marabou_robust(eval_inst=eval_inst))
    if "marabou_anomalous" in cfg.evaluations:
        evals.append(marabou_anomalous(eval_inst=eval_inst))
    if "marabou_largest_error" in cfg.evaluations:
        evals.append(marabou_largest_error(eval_inst=eval_inst))
    if "marabou_superv_robust" in cfg.evaluations:
        evals.append(marabou_superv_robust(eval_inst=eval_inst))
    if "deepoc_adv_marabou_borderpoint" in cfg.evaluations:
        evals.append(deepoc_adv_marabou_borderpoint(eval_inst=eval_inst))
    if "deepoc_adv_marabou_borderplane" in cfg.evaluations:
        evals.append(deepoc_adv_marabou_borderplane(eval_inst=eval_inst))
    if "deepoc_adv_derivative" in cfg.evaluations:
        evals.append(deepoc_adv_derivative(eval_inst=eval_inst))
    if "bias_feature_imp" in cfg.evaluations:
        evals.append(bias_feature_imp(eval_inst=eval_inst))
    if "interpolation_func_diffs_pairs" in cfg.evaluations:
        evals.append(interpolation_func_diffs_pairs(eval_inst=eval_inst))
    if "interpolation_error_plot" in cfg.evaluations:
        evals.append(interpolation_error_plot(eval_inst=eval_inst))
    if "mnist_interpolation_func_diffs_pairs" in cfg.evaluations:
        evals.append(mnist_interpolation_func_diffs_pairs(eval_inst=eval_inst))
    if "tsne_latent" in cfg.evaluations:
        evals.append(tsne_latent(eval_inst=eval_inst))
    if "orig_latent_dist_ratios" in cfg.evaluations:
        evals.append(orig_latent_dist_ratios(eval_inst=eval_inst))
    if "calc_linfct_volume" in cfg.evaluations:
        evals.append(calc_linfct_volume(eval_inst=eval_inst))
    if "superv_accuracy" in cfg.evaluations:
        evals.append(superv_accuracy(eval_inst=eval_inst))
    if "num_anomalies_sanity" in cfg.evaluations:
        evals.append(num_anomalies_sanity(eval_inst=eval_inst))
    if "deepOcc_2d_implot" in cfg.evaluations:
        evals.append(deepOcc_2d_implot(eval_inst=eval_inst))
    if "latent_avg_cos_sim" in cfg.evaluations:
        evals.append(latent_avg_cos_sim(eval_inst=eval_inst))
    if "nn_eval_avg_layer_weights" in cfg.evaluations:
        evals.append(nn_eval_avg_layer_weights(eval_inst=eval_inst))
    if "plot_closest_border_dist" in cfg.evaluations:
        evals.append(plot_closest_border_dist(eval_inst=eval_inst))
    if "avg_min_fctborder_dist" in cfg.evaluations:
        evals.append(avg_min_fctborder_dist(eval_inst=eval_inst))
    if "code_test_on_trained_model" in cfg.evaluations:
        evals.append(code_test_on_trained_model(eval_inst=eval_inst))
    if "num_subfcts_per_class" in cfg.evaluations:
        evals.append(num_subfcts_per_class(eval_inst=eval_inst))
    if "parallel_plot_same_subfcts" in cfg.evaluations:
        evals.append(parallel_plot_same_subfcts(eval_inst=eval_inst))
    if "plot_entire_train_set" in cfg.evaluations:
        evals.append(plot_entire_train_set(eval_inst=eval_inst))
    if "save_nn_in_format" in cfg.evaluations:
        evals.append(save_nn_in_format(eval_inst=eval_inst))
    if "marabou_ens_largErr" in cfg.evaluations:
        evals.append(marabou_ens_largErr(eval_inst=eval_inst, cfg=cfg))
    if "marabou_ens_normal_rob" in cfg.evaluations:
        evals.append(marabou_ens_normal_rob(eval_inst=eval_inst, cfg=cfg))
    if "marabou_svdd_normal_rob" in cfg.evaluations:
        evals.append(marabou_svdd_normal_rob(eval_inst=eval_inst, cfg=cfg))
    if "marabou_ens_normal_rob_ae" in cfg.evaluations:
        evals.append(marabou_ens_normal_rob_ae(eval_inst=eval_inst, cfg=cfg))
    if "marabou_ens_normal_rob_submodels" in cfg.evaluations:
        evals.append(marabou_ens_normal_rob_submodels(eval_inst=eval_inst, cfg=cfg))
    if "marabou_ens_anom_rob" in cfg.evaluations:
        evals.append(marabou_ens_anom_rob(eval_inst=eval_inst, cfg=cfg))
    if "lirpa_ens_normal_rob" in cfg.evaluations:
        evals.append(lirpa_ens_normal_rob(eval_inst=eval_inst, cfg=cfg))
    return eval_inst, evals


if __name__ == "__main__":
    main()
