import numpy as np
import time
import sys
import os
import importlib
from progressbar import ProgressBar

import Utility as util
import Evaluation
from Plotting import Plotting
from Variables import Variables
from Arguments import ArgumentParser, ArgumentBuilder
from Container import Container
from Data.Data import Data


class Pipeline:

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    def __init__(self, var=None):
        if var is None: var = Variables()
        self.init_variables(var)

    def execute(self):
        train_var, eval_var = self.train_var, self.eval_var
        # Initialize and synchronize processes
        self.init_processes()
        # Initialize and unpack the datasets
        self.init_datasets()
        # Pre-process the datasets
        self.preprocess_datasets()
        # Debug information
        self.log_debug_info()
        # Initialize the model, and its checkpoint + evaluation directories
        self.model = self.init_model(self.exec_var.model, self.datasets, self.var)
        self.init_paths()
        self.log_model_info()
        self.log_settings()
        ##########################
        ### MODEL OPTIMIZATION ###
        ##########################
        if var.execution.train:
            self.optimize_model()
            self.log_optimization_info()
        ########################
        ### MODEL EVALUATION ###
        ########################
        if var.execution.evaluate:
            self.load_model_for_evaluation()
            self.get_model_predictions()
            self.postprocess_predictions()
            self.quantify_prediction_accuracy()
            self.log_prediction_info()

    def init_variables(self, var):
        # Unpack variables
        exec_var = var.execution
        hyp_var = var.models.get(exec_var.model).hyperparameters
        map_var = var.mapping
        proc_var = var.processing
        train_var = var.training
        eval_var = var.evaluating
        chkpt_var = var.checkpointing
        dist_var = var.distribution
        plt_var = var.plotting
        dbg_var = var.debug
        plt = Plotting()
        proc_rank = dist_var.process_rank
        root_proc_rank = dist_var.root_process_rank
        n_procs = dist_var.n_processes
        # Make edits
        train_var.copy(var.models.get(exec_var.model).training)
        # Set variables
        self.var, self.exec_var, self.hyp_var = var, exec_var, hyp_var
        self.map_var, self.proc_var, self.train_var = map_var, proc_var, train_var
        self.eval_var, self.chkpt_var, self.dist_var = eval_var, chkpt_var, dist_var
        self.plt_var, self.dbg_var = plt_var, dbg_var
        self.plt = plt

    def init_processes(self):
        pass

    def init_datasets(self):
        self.datasets = Data(self.var)
        self.train_dataset = self.datasets.get("dataset", "train")
        self.valid_dataset = self.datasets.get("dataset", "valid")
        self.test_dataset = self.datasets.get("dataset", "test")

    def preprocess_datasets(self):
        # Already completed in the component classes of class Data
        pass

    def log_debug_info(self):
        var, dbg_var, datasets = self.var, self.dbg_var, self.datasets
        train_dataset, valid_dataset, test_dataset = self.train_dataset, self.valid_dataset, self.test_dataset
        if dbg_var.print_vars:
            n = 46
            msg = " " * n + "Variables" + " " * n
            print(util.make_msg_block(msg))
            print(var)
            print(util.make_msg_block(msg))
        if not dbg_var.print_dataset is None:
            n = 48
            msg = " " * n + "Data" + " " * n
            print(util.make_msg_block(msg))
            print(datasets.get("dataset", dbg_var.print_dataset))
            print(util.make_msg_block(msg))
        if dbg_var.dataset_memory:
            print("Training Dataset Memory Usage =", util.get_memory_of(train_dataset))
            if proc_rank == root_proc_rank:
                print("Validation Dataset Memory Usage =", util.get_memory_of(valid_dataset))
                print("Testing Dataset Memory Usage =", util.get_memory_of(test_dataset))

    def init_model(self, model_name, datasets, var):
        model_module = importlib.import_module("Models.%s" % (model_name))
        model = model_module.init(datasets.get("dataset", "train"), var)
        model.debug = self.dbg_var.view_model_forward
        return model

    def init_paths(self):
        var, chkpt_var, eval_var = self.var, self.chkpt_var, self.eval_var
        datasets, model = self.datasets, self.model
        chkpt_dir = os.sep.join([chkpt_var.checkpoint_dir, model.name(), get_id(model, datasets, var)])
        eval_dir = os.sep.join([eval_var.evaluation_dir, model.name(), get_id(model, datasets, var)])
        os.makedirs(chkpt_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        print(
            util.make_msg_block("Checkpoint Directory: %s" % (chkpt_dir.replace(os.getcwd(), ".")), "+")
        )
        print(
            util.make_msg_block("Evaluation Directory: %s" % (eval_dir.replace(os.getcwd(), ".")), "+")
        )
        self.chkpt_dir, self.eval_dir = chkpt_dir, eval_dir

    def optimize_model(self):
        train_var, chkpt_var = self.train_var, self.chkpt_var
        exec_var, map_var = self.exec_var, self.map_var
        chkpt_dir, eval_dir = self.chkpt_dir, self.eval_dir
        datasets, model, plt = self.datasets, self.model, self.plt
        train_dataset, valid_dataset, test_dataset = self.train_dataset, self.valid_dataset, self.test_dataset
        # Gather optimization variables and run
        start = time.time()
        print(util.make_msg_block("Training %s" % (model.name()), "+"))
        opt_var = Container().copy([exec_var, train_var, chkpt_var, map_var])
        opt_var.set("checkpoint_dir", chkpt_dir)
        # Initialize this model from a model checkpoint
        if isinstance(train_var.initializer, str) and os.path.exists(train_var.initializer):
            path = train_var.initializer
            if os.path.isdir(path): # Find the checkpoint id in directory ".../<model_name>/.../<chkpt.pth>"
                model_name = path.split(os.sep)[-1]
                model_ids = os.listdir(path)
                chkpt_name = "Best.pth"
                if len(model_ids) == 0:
                    raise ValueError(
                        "Directory \"%s\" does not contain any model instances to initialize %s from" % (
                            path, model.name()
                        )
                    )
                elif len(model_ids) > 1:
                    print("Found multiple model instances for initialization at \"%s\". Please choose one:" % (path))
                    print("\n".join(["%-2d : %s" % (i, model_id) for i, model_id in enumerate(model_ids)]))
                    idx = int(input("Choice: "))
                    model_id = model_ids[idx]
                else:
                    model_id = model_ids[0]
                path = os.sep.join([path, model_id, chkpt_name])
            elif os.path.isfile(path): # Simply pull name and id from checkpoint path
                model_name, model_id, chkpt_name = path.split(os.sep)[-3:]
            else:
                raise ValueError("Initializer \"%s\" was neither a directory or file" % (path))
            load_var = Container().copy([self.exec_var, self.train_var])
            opt_var.initializer = self.init_model(model_name, datasets, self.var).load(path, load_var)
        # Pull data dicts and optimize
        train_data_dict = model.pull_data(train_dataset, "train", opt_var)
        valid_data_dict = model.pull_data(valid_dataset, "valid", opt_var)
        test_data_dict = model.pull_data(test_dataset, "test", opt_var)
        model.optimize(train_data_dict, valid_data_dict, test_data_dict, opt_var)
        # Record time and save optimization variables
        self.opt_runtime = time.time() - start
        self.opt_epoch_runtime = self.opt_runtime / max(1, len(model.train_losses))
        self.opt_var = opt_var
        print(util.make_msg_block("Training %s [%.3f seconds]" % (model.name(), self.opt_runtime), "+"))

    def log_optimization_info(self):
        datasets, var = self.datasets, self.var
        chkpt_dir, eval_dir, model = self.chkpt_dir, self.eval_dir, self.model
        hyp_var, train_var, opt_var = self.hyp_var, self.train_var, self.opt_var
        plt = self.plt
        # Save training runtime
        opt_info = self.curate_optimization_info()
        path = os.sep.join([chkpt_dir, "OptimizationInfo.txt"])
        with open(path, "w") as f:
            f.write(opt_info)
        path = os.sep.join([eval_dir, "OptimizationInfo.txt"])
        with open(path, "w") as f:
            f.write(opt_info)
        # Plot learning curve
        path = os.sep.join([eval_dir, "LearningCurve.png"])
        plt.plot_learning_curve(model.train_losses, model.valid_losses, model.test_losses, path)
        # Show paths
        if not var.execution.evaluate:
            print(
                util.make_msg_block("Checkpoint Directory: %s" % (chkpt_dir.replace(os.getcwd(), ".")), "+")
            )
            print(
                util.make_msg_block("Evaluation Directory: %s" % (eval_dir.replace(os.getcwd(), ".")), "+")
            )

    def curate_optimization_info(self):
        model = self.model
        names = [
            "runtime", 
            "epoch_runtime", 
            "best_epoch", 
            "final_epoch", 
        ]
        values = [
            round(self.opt_runtime, 3),
            round(self.opt_epoch_runtime, 3), 
            -1 if not hasattr(model, "valid_losses") else model.valid_losses.index(min(model.valid_losses)), 
            -1 if not hasattr(model, "valid_losses") else len(model.valid_losses) - 1, 
        ]
        opt_var = Container().set(names, values, multi_value=True)
        return ArgumentBuilder().view(opt_var)

    def log_model_info(self):
        chkpt_dir, eval_dir, model = self.chkpt_dir, self.eval_dir, self.model
        # Save model info
        path = os.sep.join([chkpt_dir, "ModelInfo.txt"])
        with open(path, "w") as f:
            f.write(model.curate_info_str())
        path = os.sep.join([eval_dir, "ModelInfo.txt"])
        with open(path, "w") as f:
            f.write(model.curate_info_str())

    def log_settings(self):
        datasets, var = self.datasets, self.var
        chkpt_dir, eval_dir, model = self.chkpt_dir, self.eval_dir, self.model
        hyp_var, train_var = self.hyp_var, self.train_var
        # Save data settings
        data_settings = ArgumentBuilder().view(get_data_settings_var(datasets, var))
        path = os.sep.join([chkpt_dir, "DataSettings.txt"])
        with open(path, "w") as f:
            f.write(data_settings)
        path = os.sep.join([eval_dir, "DataSettings.txt"])
        with open(path, "w") as f:
            f.write(data_settings)
        # Save hyperparameter settings
        hyp_settings = ArgumentBuilder().view(hyp_var)
        path = os.sep.join([chkpt_dir, "HyperparameterSettings.txt"])
        with open(path, "w") as f:
            f.write(hyp_settings)
        path = os.sep.join([eval_dir, "HyperparameterSettings.txt"])
        with open(path, "w") as f:
            f.write(hyp_settings)
        # Save optimization settings
        opt_settings = ArgumentBuilder().view(train_var)
        path = os.sep.join([chkpt_dir, "OptimizationSettings.txt"])
        with open(path, "w") as f:
            f.write(opt_settings)
        opt_settings = ArgumentBuilder().view(train_var)
        path = os.sep.join([eval_dir, "OptimizationSettings.txt"])
        with open(path, "w") as f:
            f.write(opt_settings)

    def load_model_for_evaluation(self):
        exec_var, train_var, eval_var = self.exec_var, self.train_var, self.eval_var
        model, chkpt_dir, eval_dir = self.model, self.chkpt_dir, self.eval_dir
        # Load checkpoint and distribute the saved parameters
        path = os.sep.join([chkpt_dir, eval_var.evaluated_checkpoint])
        load_var = Container().copy([exec_var, train_var])
        model.load(path, load_var)
        self.model = model

    def get_model_predictions(self):
        # Unpack the variables
        exec_var, hyp_var, map_var = self.exec_var, self.hyp_var, self.map_var
        proc_var, train_var, eval_var = self.proc_var, self.train_var, self.eval_var
        chkpt_var, dist_var, plt_var = self.chkpt_var, self.dist_var, self.plt_var
        dbg_var, plt = self.dbg_var, self.plt
        eval_dir, chkpt_dir = self.eval_dir, self.chkpt_dir
        datasets, model, plt = self.datasets, self.model, self.plt
        train_dataset, valid_dataset, test_dataset = self.train_dataset, self.valid_dataset, self.test_dataset
        # Unpack the data
        train_spa = train_dataset.spatial
        train_tmp = train_dataset.temporal
        train_spatmp = train_dataset.spatiotemporal
        train_graph = train_dataset.graph
        valid_spa = valid_dataset.spatial
        valid_tmp = valid_dataset.temporal
        valid_spatmp = valid_dataset.spatiotemporal
        valid_graph = valid_dataset.graph
        test_spa = test_dataset.spatial
        test_tmp = test_dataset.temporal
        test_spatmp = test_dataset.spatiotemporal
        test_graph = test_dataset.graph
        # Pull groundtruth values *_Y
        if exec_var.principle_data_type == "spatial":
            if not exec_var.principle_data_form == "original":
                raise NotImplementedError(exec_var.principle_data_form)
            train_Y = train_spa.original.get("response_features", "train")
            valid_Y = valid_spa.original.get("response_features", "valid")
            test_Y = test_spa.original.get("response_features", "test")
            train_Y_gtmask = train_spa.original.get("response_gtmask", "train")
            valid_Y_gtmask = valid_spa.original.get("response_gtmask", "valid")
            test_Y_gtmask = test_spa.original.get("response_gtmask", "test")
        elif exec_var.principle_data_type == "temporal":
            raise NotImplementedError()
        elif exec_var.principle_data_type == "spatiotemporal":
            if exec_var.principle_data_form == "original":
                _, train_Y = train_spatmp.original.to_windows(
                    train_spatmp.original.get("response_features", "train"),
                    map_var.temporal_mapping[0], 
                    map_var.temporal_mapping[1], 
                    map_var.horizon
                )
                _, valid_Y = valid_spatmp.original.to_windows(
                    valid_spatmp.original.get("response_features", "valid"),
                    map_var.temporal_mapping[0], 
                    map_var.temporal_mapping[1], 
                    map_var.horizon
                )
                _, test_Y = test_spatmp.original.to_windows(
                    test_spatmp.original.get("response_features", "test"),
                    map_var.temporal_mapping[0], 
                    map_var.temporal_mapping[1], 
                    map_var.horizon
                )
                _, train_Y_gtmask = train_spatmp.original.to_windows(
                    train_spatmp.original.get("response_gtmask", "train"),
                    map_var.temporal_mapping[0], 
                    map_var.temporal_mapping[1], 
                    map_var.horizon
                )
                _, valid_Y_gtmask = valid_spatmp.original.to_windows(
                    valid_spatmp.original.get("response_gtmask", "valid"),
                    map_var.temporal_mapping[0], 
                    map_var.temporal_mapping[1], 
                    map_var.horizon
                )
                _, test_Y_gtmask = test_spatmp.original.to_windows(
                    test_spatmp.original.get("response_gtmask", "test"),
                    map_var.temporal_mapping[0], 
                    map_var.temporal_mapping[1], 
                    map_var.horizon
                )
            elif exec_var.principle_data_form == "reduced":
                _, train_Y = train_spatmp.reduced.to_windows(
                    train_spatmp.reduced.get("response_features", "train"),
                    map_var.temporal_mapping[0], 
                    map_var.temporal_mapping[1], 
                    map_var.horizon
                )
                _, valid_Y = valid_spatmp.reduced.to_windows(
                    valid_spatmp.reduced.get("response_features", "valid"),
                    map_var.temporal_mapping[0], 
                    map_var.temporal_mapping[1], 
                    map_var.horizon
                )
                _, test_Y = test_spatmp.reduced.to_windows(
                    test_spatmp.reduced.get("response_features", "test"),
                    map_var.temporal_mapping[0], 
                    map_var.temporal_mapping[1], 
                    map_var.horizon
                )
            else:
                raise NotImplementedError("Unknown principle_data_form=\"%s\"" % (exec_var.principle_data_form))
        elif exec_var.principle_data_type == "graph":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        print(util.make_msg_block("##### Evaluating Model #####", "#"))
        start = time.time()
        pred_var = Container().copy([exec_var, train_var, eval_var, map_var])
        # Pull data dicts and predict
        train_data_dict = model.pull_data(train_dataset, "train", pred_var)
        valid_data_dict = model.pull_data(valid_dataset, "valid", pred_var)
        test_data_dict = model.pull_data(test_dataset, "test", pred_var)
        if "train" in eval_var.evaluated_partitions:
            train_Yhat = model.predict(train_data_dict, pred_var)["yhat"]
        else:
            train_Yhat = train_Y
        if "valid" in eval_var.evaluated_partitions:
            valid_Yhat = model.predict(valid_data_dict, pred_var)["yhat"]
        else:
            valid_Yhat = valid_Y
        if "test" in eval_var.evaluated_partitions:
            test_Yhat = model.predict(test_data_dict, pred_var)["yhat"]
        else:
            test_Yhat = test_Y
        elapsed = time.time() - start
        print(util.make_msg_block("##### Evaluating Model [%.3f seconds] #####" % (elapsed), "#"))
        # Save the predictions
        self.train_Y, self.valid_Y, self.test_Y = train_Y, valid_Y, test_Y
        self.train_Y_gtmask, self.valid_Y_gtmask, self.test_Y_gtmask = train_Y_gtmask, valid_Y_gtmask, test_Y_gtmask
        self.train_Yhat, self.valid_Yhat, self.test_Yhat = train_Yhat, valid_Yhat, test_Yhat
        self.train_spatmp, self.train_spa, self.train_graph = train_spatmp, train_spa, train_graph
        self.valid_spatmp, self.valid_spa, self.valid_graph = valid_spatmp, valid_spa, valid_graph
        self.test_spatmp, self.test_spa, self.test_graph = test_spatmp, test_spa, test_graph

    def postprocess_predictions(self):
        # Unpack the variables
        exec_var, hyp_var, map_var = self.exec_var, self.hyp_var, self.map_var
        proc_var, train_var, eval_var = self.proc_var, self.train_var, self.eval_var
        chkpt_var, dist_var, plt_var = self.chkpt_var, self.dist_var, self.plt_var
        dbg_var, plt = self.dbg_var, self.plt
        eval_dir, chkpt_dir = self.eval_dir, self.chkpt_dir
        chkpt_dir, eval_dir = self.chkpt_dir, self.eval_dir
        datasets, model, plt = self.datasets, self.model, self.plt
        train_dataset, valid_dataset, test_dataset = self.train_dataset, self.valid_dataset, self.test_dataset
        # Unpack the data
        train_Y, valid_Y, test_Y = self.train_Y, self.valid_Y, self.test_Y
        train_Yhat, valid_Yhat, test_Yhat = self.train_Yhat, self.valid_Yhat, self.test_Yhat
        train_spatmp, train_spa, train_graph = self.train_spatmp, self.train_spa, self.train_graph
        valid_spatmp, valid_spa, valid_graph = self.valid_spatmp, self.valid_spa, self.valid_graph
        test_spatmp, test_spa, test_graph = self.test_spatmp, self.test_spa, self.test_graph
        # Undo transformation of features
        if proc_var.transform_features:
            print(util.make_msg_block("##### Reverting Transformations #####", "#"))
            start = time.time()
            if exec_var.principle_data_type == "spatial":
                tmp_var = Container().copy([train_spa.misc, proc_var])
                tmp_var.set("statistics", train_spa.statistics)
                train_Yhat = train_spa.original.transform(
                    train_Yhat, train_spa.misc.response_indices, tmp_var, revert=True
                )
                tmp_var = Container().copy([valid_spa.misc, proc_var])
                tmp_var.set("statistics", valid_spa.statistics)
                valid_Yhat = valid_spa.original.transform(
                    valid_Yhat, valid_spa.misc.response_indices, tmp_var, revert=True
                )
                tmp_var = Container().copy([test_spa.misc, proc_var])
                tmp_var.set("statistics", test_spa.statistics)
                test_Yhat = test_spa.original.transform(
                    test_Yhat, test_spa.misc.response_indices, tmp_var, revert=True
                )
            elif exec_var.principle_data_type == "temporal":
                raise NotImplementedError()
            elif exec_var.principle_data_type == "spatiotemporal":
                if exec_var.principle_data_form == "original":
                    tmp_var = Container().copy([train_spatmp.misc, proc_var])
                    tmp_var.set("statistics", train_spatmp.statistics)
                    _, periodic_indices = train_spatmp.original.to_windows(
                        train_spatmp.original.get("periodic_indices", "train"),
                        map_var.temporal_mapping[0], 
                        map_var.temporal_mapping[1], 
                        map_var.horizon
                    )
                    train_Yhat = train_spatmp.windowed.transform(
                        train_Yhat,
                        np.reshape(periodic_indices, (-1)),
                        train_spatmp.original.get("spatial_indices", "train"),
                        train_spatmp.misc.response_indices,
                        tmp_var,
                        revert=True
                    )
                    tmp_var = Container().copy([valid_spatmp.misc, proc_var])
                    tmp_var.set("statistics", valid_spatmp.statistics)
                    _, periodic_indices = valid_spatmp.original.to_windows(
                        valid_spatmp.original.get("periodic_indices", "valid"),
                        map_var.temporal_mapping[0], 
                        map_var.temporal_mapping[1], 
                        map_var.horizon
                    )
                    valid_Yhat = valid_spatmp.windowed.transform(
                        valid_Yhat,
                        np.reshape(periodic_indices, (-1)),
                        valid_spatmp.original.get("spatial_indices", "valid"),
                        valid_spatmp.misc.response_indices,
                        tmp_var,
                        revert=True
                    )
                    tmp_var = Container().copy([test_spatmp.misc, proc_var])
                    tmp_var.set("statistics", test_spatmp.statistics)
                    _, periodic_indices = test_spatmp.original.to_windows(
                        test_spatmp.original.get("periodic_indices", "test"),
                        map_var.temporal_mapping[0], 
                        map_var.temporal_mapping[1], 
                        map_var.horizon
                    )
                    test_Yhat = test_spatmp.windowed.transform(
                        test_Yhat,
                        np.reshape(periodic_indices, (-1)),
                        test_spatmp.original.get("spatial_indices", "test"),
                        test_spatmp.misc.response_indices,
                        tmp_var,
                        revert=True
                    )
                elif exec_var.principle_data_form == "reduced":
                    tmp_var = Container().copy([train_spatmp.misc, proc_var])
                    tmp_var.set("statistics", train_spatmp.reduced_statistics)
                    _, periodic_indices = train_spatmp.reduced.to_windows(
                        train_spatmp.reduced.get("periodic_indices", "train"),
                        map_var.temporal_mapping[0], 
                        map_var.temporal_mapping[1], 
                        map_var.horizon
                    )
                    train_Yhat = train_spatmp.windowed.transform(
                        train_Yhat,
                        np.reshape(periodic_indices, (-1)),
                        train_spatmp.original.get("spatial_indices", "train"),
                        train_spatmp.misc.response_indices,
                        tmp_var,
                        revert=True
                    )
                    tmp_var = Container().copy([valid_spatmp.misc, proc_var])
                    tmp_var.set("statistics", valid_spatmp.reduced_statistics)
                    _, periodic_indices = valid_spatmp.reduced.to_windows(
                        valid_spatmp.reduced.get("periodic_indices", "valid"),
                        map_var.temporal_mapping[0], 
                        map_var.temporal_mapping[1], 
                        map_var.horizon
                    )
                    valid_Yhat = valid_spatmp.windowed.transform(
                        valid_Yhat,
                        np.reshape(periodic_indices, (-1)),
                        valid_spatmp.original.get("spatial_indices", "valid"),
                        valid_spatmp.misc.response_indices,
                        tmp_var,
                        revert=True
                    )
                    tmp_var = Container().copy([test_spatmp.misc, proc_var])
                    tmp_var.set("statistics", test_spatmp.reduced_statistics)
                    _, periodic_indices = test_spatmp.reduced.to_windows(
                        test_spatmp.reduced.get("periodic_indices", "test"),
                        map_var.temporal_mapping[0], 
                        map_var.temporal_mapping[1], 
                        map_var.horizon
                    )
                    test_Yhat = test_spatmp.windowed.transform(
                        test_Yhat,
                        np.reshape(periodic_indices, (-1)),
                        test_spatmp.original.get("spatial_indices", "test"),
                        test_spatmp.misc.response_indices,
                        tmp_var,
                        revert=True
                    )
                else:
                    raise NotImplementedError("Unknown principle_data_form=\"%s\"" % (exec_var.principle_data_form))
            elif exec_var.principle_data_type == "graph":
                raise NotImplementedError()
            else:
                raise NotImplementedError()
            elapsed = time.time() - start
            print(util.make_msg_block("##### Reverting Transformations [%.3f seconds] #####" % (elapsed), "#"))
        if proc_var.adjust_predictions:
            train_Yhat = adjust_predictions(
                train_Yhat,
                train_dataset.get(exec_var.principle_data_type),
                proc_var.prediction_adjustment_map,
                proc_var.default_prediction_adjustment
            )
            valid_Yhat = adjust_predictions(
                valid_Yhat,
                valid_dataset.get(exec_var.principle_data_type),
                proc_var.prediction_adjustment_map,
                proc_var.default_prediction_adjustment
            )
            test_Yhat = adjust_predictions(
                test_Yhat,
                test_dataset.get(exec_var.principle_data_type),
                proc_var.prediction_adjustment_map,
                proc_var.default_prediction_adjustment
            )

    def quantify_prediction_accuracy(self):
        # Unpack the variables
        exec_var, eval_var, dbg_var = self.exec_var, self.eval_var, self.dbg_var
        # Unpack the data
        train_Y, valid_Y, test_Y = self.train_Y, self.valid_Y, self.test_Y
        train_Y_gtmask, valid_Y_gtmask, test_Y_gtmask = self.train_Y_gtmask, self.valid_Y_gtmask, self.test_Y_gtmask
        train_Yhat, valid_Yhat, test_Yhat = self.train_Yhat, self.valid_Yhat, self.test_Yhat
        train_spatmp, train_spa, train_graph = self.train_spatmp, self.train_spa, self.train_graph
        valid_spatmp, valid_spa, valid_graph = self.valid_spatmp, self.valid_spa, self.valid_graph
        test_spatmp, test_spa, test_graph = self.test_spatmp, self.test_spa, self.test_graph
        # Calculate NRMSE for each spatial element
        if not var.evaluating.mask_metrics:
            train_Y_gtmask, valid_Y_gtmask, test_Y_gtmask = None, None, None
        eval_con = Evaluation.evaluate_datasets(
            (train_Y, valid_Y, test_Y), (train_Yhat, valid_Yhat, test_Yhat), 
            (self.train_dataset, self.test_dataset, self.test_dataset), exec_var.principle_data_type, 
            ("train", "valid", "test"), eval_var.metrics
        ) 
        if eval_con.has("NRMSE", "train"):
            train_NRMSE = np.mean(eval_con.get("NRMSE", "train"))
            valid_NRMSE = np.mean(eval_con.get("NRMSE", "valid"))
            test_NRMSE = np.mean(eval_con.get("NRMSE", "test"))
            if dbg_var.print_errors:
                print(util.make_msg_block("Normalized Root Mean Square Error (NRMSE)", "+"))
                print(util.make_msg_block("++++  Train  +++  Valid  +++  Test  +++++", "+"))
                print(
                    util.make_msg_block(
                        "++++ %.4f  +++ %.4f  +++ %.4f +++++" % (train_NRMSE, valid_NRMSE, test_NRMSE),
                        "+"
                    )
                )
        elif eval_con.has("ACC", "train"):
            train_ACC = np.mean(eval_con.get("ACC", "train"))
            valid_ACC = np.mean(eval_con.get("ACC", "valid"))
            test_ACC = np.mean(eval_con.get("ACC", "test"))
            if dbg_var.print_errors:
                print(util.make_msg_block("+++++++++++++ Accuracy (ACC) ++++++++++++", "+"))
                print(util.make_msg_block("++++  Train  +++  Valid  +++  Test  +++++", "+"))
                print(
                    util.make_msg_block(
                        "++++ %.4f  +++ %.4f  +++ %.4f +++++" % (train_ACC, valid_ACC, test_ACC),
                        "+"
                    )
                )

    def log_prediction_info(self):
        # Unpack the variables
        exec_var, plt, plt_var = self.exec_var, self.plt, self.plt_var
        eval_var, eval_dir = self.eval_var, self.eval_dir
        # Unpack the data
        train_dataset, valid_dataset, test_dataset = self.train_dataset, self.valid_dataset, self.test_dataset
        train_Y, valid_Y, test_Y = self.train_Y, self.valid_Y, self.test_Y
        train_Y_gtmask, valid_Y_gtmask, test_Y_gtmask = self.train_Y_gtmask, self.valid_Y_gtmask, self.test_Y_gtmask
        train_Yhat, valid_Yhat, test_Yhat = self.train_Yhat, self.valid_Yhat, self.test_Yhat
        train_spatmp, train_spa, train_graph = self.train_spatmp, self.train_spa, self.train_graph
        valid_spatmp, valid_spa, valid_graph = self.valid_spatmp, self.valid_spa, self.valid_graph
        test_spatmp, test_spa, test_graph = self.test_spatmp, self.test_spa, self.test_graph
        # Curate prediction performance report
        path = os.sep.join([
            eval_dir,
            "Performance_Checkpoint[%s].txt" % (eval_var.evaluated_checkpoint.replace(".pth", ""))
        ])
        report = Evaluation.curate_evaluation_report(
            (train_Y, valid_Y, test_Y), (train_Yhat, valid_Yhat, test_Yhat), 
            (train_dataset, test_dataset, test_dataset), exec_var.principle_data_type, 
            ("train", "valid", "test"), eval_var.metrics
        ) 
        with open(path, "w") as f:
            f.write(report)
        # Cache groundtruth and prediction data
        if eval_var.cache:
            fname = "Evaluation_Partition[%s].pkl"
            if "train" in eval_var.cache_partitions:
                path = os.sep.join([eval_dir, fname % ("train")])
                eval_con = Container().set(["var", "Yhat"], [self.var, train_Yhat], multi_value=True)
                util.to_cache(eval_con, path)
            if "valid" in eval_var.cache_partitions:
                path = os.sep.join([eval_dir, fname % ("valid")])
                eval_con = Container().set(["var", "Yhat"], [self.var, valid_Yhat], multi_value=True)
                util.to_cache(eval_con, path)
            if "test" in eval_var.cache_partitions:
                path = os.sep.join([eval_dir, fname % ("test")])
                eval_con = Container().set(["var", "Yhat"], [self.var, test_Yhat], multi_value=True)
                util.to_cache(eval_con, path)
        # Plot fit of predictions to groundtruth
        train_Y, valid_Y, test_Y = self.train_Y, self.valid_Y, self.test_Y
        train_Yhat, valid_Yhat, test_Yhat = self.train_Yhat, self.valid_Yhat, self.test_Yhat
        var.plotting.set("plot_dir", eval_dir)
        if exec_var.principle_data_type == "spatial":
            if plt_var.plot_model_fit:
                from Plotting import Spatial
                spa_plt = Spatial()
                var.plotting.set("plot_dir", eval_dir)
                if "train" in plt_var.plot_partitions:
                    print("##### Plotting Training Model Fit #####")
                    start = time.time()
                    spa_plt.plot_model_fit(train_Y, train_Yhat, train_dataset, "train", var)
                    elapsed = time.time() - start
                    print("##### Plotting Training Model Fit [%.3f seconds] #####" % (elapsed))
                if "valid" in plt_var.plot_partitions:
                    print("##### Plotting Validation Model Fit #####")
                    start = time.time()
                    spa_plt.plot_model_fit(valid_Y, valid_Yhat, valid_dataset, "valid", var)
                    elapsed = time.time() - start
                    print("##### Plotting Validation Model Fit [%.3f seconds] #####" % (elapsed))
                if "test" in plt_var.plot_partitions:
                    print("##### Plotting Testing Model Fit #####")
                    start = time.time()
                    spa_plt.plot_model_fit(test_Y, test_Yhat, test_dataset, "test", var)
                    elapsed = time.time() - start
                    print("##### Plotting Testing Model Fit [%.3f seconds] #####" % (elapsed))
        elif exec_var.principle_data_type == "temporal":
            raise NotImplementedError()
        elif exec_var.principle_data_type == "spatiotemporal":
            if plt_var.plot_model_fit:
                if "train" in plt_var.plot_partitions:
                    print("##### Plotting Training Model Fit #####")
                    start = time.time()
                    plt.plot_model_fit(train_Yhat, train_spatmp, "train", var)
                    elapsed = time.time() - start
                    print("##### Plotting Training Model Fit [%.3f seconds] #####" % (elapsed))
                if "valid" in plt_var.plot_partitions:
                    print("##### Plotting Validation Model Fit #####")
                    start = time.time()
                    plt.plot_model_fit(valid_Yhat, valid_spatmp, "valid", var)
                    elapsed = time.time() - start
                    print("##### Plotting Validation Model Fit [%.3f seconds] #####" % (elapsed))
                if "test" in plt_var.plot_partitions:
                    print("##### Plotting Testing Model Fit #####")
                    start = time.time()
                    plt.plot_model_fit(test_Yhat, test_spatmp, "test", var)
                    elapsed = time.time() - start
                    print("##### Plotting Testing Model Fit [%.3f seconds] #####" % (elapsed))
        elif exec_var.principle_data_type == "graph":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        # Show paths
        print(
            util.make_msg_block("Checkpoint Directory: %s" % (self.chkpt_dir.replace(os.getcwd(), ".")), "+")
        )
        print(
            util.make_msg_block("Evaluation Directory: %s" % (self.eval_dir.replace(os.getcwd(), ".")), "+")
        )


def get_data_settings_var(datasets, var):
    train_dataset, valid_dataset, test_dataset = datasets.get("dataset", ["train", "valid", "test"])
    train_data = train_dataset.get(var.execution.principle_data_type)
    valid_data = valid_dataset.get(var.execution.principle_data_type)
    test_data = test_dataset.get(var.execution.principle_data_type)
    id_var = Container()
    id_var.datasets = Container()
    id_var.datasets.set("dataset", train_dataset.name, "train")
    id_var.datasets.set("dataset", valid_dataset.name, "valid")
    if "spatial_selection" in train_data.partitioning:
        id_var.datasets.set("spatial_selection", train_data.partitioning.get("spatial_selection", "train"), "train")
    if "spatial_selection" in valid_data.partitioning:
        id_var.datasets.set("spatial_selection", valid_data.partitioning.get("spatial_selection", "valid"), "valid")
    if "temporal_selection" in train_data.partitioning:
        id_var.datasets.set("temporal_selection", train_data.partitioning.get("temporal_selection", "train"), "train")
    if "temporal_selection" in valid_data.partitioning:
        id_var.datasets.set("temporal_selection", valid_data.partitioning.get("temporal_selection", "valid"), "valid")
    id_var.datasets.principle_data_type = var.execution.principle_data_type
    id_var.datasets.principle_data_form = var.execution.principle_data_form
    id_var.set(
        [
            "mapping", 
            "processing", 
        ], 
        [
            var.mapping, 
            var.processing, 
        ], 
        multi_value=True
    )
    return id_var


def get_data_id(datasets, var):
    id_var = get_data_settings_var(datasets, var)
    return id_var.hash(var.meta.n_id_digits)


def get_model_id(model, var):
    hyp_var = var.models.get(model.name()).hyperparameters
    return model.get_id(hyp_var, var.meta.n_id_digits)


def get_training_id(var):
    tmp_var = Container().copy(var.training)
    tmp_var.rem([])
    return tmp_var.hash(var.meta.n_id_digits)


def get_id(model, datasets, var):
    ids = [get_data_id(datasets, var), get_model_id(model, var)]
    if "training" in var.meta.id_var_additions:
        ids.append(get_training_id(var))
    return ".".join(map(str, ids))


def adjust_predictions(Yhat, data, prediction_adjustment_map, default_adjustment):
    for i, feature in enumerate(data.misc.response_features):
        adjustment = prediction_adjustment_map.get(feature, default_adjustment)
        if adjustment is None or adjustment[0] == "identity":
            continue
        if adjustment[0] == "clip":
            a, b = float(adjustment[1]), float(adjustment[2])
            Yhat[...,i] = np.clip(Yhat[...,i], a, b)
        elif adjustment[0] == "binarize":
            comparator, value = adjustment[1:]
            Yhat[...,i] = util.comparator_fn_map[comparator](Yhat[...,i], value)
        elif adjustment[0] == "round":
            digits = adjustment[1]
            Yhat[...,i] = np.round(Yhat[...,i], digits)
        else:
            raise NotImplementedError(adjustment)
    return Yhat


def make_func_args(func, var):
    req_args, var_args, kw_args, def_args = util.get_func_args(func)
    all_args = req_args + list(def_args.keys())
    func_var = var.checkout(all_args)
    func_args = {}
    for req_arg in req_args:
        func_args[req_arg] = func_var.get(req_arg, must_exist=False)
    for kw, val in def_args.items():
        if func_var.var_exists(kw):
             val = func_var.get(kw)
        func_args[kw] = val
    return func_args


if __name__ == "__main__":
    # Initialize all variables: parse arguments and merge into default variables
    argv = sys.argv[1:]
    if "--f" in argv:
        args = util.from_cache(argv[argv.index("--f")+1])
    else:
        args = ArgumentParser().parse_arguments(sys.argv[1:])
    var = Variables()
    var.merge(args)
    if var.debug.print_args:
        n = 49
        msg = " " * n + "Arguments" + " " * n
        print(util.make_msg_block(msg))
        print(args)
        print(util.make_msg_block(msg))
    Pipeline(var).execute()
