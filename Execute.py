import numpy as np
import time
import sys
import os
import NerscDistributed as nersc_dist
import Utility as util
from progressbar import ProgressBar
import importlib
from Plotting import Plotting
from Variables import Variables
from Arguments import ArgumentParser, ArgumentBuilder
from Container import Container
from Data.Data import Data


class Pipeline:

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    def __init__(self, var=None):
        if var is None: var = Variables()
        self.init_variables(var)

    def execute(self):
        train_var, eval_var = self.train_var, self.eval_var
        # Initialize and synchronize processes
        self.init_processes()
        # Initialize and unpack the data
        self.init_data()
        # Pre-process the data
        self.preprocess_data()
        # Debug information
        self.log_debug_info()
        # Initialize the model, and its checkpoint + evaluation directories
        self.init_model()
        self.init_paths()
        self.log_model_config()
        self.log_model_info()
        ##########################
        ### MODEL OPTIMIZATION ###
        ##########################
        if train_var.get("train"):
            self.optimize_model()
            self.log_optimization_info()
        ########################
        ### MODEL EVALUATION ###
        ########################
        if eval_var.get("evaluate"):
            self.load_model_for_evaluation()
            self.get_model_predictions()
            self.postprocess_predictions()
            self.quantify_prediction_accuracy()
            self.log_prediction_info()

    def init_variables(self, var):
        # Unpack variables
        exec_var = var.get("execution")
        hyp_var = var.get("models").get(exec_var.get("model")).get("hyperparameters")
        map_var = var.get("mapping")
        proc_var = var.get("processing")
        train_var = var.get("training")
        eval_var = var.get("evaluating")
        chkpt_var = var.get("checkpointing")
        dist_var = var.get("distribution")
        plt_var = var.get("plotting")
        graph_var = var.get("graph")
        dbg_var = var.get("debug")
        plt = Plotting()
        proc_rank = dist_var.get("process_rank")
        root_proc_rank = dist_var.get("root_process_rank")
        n_procs = dist_var.get("n_processes")
        # Make edits
        train_var.copy(var.get("models").get(exec_var.get("model")).get("training"))
        # Set variables
        self.var, self.exec_var, self.hyp_var = var, exec_var, hyp_var
        self.map_var, self.proc_var, self.train_var = map_var, proc_var, train_var
        self.eval_var, self.chkpt_var, self.dist_var = eval_var, chkpt_var, dist_var
        self.plt_var, self.graph_var, self.dbg_var = plt_var, graph_var, dbg_var
        self.plt = plt
        self.proc_rank, self.root_proc_rank, self.n_procs = proc_rank, root_proc_rank, n_procs

    def init_processes(self):
        dist_var = self.dist_var
        proc_rank, root_proc_rank, n_procs = self.proc_rank, self.root_proc_rank, self.n_procs
        if dist_var.get("nersc"):
            proc_rank, n_procs = nersc_dist.init_workers(backend=dist_var.get("backend"))
        elif n_procs > 1:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "54321"
            torch_dist.init_process_group(backend=dist_var.get("backend"), world_size=n_procs, rank=proc_rank)

    def init_data(self):
        var = self.var
        data = Data(var)
        self.data = data

    def preprocess_data(self):
        # Already completed in the component classes of class Data
        pass

    def log_debug_info(self):
        var, dbg_var, data = self.var, self.dbg_var, self.data
        proc_rank, root_proc_rank, n_procs = self.proc_rank, self.root_proc_rank, self.n_procs
        if proc_rank == root_proc_rank:
            if dbg_var.get("print_vars"):
                n = 46
                msg = " " * n + "Variables" + " " * n
                print(util.make_msg_block(msg))
                print(var)
                print(util.make_msg_block(msg))
            if dbg_var.get("print_data"):
                n = 48
                msg = " " * n + "Data" + " " * n
                print(util.make_msg_block(msg))
                print(data)
                print(util.make_msg_block(msg))
            if dbg_var.get("data_memory"):
                print("Training Dataset Memory Usage =", util.get_memory_of(data.get("dataset", "train")))
                if proc_rank == root_proc_rank:
                    print("Validation Dataset Memory Usage =", util.get_memory_of(data.get("dataset", "valid")))
                    print("Testing Dataset Memory Usage =", util.get_memory_of(data.get("dataset", "test")))

    def init_model(self):
        exec_var, data = self.exec_var, self.data
        model_module = importlib.import_module("Models." + exec_var.get("model"))
        model = model_module.init(data.get("dataset", "train"), var)
        self.model = model

    def init_paths(self):
        exec_var, proc_var, map_var = self.exec_var, self.proc_var, self.map_var
        hyp_var, graph_var, chkpt_var = self.hyp_var, self.graph_var, self.chkpt_var
        eval_var, data, model = self.eval_var, self.data, self.model
        proc_rank, root_proc_rank, n_procs = self.proc_rank, self.root_proc_rank, self.n_procs
        train_spatmp = data.get("dataset", "train").get("spatiotemporal")
        config_var = Container().copy([exec_var, proc_var, map_var, hyp_var, graph_var])
        config_var.set(
            "spatial_selection", 
            train_spatmp.get("partitioning").get("spatial_selection", ["train", "valid", "test"]), 
            ["train", "valid", "test"], 
            multi_value=True
        )
        config_var.set(
            "temporal_selection", 
            train_spatmp.get("partitioning").get("temporal_selection", ["train", "valid", "test"]), 
            ["train", "valid", "test"], 
            multi_value=True
        )
        config_var.set(
            ["n_temporal_in", "n_temporal_out"], 
            map_var.get("temporal_mapping"), 
            multi_value=True
        )
        chkpt_dir = os.sep.join([chkpt_var.get("checkpoint_dir"), model.name(), model.get_id(config_var)])
        eval_dir = os.sep.join([eval_var.get("evaluation_dir"), model.name(), model.get_id(config_var)])
        if proc_rank == root_proc_rank:
            os.makedirs(chkpt_dir, exist_ok=True)
            os.makedirs(eval_dir, exist_ok=True)
        self.config_var, self.chkpt_dir, self.eval_dir = config_var, chkpt_dir, eval_dir

    def log_model_config(self):
        chkpt_dir, eval_dir, config_var = self.chkpt_dir, self.eval_dir, self.config_var
        model = self.model
        # Save configuration to checkpoint directory
        config_path = os.sep.join([chkpt_dir, "Configuration.txt"])
        with open(config_path, "w") as f:
            f.write(model.curate_config(config_var))
        # Save configuration to evaluation directory
        config_path = os.sep.join([eval_dir, "Configuration.txt"])
        with open(config_path, "w") as f:
            f.write(model.curate_config(config_var))

    def optimize_model(self):
        train_var, chkpt_var = self.train_var, self.chkpt_var
        config_var, map_var = self.config_var, self.map_var
        chkpt_dir, eval_dir = self.chkpt_dir, self.eval_dir
        proc_rank, root_proc_rank, n_procs = self.proc_rank, self.root_proc_rank, self.n_procs
        data, model, plt = self.data, self.model, self.plt
        # Gather optimization variables and run
        start = time.time()
        if proc_rank == root_proc_rank:
            print("[%d] -> .....Training %s....." % (proc_rank, model.name()))
        opt_var = Container().copy([train_var, chkpt_var, map_var])
        opt_var.set("checkpoint_dir", chkpt_dir)
        model.optimize(
            data.get("dataset", "train"), 
            data.get("dataset", "valid"), 
            data.get("dataset", "test"), 
            opt_var
        )
        self.opt_runtime = time.time() - start
        self.opt_var = opt_var
        if proc_rank == root_proc_rank:
            print("[%d] -> .....Training %s..... [%.3f seconds]" % (proc_rank, model.name(), self.opt_runtime))

    def log_optimization_info(self):
        proc_rank, root_proc_rank, n_procs = self.proc_rank, self.root_proc_rank, self.n_procs
        chkpt_dir, eval_dir, model = self.chkpt_dir, self.eval_dir, self.model
        hyp_var, train_var, opt_var = self.hyp_var, self.train_var, self.opt_var
        plt = self.plt
        # Save training runtime 
        if proc_rank == root_proc_rank and train_var.get("train"):
            opt_info = self.curate_optimization_info()
            path = os.sep.join([chkpt_dir, "OptimizationInfo.txt"])
            with open(path, "w") as f:
                f.write(opt_info)
        # Plot learning curve
        if proc_rank == root_proc_rank:
            path = os.sep.join([eval_dir, "LearningCurve.png"])
            plt.plot_learning_curve(model.train_losses, model.valid_losses, model.test_losses, path)
        # Save hyperparameter settings
        if proc_rank == root_proc_rank:
            hyp_settings = ArgumentBuilder().view(hyp_var)
            path = os.sep.join([chkpt_dir, "HyperparameterSettings.txt"])
            with open(path, "w") as f:
                f.write(hyp_settings)
        # Save optimization settings
        if proc_rank == root_proc_rank:
            opt_settings = ArgumentBuilder().view(train_var)
            path = os.sep.join([chkpt_dir, "OptimizationSettings.txt"])
            with open(path, "w") as f:
                f.write(opt_settings)

    def curate_optimization_info(self):
        model, opt_runtime = self.model, self.opt_runtime
        lines = []
        lines += ["Runtime(seconds): %.3f" % (opt_runtime)]
        return "\n".join(lines)

    def log_model_info(self):
        proc_rank, root_proc_rank, n_procs = self.proc_rank, self.root_proc_rank, self.n_procs
        chkpt_dir, model = self.chkpt_dir, self.model
        # Save model info
        if proc_rank == root_proc_rank:
            path = os.sep.join([chkpt_dir, "ModelInfo.txt"])
            with open(path, "w") as f:
                f.write(model.curate_info())

    def load_model_for_evaluation(self):
        dist_var, train_var, eval_var = self.dist_var, self.train_var, self.eval_var
        model, chkpt_dir, eval_dir = self.model, self.chkpt_dir, self.eval_dir
        # Load checkpoint and distribute the saved parameters
        path = os.sep.join([chkpt_dir, eval_var.get("evaluated_checkpoint")])
        load_var = Container().copy([dist_var, train_var])
        model.load(load_var, path)
        self.model = model

    def get_model_predictions(self):
        # Unpack the variables
        exec_var, hyp_var, map_var = self.exec_var, self.hyp_var, self.map_var
        proc_var, train_var, eval_var = self.proc_var, self.train_var, self.eval_var
        chkpt_var, dist_var, plt_var = self.chkpt_var, self.dist_var, self.plt_var
        graph_var, dbg_var, plt = self.graph_var, self.dbg_var, self.plt
        config_var, eval_dir, chkpt_dir = self.config_var, self.eval_dir, self.chkpt_dir
        proc_rank, root_proc_rank, n_procs = self.proc_rank, self.root_proc_rank, self.n_procs
        data, model, plt = self.data, self.model, self.plt
        # Unpack the data
        train_spatmp = data.get("dataset", "train").get("spatiotemporal")
        train_spa = data.get("dataset", "train").get("spatial")
        train_graph = data.get("dataset", "train").get("graph")
        valid_spatmp = data.get("dataset", "valid").get("spatiotemporal")
        valid_spa = data.get("dataset", "valid").get("spatial")
        valid_graph = data.get("dataset", "valid").get("graph")
        test_spatmp = data.get("dataset", "test").get("spatiotemporal")
        test_spa = data.get("dataset", "test").get("spatial")
        test_graph = data.get("dataset", "test").get("graph")
        # Make the predictions
        _, train_Y = train_spatmp.reduced.to_windows(
            train_spatmp.reduced.get("response_features", "train"), 
            map_var.temporal_mapping[0], 
            map_var.temporal_mapping[1]
        )
        _, valid_Y = valid_spatmp.reduced.to_windows(
            valid_spatmp.reduced.get("response_features", "valid"), 
            map_var.temporal_mapping[0], 
            map_var.temporal_mapping[1]
        )
        _, test_Y = test_spatmp.reduced.to_windows(
            test_spatmp.reduced.get("response_features", "test"), 
            map_var.temporal_mapping[0], 
            map_var.temporal_mapping[1]
        )
        train_Y = np.reshape(train_Y, (-1,) + train_Y.shape[2:])
        valid_Y = np.reshape(valid_Y, (-1,) + valid_Y.shape[2:])
        test_Y = np.reshape(test_Y, (-1,) + test_Y.shape[2:])
        msg = "Evaluation Directory: %s" % (eval_dir)
        print(util.make_msg_block(msg, "+"))
        print("[%d] -> .....Evaluating Model....." % (proc_rank))
        start = time.time()
        pred_var = Container().copy([train_var, eval_var, map_var])
        if "train" in eval_var.get("evaluated_partitions"):
            train_Yhat = model.predict(data.get("dataset", "train"), "train", pred_var)
        else:
            train_Yhat = train_Y
        if "valid" in eval_var.get("evaluated_partitions"):
            valid_Yhat = model.predict(data.get("dataset", "valid"), "valid", pred_var)
        else:
            valid_Yhat = valid_Y
        if "test" in eval_var.get("evaluated_partitions"):
            test_Yhat = model.predict(data.get("dataset", "test"), "test", pred_var)
        else:
            test_Yhat = test_Y
        elapsed = time.time() - start
        print("[%d] -> .....Evaluating Model..... [%.3f seconds]" % (proc_rank, elapsed))
        # Save the predictions
        self.train_Y, self.valid_Y, self.test_Y = train_Y, valid_Y, test_Y
        self.train_Yhat, self.valid_Yhat, self.test_Yhat = train_Yhat, valid_Yhat, test_Yhat
        self.train_spatmp, self.train_spa, self.train_graph = train_spatmp, train_spa, train_graph
        self.valid_spatmp, self.valid_spa, self.valid_graph = valid_spatmp, valid_spa, valid_graph
        self.test_spatmp, self.test_spa, self.test_graph = test_spatmp, test_spa, test_graph

    def postprocess_predictions(self):
        # Unpack the variables
        exec_var, hyp_var, map_var = self.exec_var, self.hyp_var, self.map_var
        proc_var, train_var, eval_var = self.proc_var, self.train_var, self.eval_var
        chkpt_var, dist_var, plt_var = self.chkpt_var, self.dist_var, self.plt_var
        graph_var, dbg_var, plt = self.graph_var, self.dbg_var, self.plt
        config_var, eval_dir, chkpt_dir = self.config_var, self.eval_dir, self.chkpt_dir
        proc_rank, root_proc_rank, n_procs = self.proc_rank, self.root_proc_rank, self.n_procs
        chkpt_dir, eval_dir = self.chkpt_dir, self.eval_dir
        data, model, plt = self.data, self.model, self.plt
        # Unpack the data
        train_Y, valid_Y, test_Y = self.train_Y, self.valid_Y, self.test_Y
        train_Yhat, valid_Yhat, test_Yhat = self.train_Yhat, self.valid_Yhat, self.test_Yhat
        train_spatmp, train_spa, train_graph = self.train_spatmp, self.train_spa, self.train_graph
        valid_spatmp, valid_spa, valid_graph = self.valid_spatmp, self.valid_spa, self.valid_graph
        test_spatmp, test_spa, test_graph = self.test_spatmp, self.test_spa, self.test_graph
        # Undo transformation of features
        tmp_var = Container().copy([train_spatmp.get("misc"), proc_var])
        tmp_var.set("metrics", train_spatmp.get("metrics"))
        if proc_var.get("transform_features"):
            print("[%d] -> .....Reverting Transformations....." % (proc_rank))
            start = time.time()
            _, periodic_indices = train_spatmp.reduced.to_windows(
                train_spatmp.reduced.get("periodic_indices", "train"), 
                map_var.temporal_mapping[0], 
                map_var.temporal_mapping[1]
            )
            train_Yhat = train_spatmp.windowed.transform(
                train_Yhat,
                np.reshape(periodic_indices, (-1)), 
                train_spatmp.get("original").get("spatial_indices", "train"),
                train_spatmp.get("misc").get("response_indices"),
                tmp_var,
                revert=True
            )
        tmp_var.set("metrics", valid_spatmp.get("metrics"))
        if proc_var.get("transform_features"):
            _, periodic_indices = valid_spatmp.reduced.to_windows(
                valid_spatmp.reduced.get("periodic_indices", "valid"), 
                map_var.temporal_mapping[0], 
                map_var.temporal_mapping[1]
            )
            valid_Yhat = valid_spatmp.windowed.transform(
                valid_Yhat,
                np.reshape(periodic_indices, (-1)), 
                valid_spatmp.get("original").get("spatial_indices", "valid"),
                valid_spatmp.get("misc").get("response_indices"),
                tmp_var, 
                revert=True
            )
        tmp_var.set("metrics", test_spatmp.get("metrics"))
        if proc_var.get("transform_features"):
            _, periodic_indices = test_spatmp.reduced.to_windows(
                test_spatmp.reduced.get("periodic_indices", "test"), 
                map_var.temporal_mapping[0], 
                map_var.temporal_mapping[1]
            )
            test_Yhat = test_spatmp.windowed.transform(
                test_Yhat,
                np.reshape(periodic_indices, (-1)), 
                test_spatmp.get("original").get("spatial_indices", "test"),
                test_spatmp.get("misc").get("response_indices"),
                tmp_var, 
                revert=True
            )
            elapsed = time.time() - start
            print("[%d] -> .....Reverting Transformations..... [%.3f seconds]" % (proc_rank, elapsed))
        if proc_var.get("adjust_predictions"):
            train_Yhat = adjust_predictions(
                train_Yhat, 
                train_spatmp, 
                proc_var.get("prediction_adjustment_map"), 
                proc_var.get("default_prediction_adjustment")
            )
        if proc_var.get("adjust_predictions"):
            valid_Yhat = adjust_predictions(
                valid_Yhat, 
                valid_spatmp, 
                proc_var.get("prediction_adjustment_map"), 
                proc_var.get("default_prediction_adjustment")
            )
        if proc_var.get("adjust_predictions"):
            test_Yhat = adjust_predictions(
                test_Yhat, 
                test_spatmp, 
                proc_var.get("prediction_adjustment_map"), 
                proc_var.get("default_prediction_adjustment")
            )

    def quantify_prediction_accuracy(self):
        # Unpack the variables
        proc_rank, root_proc_rank, n_procs = self.proc_rank, self.root_proc_rank, self.n_procs
        dbg_var = self.dbg_var
        # Unpack the data
        train_Y, valid_Y, test_Y = self.train_Y, self.valid_Y, self.test_Y
        train_Yhat, valid_Yhat, test_Yhat = self.train_Yhat, self.valid_Yhat, self.test_Yhat
        train_spatmp, train_spa, train_graph = self.train_spatmp, self.train_spa, self.train_graph
        valid_spatmp, valid_spa, valid_graph = self.valid_spatmp, self.valid_spa, self.valid_graph
        test_spatmp, test_spa, test_graph = self.test_spatmp, self.test_spa, self.test_graph
        # Calculate NRMSE for each subbasin
        train_mins = train_spatmp.filter_axis(
            train_spatmp.get("metrics").get("minimums"),
            [1, 2],
            [
                train_spatmp.get("original").get("spatial_indices", "train"), 
                train_spatmp.get("misc").get("response_indices")
            ]
        )
        train_maxes = train_spatmp.filter_axis(
            train_spatmp.get("metrics").get("maximums"),
            [1, 2],
            [
                train_spatmp.get("original").get("spatial_indices", "train"), 
                train_spatmp.get("misc").get("response_indices")
            ]
        )
        valid_mins = valid_spatmp.filter_axis(
            valid_spatmp.get("metrics").get("minimums"),
            [1, 2],
            [
                valid_spatmp.get("original").get("spatial_indices", "valid"), 
                valid_spatmp.get("misc").get("response_indices")
            ]
        )
        valid_maxes = valid_spatmp.filter_axis(
            valid_spatmp.get("metrics").get("maximums"),
            [1, 2],
            [
                valid_spatmp.get("original").get("spatial_indices", "valid"), 
                valid_spatmp.get("misc").get("response_indices")
            ]
        )
        test_mins = test_spatmp.filter_axis(
            test_spatmp.get("metrics").get("minimums"),
            [1, 2],
            [
                test_spatmp.get("original").get("spatial_indices", "test"), 
                test_spatmp.get("misc").get("response_indices")
            ]
        )
        test_maxes = test_spatmp.filter_axis(
            test_spatmp.get("metrics").get("maximums"),
            [1, 2],
            [
                test_spatmp.get("original").get("spatial_indices", "test"), 
                test_spatmp.get("misc").get("response_indices")
            ]
        )
        train_NRMSEs = util.NRMSE(train_Y, train_Yhat, mins=train_mins, maxes=train_maxes)
        valid_NRMSEs = util.NRMSE(valid_Y, valid_Yhat, mins=valid_mins, maxes=valid_maxes)
        test_NRMSEs = util.NRMSE(test_Y, test_Yhat, mins=test_mins, maxes=test_maxes)
        if dbg_var.get("print_spatial_errors"):
            print("train_NRMSEs =", train_NRMSEs.shape, "=")
            for i in range(train_NRMSEs.shape[0]):
                print(train_spatmp.get("original").get("spatial_labels", "train")[i], train_NRMSEs[i])
            print("valid_NRMSEs =", valid_NRMSEs.shape, "=")
            for i in range(valid_NRMSEs.shape[0]):
                print(valid_spatmp.get("original").get("spatial_labels", "valid")[i], valid_NRMSEs[i])
            print("test_NRMSEs =", test_NRMSEs.shape, "=")
            for i in range(test_NRMSEs.shape[0]):
                print(test_spatmp.get("original").get("spatial_labels", "test")[i], test_NRMSEs[i])
        train_NRMSE = np.mean(train_NRMSEs)
        valid_NRMSE = np.mean(valid_NRMSEs)
        test_NRMSE = np.mean(test_NRMSEs)
        if dbg_var.get("print_errors"):
            print(util.make_msg_block("Normalized Root Mean Square Error (NRMSE)", "+"))
            print(util.make_msg_block("++++  Train  +++  Valid  +++  Test  +++++", "+"))
            print(
                util.make_msg_block(
                    "++++ %.4f  +++ %.4f  +++ %.4f +++++" % (train_NRMSE, valid_NRMSE, test_NRMSE), 
                    "+"
                )
            )
        self.train_mins, self.valid_mins, self.test_mins = train_mins, valid_mins, test_mins
        self.train_maxes, self.valid_maxes, self.test_maxes = train_maxes, valid_maxes, test_maxes
        

    def log_prediction_info(self):
        # Unpack the variables
        plt, plt_var = self.plt, self.plt_var
        eval_var, eval_dir = self.eval_var, self.eval_dir
        proc_rank, root_proc_rank, n_procs = self.proc_rank, self.root_proc_rank, self.n_procs
        # Unpack the data
        train_Y, valid_Y, test_Y = self.train_Y, self.valid_Y, self.test_Y
        train_Yhat, valid_Yhat, test_Yhat = self.train_Yhat, self.valid_Yhat, self.test_Yhat
        train_spatmp, train_spa, train_graph = self.train_spatmp, self.train_spa, self.train_graph
        valid_spatmp, valid_spa, valid_graph = self.valid_spatmp, self.valid_spa, self.valid_graph
        test_spatmp, test_spa, test_graph = self.test_spatmp, self.test_spa, self.test_graph
        train_mins, valid_mins, test_mins = self.train_mins, self.valid_mins, self.test_mins
        train_maxes, valid_maxes, test_maxes = self.train_maxes, self.valid_maxes, self.test_maxes
        # Curate prediction performance report
        path = os.sep.join([
            eval_dir, 
            "Performance_Checkpoint[%s].txt" % (eval_var.get("evaluated_checkpoint").replace(".pth", ""))
        ])
        report = util.curate_performance_report(
            [train_Y, valid_Y, test_Y],
            [train_Yhat, valid_Yhat, test_Yhat],
            ["train", "valid", "test"],
            [
                train_spatmp.get("original").get("spatial_labels", "train"),
                valid_spatmp.get("original").get("spatial_labels", "valid"),
                test_spatmp.get("original").get("spatial_labels", "test")
            ],
            [
                train_spatmp.get("loading").get("spatial_label_field").capitalize(), 
                valid_spatmp.get("loading").get("spatial_label_field").capitalize(), 
                test_spatmp.get("loading").get("spatial_label_field").capitalize(),
            ], 
            train_spatmp.get("misc").get("response_features"),
            eval_var.get("metrics"), 
            [
                {"mins": train_mins, "maxes": train_maxes}, 
                {"mins": valid_mins, "maxes": valid_maxes}, 
                {"mins": test_mins, "maxes": test_maxes}, 
            ], 
        )
        with open(path, "w") as f:
            f.write(report)
        # Cache groundtruth and prediction data
        if eval_var.get("cache"):
            fname = "Evaluation_Partition[%s].pkl"
            if "train" in eval_var.get("cache_partitions"):
                path = os.sep.join([eval_dir, fname % ("train")])
                eval_con = Container().set(["var", "Yhat"], [self.var, train_Yhat], multi_value=True)
                util.to_cache(eval_con, path)
            if "valid" in eval_var.get("cache_partitions"):
                path = os.sep.join([eval_dir, fname % ("valid")])
                eval_con = Container().set(["var", "Yhat"], [self.var, valid_Yhat], multi_value=True)
                util.to_cache(eval_con, path)
            if "test" in eval_var.get("cache_partitions"):
                path = os.sep.join([eval_dir, fname % ("test")])
                eval_con = Container().set(["var", "Yhat"], [self.var, test_Yhat], multi_value=True)
                util.to_cache(eval_con, path)
        # Plot fit of predictions to groundtruth
        if plt_var.get("plot_model_fit"):
            var.get("plotting").set("plot_dir", eval_dir)
            if "train" in plt_var.get("plot_partitions"):
                print("[%d] -> .....Plotting Training Model Fit....." % (proc_rank))
                start = time.time()
                plt.plot_model_fit(train_Yhat, train_spatmp, "train", var)
                elapsed = time.time() - start
                print("[%d] -> .....Plotting Training Model Fit..... [%.3f seconds]" % (proc_rank, elapsed))
            if "valid" in plt_var.get("plot_partitions"):
                print("[%d] -> .....Plotting Validation Model Fit....." % (proc_rank))
                start = time.time()
                plt.plot_model_fit(valid_Yhat, valid_spatmp, "valid", var)
                elapsed = time.time() - start
                print("[%d] -> .....Plotting Validation Model Fit..... [%.3f seconds]" % (proc_rank, elapsed))
            if "test" in plt_var.get("plot_partitions"):
                print("[%d] -> .....Plotting Testing Model Fit....." % (proc_rank))
                start = time.time()
                plt.plot_model_fit(test_Yhat, test_spatmp, "test", var)
                elapsed = time.time() - start
                print("[%d] -> .....Plotting Testing Model Fit..... [%.3f seconds]" % (proc_rank, elapsed))

def adjust_predictions(Yhat, spatmp, prediction_adjustment_map, default_adjustment):
    for i in range(spatmp.get("misc").get("n_responses")):
        feature = spatmp.get("misc").get("response_features")[i]
        if feature not in prediction_adjustment_map:
            adjustment = default_adjustment
        else:
            adjustment = prediction_adjustment_map[feature]
        if adjustment[0] == "none":
            Yhat = Yhat
        elif adjustment[0] == "limit":
            a, b = int(adjustment[1]), (None if adjustment[2] == "+" else int(adjustment[2]))
            Yhat[:,:,:,i] = np.clip(Yhat[:,:,:,i], a, b)
        else:
            raise NotImplementedError()
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
    args = ArgumentParser().parse_arguments(sys.argv[1:])
    var = Variables()
    var.merge(args)
    if var.get("debug").get("print_args"):
        n = 49
        msg = " " * n + "Arguments" + " " * n
        print(util.make_msg_block(msg))
        print(args)
        print(util.make_msg_block(msg))
    Pipeline(var).execute()
