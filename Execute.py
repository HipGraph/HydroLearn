import numpy as np
import time
import datetime as dt
from progressbar import ProgressBar
import sys
import os
import NerscDistributed as nersc_dist
import Utility as util
from importlib import import_module
from Plotting import Plotting
from Variables import Variables
from Arguments import ArgumentParser
from Container import Container
from SpatiotemporalData import SpatiotemporalData
from Data.Data import Data


np.set_printoptions(precision=4, suppress=True, linewidth=200)


def main(argv):
    var = Variables()
    args = ArgumentParser(argv[1:])
    var.merge(args)
    # Unpack all variables
    print(var.to_string(True))
    quit()
    plt = Plotting()
    exec_var = var.get("execution")
    hyp_var = var.get("models").get(exec_var.get("model")).get("hyperparameters")
    train_var = var.get("training")
    eval_var = var.get("evaluating")
    chkpt_var = var.get("checkpointing")
    dist_var = var.get("distribution")
    plt_var = var.get("plotting")
    grph_var = var.get("graph")
    dbg_var = var.get("debug")
    proc_rank = dist_var.get("process_rank")
    root_proc_rank = dist_var.get("root_process_rank")
    n_procs = dist_var.get("n_processes")
    # Synchronize processes
    if dist_var.get("nersc"):
        proc_rank, n_procs = nersc_dist.init_workers(backend=dist_var.get("backend"))
    elif n_procs > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "54321"
        torch_dist.init_process_group(backend=dist_var.get("backend"), world_size=n_procs, rank=proc_rank)
    # Instantiate data
    data = Data(var)
    train_spatmp = data.get("spatiotemporal", "train")
    if proc_rank == root_proc_rank:
        valid_spatmp = data.get("spatiotemporal", "valid")
        test_spatmp = data.get("spatiotemporal", "test")
    grph = data.get("graph", "train")
    spa = data.get("spatial", "train")
    print_data = True
    print_data = False
    if dbg_var.get("print_data")[0]:
        msg = "Spatiotemporal Data"
        print(util.make_msg_block(msg, "#"))
        print(data.to_string(dbg_var.get("print_data")[1]))
        print(util.make_msg_block(msg, "#"))
    # Initialize model
    model_module = import_module("Models."+exec_var.get("model"))
    init_var = Container().copy([hyp_var, train_spatmp, grph, spa])
    model = model_module.init(init_var)
    # Create model checkpoint and evaluation directories
    config_var = Container().copy([exec_var, train_spatmp, hyp_var, grph_var])
    chkpt_dir = util.path([chkpt_var.get("checkpoint_dir"), model.name(), model.get_id(config_var)])
    eval_dir = util.path([eval_var.get("evaluation_dir"), model.name(), model.get_id(config_var)])
    if proc_rank == root_proc_rank:
        os.makedirs(chkpt_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
    # Unpack the data
    train_X = train_spatmp.get("transformed_input_windowed", "train")
    train_Y = train_spatmp.get("transformed_output_windowed", "train")
    if proc_rank == root_proc_rank:
        valid_X = valid_spatmp.get("transformed_input_windowed", "valid")
        valid_Y = valid_spatmp.get("transformed_output_windowed", "valid")
        test_X = test_spatmp.get("transformed_input_windowed", "test")
        test_Y = test_spatmp.get("transformed_output_windowed", "test")
    if dbg_var.get("data_memory"):
        print("Training Memory Usage =", util.get_memory_of(train_X), util.get_memory_of(train_Y))
        if proc_rank == root_proc_rank:
            print("Validation Memory Usage =", util.get_memory_of(valid_X), util.get_memory_of(valid_Y))
            print("Testing Memory Usage =", util.get_memory_of(train_X), util.get_memory_of(train_Y))
    # Execute model optimization
    if train_var.get("train"):
        # Save configuration to checkpoint directory
        config_path = util.path([chkpt_dir, "Configuration.txt"])
        with open(config_path, "w") as f:
            f.write(model.curate_config(config_var))
        # Checkout data used by this model
        train, valid, test = data_checkout(data, model, "train", "train"), None, None
        if proc_rank == root_proc_rank:
            valid, test = data_checkout(data, model, "valid", "train"), data_checkout(data, model, "test", "train")
        if proc_rank == root_proc_rank:
            print("[%d] -> .....Training %s....." % (proc_rank, model.name()))
            start = time.time()
        # Construct optimization keyword arguments and run
        opt_var = Container().copy([train_var, chkpt_var])
        opt_var.set(["train", "valid", "test", "chkpt_dir"], [train, valid, test, chkpt_dir])
        opt_args = make_func_args(model.optimize, opt_var)
        model.optimize(**opt_args)
        # Save training runtime and plot learning curve
        if proc_rank == root_proc_rank:
            elapsed = time.time() - start
            print("[%d] -> .....Training %s..... [%.3f seconds]" % (proc_rank, model.name(), elapsed))
            path = util.path([chkpt_dir, "TrainingRuntime.txt"])
            with open(path, "w") as f:
                f.write("%.3f seconds" % (elapsed))
            path = util.path([eval_dir, "LearningCurve.png"])
            plt.plot_learning_curve(model.train_losses, model.valid_losses, model.test_losses, path)
    # Execute model Evaluation
    if eval_var.get("evaluate"):
        # Save configuration to evaluation directory
        config_path = util.path([eval_dir, "Configuration.txt"])
        with open(config_path, "w") as f:
            f.write(model.curate_config(config_var))
        # Load checkpoint and distribute the saved parameters
        chkpt_path = util.path([chkpt_dir, eval_var.get("evaluated_checkpoint")])
        load_var = Container().copy([dist_var, train_var])
        model.load(load_var, chkpt_path)
        if proc_rank == root_proc_rank:
            msg = "Evaluation Directory: %s" % (eval_dir)
            print(util.make_msg_block(msg, "+"))
            print("[%d] -> .....Evaluating Model....." % (proc_rank))
            start = time.time()
            train = data_checkout(data, model, "train", "evaluate")
            valid = data_checkout(data, model, "valid", "evaluate")
            test = data_checkout(data, model, "test", "evaluate")
            train_Yhat = model.predict(train)
            valid_Yhat = model.predict(valid)
            test_Yhat = model.predict(test)
            # Undo transformation of features
            if train_spatmp.get("transform_features"):
                print("[%d] -> .....Reverting Transformations....." % (proc_rank))
                start = time.time()
                train_Yhat = train_spatmp.transform_windowed(
                    train_Yhat,
                    [1, 2, 3],
                    util.convert_dates_to_daysofyear(
                        train_spatmp.get("output_windowed_temporal_labels", "train")
                    ) - 1,
                    train_spatmp.get("original_spatial_indices", "train"),
                    train_spatmp.get("response_indices"),
                    train_spatmp,
                    revert=True
                )
                train_Y = train_spatmp.transform_windowed(
                    train_Y,
                    [1, 2, 3],
                    util.convert_dates_to_daysofyear(
                        train_spatmp.get("output_windowed_temporal_labels", "train")
                    ) - 1,
                    train_spatmp.get("original_spatial_indices", "train"),
                    train_spatmp.get("response_indices"),
                    train_spatmp,
                    revert=True
                )
            if valid_spatmp.get("transform_features"):
                valid_Yhat = valid_spatmp.transform_windowed(
                    valid_Yhat,
                    [1, 2, 3],
                    util.convert_dates_to_daysofyear(
                        valid_spatmp.get("output_windowed_temporal_labels", "valid")
                    ) - 1,
                    valid_spatmp.get("original_spatial_indices", "valid"),
                    valid_spatmp.get("response_indices"),
                    valid_spatmp,
                    revert=True
                )
                valid_Y = valid_spatmp.transform_windowed(
                    valid_Y,
                    [1, 2, 3],
                    util.convert_dates_to_daysofyear(
                        valid_spatmp.get("output_windowed_temporal_labels", "valid")
                    ) - 1,
                    valid_spatmp.get("original_spatial_indices", "valid"),
                    valid_spatmp.get("response_indices"),
                    valid_spatmp,
                    revert=True
                )
            if test_spatmp.get("transform_features"):
                test_Yhat = test_spatmp.transform_windowed(
                    test_Yhat,
                    [1, 2, 3],
                    util.convert_dates_to_daysofyear(
                        test_spatmp.get("output_windowed_temporal_labels", "test")
                    ) - 1,
                    test_spatmp.get("original_spatial_indices", "test"),
                    test_spatmp.get("response_indices"),
                    test_spatmp,
                    revert=True
                )
                test_Y = test_spatmp.transform_windowed(
                    test_Y,
                    [1, 2, 3],
                    util.convert_dates_to_daysofyear(
                        test_spatmp.get("output_windowed_temporal_labels", "test")
                    ) - 1,
                    test_spatmp.get("original_spatial_indices", "test"),
                    test_spatmp.get("response_indices"),
                    test_spatmp,
                    revert=True
                )
                elapsed = time.time() - start
                print("[%d] -> .....Reverting Transformations..... [%.3f seconds]" % (proc_rank, elapsed))
            if comm_var.get("adjust_predictions"):
                train_Yhat = adjust_predictions(
                    train_Yhat, 
                    train_spatmp, 
                    comm_var.get("prediction_adjustment_map")
                )
            if comm_var.get("adjust_predictions"):
                valid_Yhat = adjust_predictions(
                    valid_Yhat, 
                    valid_spatmp, 
                    comm_var.get("prediction_adjustment_map")
                )
            if comm_var.get("adjust_predictions"):
                test_Yhat = adjust_predictions(
                    test_Yhat, 
                    test_spatmp, 
                    comm_var.get("prediction_adjustment_map")
                )
            # Calculate NRMSE for each subbasin
            train_mins = train_spatmp.filter_axes(
                train_spatmp.get("reduced_minimums"),
                [1, 2],
                [train_spatmp.get("original_spatial_indices", "train"), train_spatmp.get("response_indices")]
            )
            train_maxes = train_spatmp.filter_axes(
                train_spatmp.get("reduced_maximums"),
                [1, 2],
                [train_spatmp.get("original_spatial_indices", "train"), train_spatmp.get("response_indices")]
            )
            train_NRMSEs = util.compute_nrmse(train_Yhat, train_Y, train_mins, train_maxes)
            valid_mins = valid_spatmp.filter_axes(
                valid_spatmp.get("reduced_minimums"),
                [1, 2],
                [valid_spatmp.get("original_spatial_indices", "valid"), valid_spatmp.get("response_indices")]
            )
            valid_maxes = valid_spatmp.filter_axes(
                valid_spatmp.get("reduced_maximums"),
                [1, 2],
                [valid_spatmp.get("original_spatial_indices", "valid"), valid_spatmp.get("response_indices")]
            )
            valid_NRMSEs = util.compute_nrmse(valid_Yhat, valid_Y, valid_mins, valid_maxes)
            test_mins = test_spatmp.filter_axes(
                test_spatmp.get("reduced_minimums"),
                [1, 2],
                [test_spatmp.get("original_spatial_indices", "test"), test_spatmp.get("response_indices")]
            )
            test_maxes = test_spatmp.filter_axes(
                test_spatmp.get("reduced_maximums"),
                [1, 2],
                [test_spatmp.get("original_spatial_indices", "test"), test_spatmp.get("response_indices")]
            )
            test_NRMSEs = util.compute_nrmse(test_Yhat, test_Y, test_mins, test_maxes)
            print("train_NRMSEs =", train_NRMSEs.shape, "=")
            for i in range(train_NRMSEs.shape[0]):
                print(train_spatmp.get("original_spatial_labels", "train")[i], train_NRMSEs[i])
            print("valid_NRMSEs =", valid_NRMSEs.shape, "=")
            for i in range(valid_NRMSEs.shape[0]):
                print(valid_spatmp.get("original_spatial_labels", "valid")[i], valid_NRMSEs[i])
            print("test_NRMSEs =", test_NRMSEs.shape, "=")
            for i in range(test_NRMSEs.shape[0]):
                print(test_spatmp.get("original_spatial_labels", "test")[i], test_NRMSEs[i])
            train_NRMSE = np.mean(train_NRMSEs)
            valid_NRMSE = np.mean(valid_NRMSEs)
            test_NRMSE = np.mean(test_NRMSEs)
            print("NRMSEs -> train=%.4f, valid=%.4f, test=%.4f" % (train_NRMSE, valid_NRMSE, test_NRMSE))
            path = eval_dir + os.sep + eval_var.get("evaluated_checkpoint").replace(".pth", "") + "_Errors.txt"
            error_report = util.curate_error_report(
                [train_NRMSEs, valid_NRMSEs, test_NRMSEs],
                [
                    train_spatmp.get("original_spatial_labels", "train"),
                    valid_spatmp.get("original_spatial_labels", "valid"),
                    test_spatmp.get("original_spatial_labels", "test")
                ],
                train_spatmp.get("response_features"),
                ["train", "valid", "test"],
            )
            with open(path, "w") as f:
                f.write(error_report)
            elapsed = time.time() - start
            eval_range = eval_var.get("evaluation_range")
            print("[%d] -> .....Evaluating Model..... [%.3f seconds]" % (proc_rank, elapsed))
            if plt_var.get("plot_model_fit"):
                print("[%d] -> .....Plotting Training Model Fit....." % (proc_rank))
                start = time.time()
                plt.plot_model_fit(train_Yhat, train_spatmp, "train", eval_range, eval_dir, plt_var)
                elapsed = time.time() - start
                print("[%d] -> .....Plotting Training Model Fit..... [%.3f seconds]" % (proc_rank, elapsed))
                print("[%d] -> .....Plotting Validation Model Fit....." % (proc_rank))
                start = time.time()
                plt.plot_model_fit(valid_Yhat, valid_spatmp, "valid", eval_range, eval_dir, plt_var)
                elapsed = time.time() - start
                print("[%d] -> .....Plotting Training Model Fit..... [%.3f seconds]" % (proc_rank, elapsed))
                print("[%d] -> .....Plotting Testing Model Fit....." % (proc_rank))
                start = time.time()
                plt.plot_model_fit(test_Yhat, test_spatmp, "test", eval_range, eval_dir, plt_var)
                elapsed = time.time() - start
                print("[%d] -> .....Plotting Testing Model Fit..... [%.3f seconds]" % (proc_rank, elapsed))


def adjust_predictions(Yhat, spatmp, prediction_adjustment_map):
    for i in range(spatmp.get("n_responses")):
        feature = spatmp.get("response_features")[i]
        adjustment = prediction_adjustment_map[feature]
        if adjustment[0] == "none":
            Yhat = Yhat
        elif adjustment[0] == "limit":
            a, b = int(adjustment[1]), (None if adjustment[2] == "+" else int(adjustment[2]))
            Yhat[:,:,:,i] = np.clip(Yhat[:,:,:,i], a, b)
        else:
            raise NotImplementedError()
    return Yhat


def data_checkout(data, model, partition, mode):
    def checkout(data, checkout_map, partition):
        checked = []
        for subdata, items in checkout_map.items():
            if data.var_exists(subdata, partition):
                container = data.get(subdata, partition)
            elif data.var_exists(subdata, None):
                container = data.get(subdata, None)
            else:
                raise KeyError("Missing \"%s\" as partitioned or non-partitioned variable in data" % (subdata))
            for item in items:
                if container.var_exists(item, partition):
                    checked += [container.get(item, partition)]
                elif container.var_exists(item, None):
                    checked += [container.get(item, None)]
                else:
                    raise KeyError(
                        "Missing \"%s\" as partitioned or non-partitioned variable in sub-data \"%s\"" % (
                            item, 
                            subdata, 
                        )
                    )
        return checked
    if mode == "train":
        model_checkout_map = {
            "GEOMAN": {
                "spatial": ["original"], 
            }, 
            "GNN": {
                "graph": ["spatial_indices", "edges"], 
            }, 
            "common": {
                "spatiotemporal": ["transformed_input_windowed", "transformed_output_windowed"], 
            }, 
            "extra": {
                "spatiotemporal": [], 
            }
        }
    elif mode == "evaluate":
        model_checkout_map = {
            "GEOMAN": {
                "spatial": ["original"], 
            }, 
            "GNN": {
                "graph": ["spatial_indices", "edges"], 
            }, 
            "common": {
                "spatiotemporal": ["transformed_input_windowed"], 
            }, 
            "extra": {
                "spatiotemporal": ["n_temporal_out"], 
            }
        }
    else:
        raise NotImplementedError()
    checked = []
    checked += checkout(data, model_checkout_map["common"], partition)
    if model.name() in model_checkout_map:
        checked += checkout(data, model_checkout_map[model.name()], partition)
    checked += checkout(data, model_checkout_map["extra"], partition)
    return checked


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


# Precondition(s):
#   spatiotemporal_X.shape = (n_samples, n_temporal_in, n_spatial, n_predictors)
#   spatiotemporal_Y.shape = (n_samples, n_temporal_out, n_spatial, n_responses)
#   spatial.shape = (n_spatial, n_exogenous)
def axes_to_req_layout(data, model):
    pass


if __name__ == "__main__":
    main(sys.argv)
