import glob
import os
import sys
import pandas as pd
import numpy as np
import Utility as util
from Plotting import Plotting
from Container import Container
from Arguments import ArgumentParser


def parse_config(contents, con=None):
    if con is None: con = Container()
    lines = contents.split("\n")
    argv = []
    for i in range(len(lines)):
        argv += lines[i].split(" = ")
    ArgumentParser().parse_arguments(argv, con)
    return con

def parse_errors(contents, con=None):
    if con is None: con = Container()
    partition, feature_label, spatial_label, error_name = None, None, None, None
    lines = contents.split("\n")
    for i in range(len(lines)):
        if lines[i].startswith("\t\t"): # Error line for a spatial element
            fields = lines[i].split()
            spatial_label, error_name, error = " ".join(fields[1:-3]), fields[-3], float(fields[-1])
            con.get(error_name, partition).get(feature_label)[spatial_label] = error
        elif lines[i].startswith("\t"): # Error line for a response variable
            fields = lines[i].split()
            feature_label, error_name, error = fields[0], fields[1], float(fields[-1])
            con.get(error_name, partition).set(feature_label, {})
        elif len(lines[i]) > 0 and not lines[i].startswith(("#", " ")): # Error line for a partition
            fields = lines[i].split()
            partition, error_name, error = fields[0], fields[1], float(fields[-1])
            con.set(error_name, Container(), partition)
        else:
            continue
    return con


def parse_info(contents, con=None):
    if con is None: con = Container()
    lines = contents.split("\n")
    for line in lines:
        fields = line.split(":")
        name, value = fields[0].strip(), fields[1].strip()
        if False and "Runtime" in name:
            value = str(round(float(value))) + "s"
        con.set(name, value)
    return con


def get_errors(path, con=None):
    if con is None: con = Container()
    with open(path, "r") as f:
        parse_errors(f.read(), con)
    return con


def get_info(path, con=None):
    if con is None: con = Container()
    with open(path, "r") as f:
        parse_info(f.read(), con)
    return con


def get_eval(path):
    return util.from_cache(path)
    

def get_all_evals(path, con=None):
    if con is None: con = Container()
    eval_paths = util.get_paths(path, "^Evaluation_Partition\[(train|valid|test)]\.pkl", recurse=True)
    for path in eval_paths:
        model_name, config_id = path.split(os.sep)[-3:-1]
        fname = os.path.basename(path)
        if not model_name in con:
            con.set(model_name, Container())
        if not config_id in con.get(model_name):
            con.get(model_name).set(config_id, Container())
        partition = fname.replace("Evaluation_Partition[", "").replace("].pkl", "")
        eval_var = get_eval(path)
        con.get(model_name).get(config_id).set("Yhat", eval_var.Yhat, partition)
    return con


def get_all_configs(path, con=None):
    if con is None: con = Container()
    config_paths = util.get_paths(path, "^Configuration\.txt$", recurse=True)
    for path in config_paths:
        model_name, config_id = path.split(os.sep)[-3:-1]
        with open(path, "r") as f:
            if not model_name in con:
                con.set(model_name, Container())
            con.get(model_name).set(config_id, parse_config(f.read(), Container()))
    return con


def get_all_errors(path, con=None):
    if con is None: con = Container()
    report_paths = util.get_paths(path, "^Performance_Checkpoint\[.*]\.txt$", recurse=True)
    for path in report_paths:
        model_name, config_id = path.split(os.sep)[-3:-1]
        with open(path, "r") as f:
            if not model_name in con:
                con.set(model_name, Container())
            con.get(model_name).set(config_id, parse_errors(f.read(), Container()))
    return con


def get_all_info(path, con=None):
    if con is None: con = Container()
    info_paths = util.get_paths(path, "^.*Info\.txt$", recurse=True)
    for path in info_paths:
        model_name, config_id = path.split(os.sep)[-3:-1]
        with open(path, "r") as f:
            contents = f.read()
        if not con.var_exists(model_name):
            con.set(model_name, Container())
        if not con.get(model_name).var_exists(config_id):
            con.get(model_name).set(config_id, Container())
        info_con = parse_info(contents)
        con.get(model_name).get(config_id).copy(info_con)
    return con


def multimodel_errors_to_table(err_con, partition="test", metric="NRMSE"):
    data_map = {}
    models = list(err_con.get_keys())
    for model in models:
        errs = err_con.get(model).get(metric, partition, recurse=True)
        features = list(errs.get_keys())
        for feature in features:
            feature_errs = errs.get(feature)
            if "MT_178" in feature_errs:
                del feature_errs["MT_178"]
            test_err = np.mean(list(feature_errs.values()))
        data_map[model] = [test_err]
    return pd.DataFrame(data_map)


def multimodel_info_to_table(info_con):
    data_map = {}
    models = list(info_con.get_keys())
    for model in models:
        config_id = list(info_con.get(model).get_keys())[0]
        key_values = info_con.get(model).get(config_id).get_key_values()
        info_names = [key for key, value in key_values]
        info_values = [value for key, value in key_values]
        data_map[model] = info_values
    return pd.DataFrame(data_map, index=info_names)
