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
    return parse_vars(contents, con)


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


def parse_eval(contents, con=None):
    if con is None: con = Container()
    return util.from_cache(path)


def parse_info(contents, con=None):
    return parse_vars(contents, con)


def parse_setting(contents, con=None):
    return parse_vars(contents, con)


def parse_vars(contents, con=None):
    if con is None: con = Container()
    lines = contents.split("\n")
    argv = []
    for i in range(len(lines)):
        if lines[i] == "": # empty lines can get added at end depending on OS
            continue
        argv += lines[i].split(" = ")
    ArgumentParser().parse_arguments(argv, con)
    return con


def get_config(path, con=None):
    return get_vars(path, con)


def get_errors(path, con=None):
    if con is None: con = Container()
    if os.path.exists(path):
        with open(path, "r") as f:
            parse_errors(f.read(), con)
    return con


def get_eval(path, con=None):
    return parse_eval(path, con)


def get_infos(path, con=None):
    return get_vars(path, con)


def get_settings(path, con=None):
    return get_vars(path, con)


def get_vars(path, con=None):
    if con is None: con = Container()
    if os.path.exists(path):
        with open(path, "r") as f:
            parse_vars(f.read(), con)
    return con


def get_all_errors(path, con=None, method="efficient"):
    if con is None: con = Container()
    if method == "efficient":
        paths = get_paths(path, "Performance_Checkpoint[Best].txt")
#        paths = get_paths(path, "Performance_Checkpoint[Final].txt")
    else:
        paths = util.get_paths(path, "^Performance_Checkpoint\[.*]\.txt$", recurse=True)
    for path in paths:
        model_name, model_id = path.split(os.sep)[-3:-1]
        if not model_name in con:
            con.set(model_name, Container())
        con.get(model_name).set(model_id, get_errors(path))
    return con


def get_all_evals(path, con=None, method="efficient"):
    if con is None: con = Container()
    if method == "efficient":
        paths = get_paths(path, "Evaluation_Partition[train].txt")
        paths += get_paths(path, "Evaluation_Partition[valid].txt")
        paths += get_paths(path, "Evaluation_Partition[test].txt")
    else:
        paths = util.get_paths(path, "^Evaluation_Partition\[(train|valid|test)]\.pkl", recurse=True)
    for path in paths:
        model_name, model_id = path.split(os.sep)[-3:-1]
        fname = os.path.basename(path)
        if not model_name in con:
            con.set(model_name, Container())
        if not model_id in con.get(model_name):
            con.get(model_name).set(model_id, Container())
        partition = fname.replace("Evaluation_Partition[", "").replace("].pkl", "")
        eval_var = get_eval(path)
        con.get(model_name).get(model_id).set("Yhat", eval_var.Yhat, partition)
    return con


def get_all_infos(path, con=None, method="efficient"):
    if con is None: con = Container()
    if method == "efficient":
        paths = get_paths(path, "ModelInfo.txt")
        paths += get_paths(path, "OptimizationInfo.txt")
    else:
        paths = util.get_paths(path, "^.*Info\.txt$", recurse=True)
    for path in paths:
        model_name, model_id = path.split(os.sep)[-3:-1]
        if not model_name in con:
            con.set(model_name, Container())
        if not model_id in con.get(model_name):
            con.get(model_name).set(model_id, Container())
        con.get(model_name).get(model_id).copy(get_infos(path))
    return con


def get_all_settings(path, con=None, method="efficient"):
    if con is None: con = Container()
    if method == "efficient":
        paths = []
        paths += get_paths(path, "DataSettings.txt")
        paths += get_paths(path, "HyperparameterSettings.txt")
        paths += get_paths(path, "OptimizationSettings.txt")
    else:
        paths = util.get_paths("^.*Settings\.txt", recurse=True)
    for path in paths:
        model_name, model_id = path.split(os.sep)[-3:-1]
        if not model_name in con:
            con.set(model_name, Container())
        if not model_id in con.get(model_name):
            con.get(model_name).set(model_id, Container())
        con.get(model_name).get(model_id).copy(get_settings(path))
    return con

from multiprocessing import Process, Queue
def get_cache(eval_dir, chkpt_dir, con=None, method="efficient"):
    if con is None: con = Container()
    con.errors = get_all_errors(eval_dir, method=method)
    con.settings = get_all_settings(chkpt_dir, method=method)
    con.infos = get_all_infos(chkpt_dir, method=method)
    return con


def get_paths(root_dir, fname):
    paths = []
    model_names = os.listdir(root_dir)
    for model_name in model_names:
        model_dir = os.path.join(root_dir, model_name)
        model_ids = os.listdir(model_dir)
        for model_id in model_ids:
            model_id_dir = os.path.join(model_dir, model_id)
            path = os.path.join(model_id_dir, fname)
            if os.path.exists(path):
                paths.append(path)
    return paths


def find_model_id(cache, model, conditions=None, on_multi="get-choice", return_channel_con=False, return_channel_name=False):
    if conditions is None or conditions == []:
        channel_name, channel_con, _ = cache[0]
        found = channel_con.get(model)
        if len(found) == 0:
            raise ValueError("No model instances found for model=\"%s\"" % (model))
        elif len(found) == 1:
            model_id, var, _ = found[0]
        elif on_multi == "get-choice":
            print("Found multiple model instances for model=\"%s\". Please choose one:" % (model))
            print("\n".join(["%02d : %s" % (i, found[i][0]) for i in range(len(found))]))
            idx = int(input("Choice: "))
            model_id, var, _ = found[idx]
        elif on_multi == "get-first":
            model_id, var, _ = found[0]
        elif on_multi == "get-last":
            model_id, var, _ = found[-1]
        else:
            raise ValueError()
    else:
        if isinstance(conditions, dict):
            found = cache.find(**conditions)
            if len(found) == 0:
                raise ValueError(
                    "Input conditions=%s did not result in any found cache channels" % (str(conditions))
                )
            channel_name, channel_con, _ = found[0]
            found = channel_con.find(**conditions, path=[model])
        else:
            if isinstance(conditions[0], str): # single condition - wrap to treat and multi
                conditions = [conditions]
            names = [condition[0] for condition in conditions]
            comparators = [condition[1] for condition in conditions]
            values = [condition[2] for condition in conditions]
            found = cache.find(names[0], values[0], comparators[0])
            if len(found) == 0:
                raise ValueError(
                    "Input conditions=%s did not result in any found cache channels" % (str(conditions))
                )
            channel_name, channel_con, _ = found[0]
            found = channel_con.find(names, values, comparators, path=[model])
        if len(found) == 0:
            raise ValueError("No model instances found for model=\"%s\" matching conditions=%s" % (model, str(conditions)))
        elif len(found) == 1:
            model_id, var, _ = found[0]
        elif on_multi == "get-choice":
            print("Found multiple model instances for model=\"%s\" matching conditions=%s. Please choose one:" % (model, str(conditions)))
            print("\n".join(["%d : %s" % (i, found[i][0]) for i in range(len(found))]))
            idx = int(input("Choice: "))
            model_id, var, _ = found[idx]
        elif on_multi == "get-first":
            model_id, var, _ = found[0]
        elif on_multi == "get-last":
            model_id, var, _ = found[-1]
        else:
            raise ValueError()
    if return_channel_con and return_channel_name:
        return model_id, channel_con, channel_name
    elif return_channel_con:
        return model_id, channel_con
    elif return_channel_name:
        return model_id, channel_name
    return model_id
