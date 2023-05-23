import torch
import math
import numpy as np
import datetime as dt
import pickle
import pandas as pd
import time
import signal
import sys
import hashlib
import networkx as nx
import json
import re
import os
import inspect
import scipy


comparator_fn_map = {
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b, 
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b, 
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b, 
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b, 
}


operator_fn_map = {
    "+": lambda a, b: a + b, 
    "-": lambda a, b: a - b, 
    "*": lambda a, b: a * b, 
    "/": lambda a, b: a / b, 
    "**": lambda a, b: a ** b, 
    "//": lambda a, b: a // b, 
    "%": lambda a, b: a % b, 
}


month_name_map = {
    1: "January", 
    2: "February", 
    3: "March", 
    4: "April", 
    5: "May", 
    6: "June", 
    7: "July", 
    8: "August", 
    9: "September", 
    10: "October", 
    11: "November", 
    12: "December"
}


class TimeoutExpired(Exception):
    pass


def input_with_timeout(prompt, timeout, timer=time.monotonic):
    import msvcrt
    # Source : Alex Martelli @ https://stackoverflow.com/questions/2933399/how-to-set-time-limit-on-raw-input/2933423#2933423
    sys.stdout.write(prompt)
    sys.stdout.flush()
    endtime = timer() + timeout
    result = []
    while timer() < endtime:
        if msvcrt.kbhit():
            result.append(msvcrt.getwche()) #XXX can it block on multibyte characters?
            if result[-1] == '\r':
                return ''.join(result[:-1])
        time.sleep(0.04) # just to yield to other processes/threads
    raise TimeoutExpired


# Description:
#   Converts between naming convetions
# Arguments:
#   name - the name to convert
#   orig_conv - convention used for name
#   targ_conv - convention to convert name into
def convert_name_convention(name, orig_conv, targ_conv):
    name_fields, new_name = [], ""
    # parse into fields
    if orig_conv == "Pascal":
        name_fields = re.findall("[a-zA-Z][^A-Z]*", name)
    elif orig_conv == "camel":
        name_fields = re.findall("[a-zA-Z][^A-Z]*", name)
    elif orig_conv == "snake":
        name_fields = name.split("_")
    elif orig_conv == "kebab":
        name_fields = name.split("-")
    else:
        raise NotImplementedError("Unknown naming convertion \"%s\"" % (orig_conv))
    # join fields into new name
    if targ_conv == "Pascal":
        new_name = "".join([name_field[0].upper() + name_field[1:] for name_field in name_fields])
    elif targ_conv == "camel":
        new_name = "".join([name_field[0].upper() + name_field[1:] for name_field in name_fields])
        new_name = new_name[0].lower() + new_name[1:]
    elif targ_conv == "snake":
        new_name = "_".join(name_fields)
    elif targ_conv == "kebab":
        new_name = "-".join(name_fields)
    else:
        raise NotImplementedError("Unknown naming convertion \"%s\"" % (targ_conv))
    return new_name


def get_paths(search_dir, path_regex, recurse=False, debug=False):
    if not os.path.exists(search_dir):
        raise FileNotFoundError(search_dir)
    paths = []
    for root, dirnames, filenames in os.walk(search_dir):
        for filename in filenames:
            if debug:
                print(path_regex, root, filename)
            if re.match(path_regex, filename):
                paths += [os.path.join(root, filename)]
        if not recurse:
            break
    return paths


class Types:

    def is_anything(item):
        return True

    def is_none(item):
        return item is None

    def is_numeric(item):
        return isinstance(item, int) or isinstance(item, float)

    def is_string(item):
        return isinstance(item, str)

    def is_list(item):
        return isinstance(item, list)

    def is_list_of_strings(item):
        if isinstance(item, list) and len(item) > 0:
            return all(isinstance(i, str) for i in item)
        return False


def get_func_args(func):
    args, var_args, kw_args, def_vals = inspect.getargspec(func)
    if "self" in args:
        args.remove("self")
    req_args = args[:-len(def_vals)]
    def_args = dict(zip(list_subtract(list(args), req_args), def_vals))
    return req_args, var_args, kw_args, def_args


def get_interval_from_selection(selection):
    mode = selection[0]
    implemented_modes = ["interval", "range", "literal"]
    if mode not in implemented_modes:
        raise NotImplementedError("Given mode \"%s\" but only modes \"%s\" are implemented" % (
            mode,
            ",".join(implemented_modes))
        )
    if mode == "interval":
        interval = selection[1:]
    elif mode == "range":
        interval = selection[1:2]
    elif mode == "literal":
        sorted_selection = sorted(selection[1:])
        interval = [sorted_selection[1], sorted_selection[-1]]
    return interval


def compute_zscore_classes(vals, means, stddevs, z_intervals):
    classes = np.zeros(vals.shape)
    for i in range(len(z_intervals)):
        min_vals, max_vals = means + z_intervals[i][0] * stddevs, means + z_intervals[i][1] * stddevs
        if z_intervals[i][0] < -1:
            lower_mask, upper_mask = vals >= min_vals, vals < max_vals
        elif z_intervals[i][0] == -1 and z_intervals[i][1] == 1:
            lower_mask, upper_mask = vals >= min_vals, vals <= max_vals
        else:
            lower_mask, upper_mask = vals > min_vals, vals <= max_vals
        """
        print("Interval =", z_intervals[i])
        print("min/max =", min_vals[:3], max_vals[:3])
        print("vals =", vals[:3])
        print("lower/upper mask =", lower_mask[:3], upper_mask[:3])
        """
        indices = np.where(np.logical_and(lower_mask, upper_mask))[0]
        if indices.shape[0] > 0:
            classes[indices] = i
    return classes


def compute_zscore_confusion(preds, gts, means, stddevs, normalize=False):
    z_intervals = [[-8, -2], [-2, -1.5], [-1.5, -1], [-1, 1], [1, 1.5], [1.5, 2], [2, 8]]
    pred_classes = compute_zscore_classes(preds, means, stddevs, z_intervals)
    gt_classes = compute_zscore_classes(gts, means, stddevs, z_intervals)
    from sklearn.metrics import confusion_matrix
    normalization = ("true" if normalize else None)
    confusion = confusion_matrix(gt_classes, pred_classes, labels=np.arange(len(z_intervals)), normalize=normalization)
    return confusion


def compute_events(values, means, stddevs):
    intervals = [[-8, -2], [-2, -1.5], [-1.5, -1], [-1, 1], [1, 1.5], [1.5, 2], [2, 8]]
    interval_events_map = {}
    for interval in intervals:
        lower_bounds = means + interval[0] * stddevs
        upper_bounds = means + interval[1] * stddevs
        if interval[0] < 0 and interval[1] < 0:# Below mean
            if interval[0] == -8:# Extreme => (-8,-2)
                extremes_below = np.ma.masked_where(
                    np.logical_not(values < upper_bounds),
                    values
                )
                events = extremes_below
            elif interval[0] == -2:# Severe => [-2,-1.5)
                severes_below = np.ma.masked_where(
                    np.logical_not(
                        np.logical_and(
                            values >= lower_bounds,
                            values < upper_bounds
                        )
                    ),
                    values
                )
                events = severes_below
            elif interval[0] == -1.5:# Moderate
                moderates_below = np.ma.masked_where(
                    np.logical_not(
                        np.logical_and(
                            values >= lower_bounds,
                            values < upper_bounds
                        )
                    ),
                    values
                )
                events = moderates_below
        elif interval[0] > 0 and interval[1] > 0:# Above mean
            if interval[1] == 8:# Extreme
                extremes_above = np.ma.masked_where(
                    np.logical_not(values > lower_bounds),
                    values
                )
                events = extremes_above
            elif interval[1] == 2:# Severe
                severes_above = np.ma.masked_where(
                    np.logical_not(
                        np.logical_and(
                            values > lower_bounds,
                            values <= upper_bounds
                        )
                    ),
                    values
                )
                events = severes_above
            elif interval[1] == 1.5:# Moderate
                moderates_above = np.ma.masked_where(
                    np.logical_not(
                        np.logical_and(
                            values > lower_bounds,
                            values <= upper_bounds
                        )
                    ),
                    values
                )
                events = moderates_above
        else:
            normals = np.ma.masked_where(
                np.logical_not(
                    np.logical_and(
                        values >= lower_bounds,
                        values <= upper_bounds
                    )
                ),
                values
            )
            events = normals
        interval_events_map[",".join(map(str, interval))] = events
    return interval_events_map


def sort_dict(a, by="key", ascending=True):
    if by == "value":
        return dict(sorted(a.items(), key=lambda item: item[1], reverse=(not ascending)))
    if by == "key":
        b = {}
        keys = sorted(list(a.keys()))
        for key in keys:
            b[key] = a[key]
    return b


def get_stats(a):
    percentiles = np.percentile(a, [25, 50, 75])
    stat_value_map = {
        "count": a.shape[0],
        "mean": np.mean(a),
        "std": np.std(a),
        "min": np.min(a),
        "25%": percentiles[0],
        "50%": percentiles[1],
        "75%": percentiles[2],
        "max": np.max(a)
    }
    return stat_value_map


def get_least_similar(similarities, src, k):
    G = nx.convert_matrix.from_numpy_matrix(similarities)
    distances, paths = nx.algorithms.shortest_paths.weighted.single_source_dijkstra(G, source=src, cutoff=k)
    least_similar = {}
    for target, path in paths.items():
        if len(path) == k:
            path_str = ",".join(map(str, path))
            least_similar[path_str] = distances[target]
    return dict(sorted(least_similar.items(), key=lambda item: item[1]))


def format_memory(n_bytes):
    if n_bytes == 0:
        return "0B"
    mem_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(n_bytes, 1024)))
    p = math.pow(1024, i)
    s = int(round(n_bytes / p, 2))
    return "%s%s" % (s, mem_name[i])


def memory_of(item):
    if isinstance(item, torch.Tensor):
        n_bytes = item.element_size() * item.nelement()
    else:
        n_bytes = sys.getsizeof(item)
    return format_memory(n_bytes)


# Creates indices that index a set of windows
#   timesteps: number of time-steps in source
#   length: window length => time-steps per window
#   stride: stride length => time-steps between window origins
#   offset: offset length => time-step where first window begins
def sliding_window_indices(timesteps, length, stride=1, offset=0):
    n = (timesteps - length + 1 - offset) // stride
    return np.tile(np.arange(length), (n, 1)) + stride * np.reshape(np.arange(n), (-1, 1)) + offset


def input_output_window_indices(timesteps, in_length, out_length, horizon=1, stride=1, offset=0):
    """ Creates indices that index a set of input and output windows
    Arguments
    ---------
    timesteps : int
        number of time-steps in source
    in_length : int
        number of time-steps per input window
    out_length : int
        number of time-steps per output window
    horizon : int
        offset (in time-steps) of output windows relative to last time-step of each input window
        see notes below
    stride : int
        number of time-steps between window origins
    offset : int
        time-step where first window begins

    Notes
    -----
    horizon :
        at horizon=1  => input/output window indices are [0, 1, 2]/[3, 4, 5]
        at horizon=3  => input/output window indices are [0, 1, 2]/[5, 6, 7]
        at horizon=-2 => input/output window indices are [0, 1, 2]/[0, 1, 2]
    """
    indices = sliding_window_indices(timesteps, in_length + (horizon - 1) + out_length, stride, offset)
    return indices[:,:in_length], indices[:,-out_length:]


def contiguous_window_indices(n, length, stride=1, offset=0):
    assert length % stride == 0, "Windows cannot be contiguous with stride != 1"
    return np.arange(offset, n, length // stride)


# Calculate mean absolute error (MAE) for each spatial element and feature
# Pre-conditions:
#   Y.shape=(n_sample, n_temporal, n_spatial, n_feature)
#   Yhat.shape=(n_sample, n_temporal, n_spatial, n_feature)
# Post-conditions:
#   MAE.shape=(n_spatial, n_feature)
def MAE(Y, Yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", (0, 1))
    mask = kwargs.get("mask", None)
    # Compute
    if not mask is None:
        Yhat = np.copy(Yhat)
        Yhat[~mask] = Y[~mask]
    return np.mean(np.abs(Y - Yhat), axis)


# Calculate mean square error (MSE) for each spatial element and feature
# Pre-conditions:
#   Y.shape=(n_sample, n_temporal, n_spatial, n_feature)
#   Yhat.shape=(n_sample, n_temporal, n_spatial, n_feature)
# Post-conditions:
#   MSE.shape=(n_spatial, n_feature)
def MSE(Y, Yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", (0, 1))
    mask = kwargs.get("mask", None)
    # Compute
    if not mask is None:
        Yhat = np.copy(Yhat)
        Yhat[~mask] = Y[~mask]
    return np.mean((Y - Yhat)**2, axis)


# Calculate root mean square error (RMSE) for each spatial element and feature
# Pre-conditions:
#   Y.shape=(n_sample, n_temporal, n_spatial, n_feature)
#   Yhat.shape=(n_sample, n_temporal, n_spatial, n_feature)
# Post-conditions:
#   MAPE.shape=(n_spatial, n_feature)
def MAPE(Y, Yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", (0, 1))
    eps = kwargs.get("eps", np.finfo(np.float64).eps)
    mask = kwargs.get("mask", None)
    # Compute
    if not mask is None:
        Yhat = np.copy(Yhat)
        Yhat[~mask] = Y[~mask]
    if 0:
        print((Y - Yhat).shape)
        print(np.abs(Y - Yhat).shape)
        print(np.maximum(np.abs(Y), eps).shape)
#        print(np.abs((Y - Yhat) / np.maximum(np.abs(Y), eps)))
        print(np.max(np.abs((Y - Yhat) / np.maximum(np.abs(Y), eps))))
        input()
    return np.mean(np.minimum(np.abs((Y - Yhat) / np.maximum(np.abs(Y), eps)), 100), axis)
    return np.mean(np.abs((Y - Yhat) / np.maximum(np.abs(Y), eps)), axis)


# Calculate root mean square error (RMSE) for each spatial element and feature
# Pre-conditions:
#   Y.shape=(n_sample, n_temporal, n_spatial, n_feature)
#   Yhat.shape=(n_sample, n_temporal, n_spatial, n_feature)
# Post-conditions:
#   RMSE.shape=(n_spatial, n_feature)
def RMSE(Y, Yhat, **kwargs):
    # Unpack args
    mask = kwargs.get("mask", None)
    # Compute
    if not mask is None:
        Yhat = np.copy(Yhat)
        Yhat[~mask] = Y[~mask]
    return np.sqrt(MSE(Y, Yhat, **kwargs))


# Calculate normalized root mean square error (NRMSE) for each spatial element and feature
# Preconditions:
#   Y.shape=(n_sample, n_temporal, n_spatial, n_feature)
#   Yhat.shape=(n_sample, n_temporal, n_spatial, n_feature)
#   mins.shape=(period_size, n_spatial, n_feature)
#   maxes.shape=(period_size, n_spatial, n_feature)
# Post-conditions:
#   NRMSE.shape=(n_spatial, n_feature)
def NRMSE(Y, Yhat, **kwargs):
    # Unpack args
    mins, maxes = kwargs["mins"], kwargs["maxes"]
    axis = kwargs.get("axis", (0, 1))
    eps = kwargs.get("eps", np.finfo(np.float64).eps)
    mask = kwargs.get("mask", None)
    # Setup
    if 0:
        mins = np.min(mins, axis=0)
        maxes = np.max(maxes, axis=0)
    # Compute
    if not mask is None:
        Yhat = np.copy(Yhat)
        Yhat[~mask] = Y[~mask]
    return RMSE(Y, Yhat, **kwargs) / np.maximum(maxes - mins, eps)


def UPR(Y, Yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", (0, 1))
    margin = kwargs.get("margin", 5/100)
    mask = kwargs.get("mask", None)
    # Compute
    if not mask is None:
        Yhat = np.copy(Yhat)
        Yhat[~mask] = Y[~mask]
    up_count = np.sum(Yhat < ((1 - margin) * Y), axis)
    N = np.prod(np.take(Y.shape, axis))
    return up_count / N


def OPR(Y, Yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", (0, 1))
    margin = kwargs.get("margin", 5/100)
    mask = kwargs.get("mask", None)
    # Compute
    if not mask is None:
        Yhat = np.copy(Yhat)
        Yhat[~mask] = Y[~mask]
    op_count = np.sum(Yhat > ((1 + margin) * Y), axis)
    N = np.prod(np.take(Y.shape, axis))
    return op_count / N


def MR(Y, Yhat, **kwargs):
    # Compute
    return UPR(Y, Yhat, **kwargs) + OPR(Y, Yhat, **kwargs)


def RRSE(Y, Yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", (0, 1))
    means = kwargs["means"]
    eps = kwargs.get("eps", np.finfo(np.float64).eps)
    # Setup
    if 0:
        means = np.mean(means, axis=0)[None,None,:,:]
    else:
        means = means[None,None,:,:]
    # Compute
    return np.sqrt(np.sum((Y - Yhat)**2, axis) / np.sum((Y - means)**2, axis))


def CORR(Y, Yhat, **kwargs):
    axis = kwargs.get("axis", (0, 1))
    means = kwargs["means"]
    eps = kwargs.get("eps", np.finfo(np.float64).eps)
    # Setup
    M = means[None,None,:,:]
    # Compute
    return np.sum((Yhat - M) * (Y - M), axis) / (np.sqrt(np.sum((Yhat - M)**2, axis)) * np.sqrt(np.sum((Y - M)**2, axis)))


# Preconditions:
#   errors: a list of numpy arrays with shape=(n_spatial, n_feature)
def curate_performance_report(Ys, Yhats, partitions, spatial_labels, spatial_names, feature_labels, metrics, metric_kwargs):
    if not isinstance(Ys, list) or not isinstance(Yhats, list) or not isinstance(partitions, list):
        raise ValueError("Ground-truths, predictions, and partitions must be lists (Ys, Yhats, partitions).")
    if not (len(Ys) == len(Yhats) == len(partitions)):
        raise ValueError("Ground-truths, predictions, and partitions must be equal length.")
    metric_func_map = {
        "MAE": MAE,
        "MSE": MSE,
        "MAPE": MAPE,
        "RMSE": RMSE,
        "NRMSE": NRMSE,
        "UPR": UPR,
        "OPR": OPR,
        "MR": MR, 
        "RRSE": RRSE, 
        "CORR": CORR, 
    }
    if metrics == "*":
        metrics = list(metric_func_map.keys())
    lines = []
    for metric in metrics:
        for i in range(len(partitions)):
            Y, Yhat, partition = Ys[i], Yhats[i], partitions[i]
            performances = metric_func_map[metric](Y, Yhat, **metric_kwargs[i])
            lines += [
                "%s %s = %.4f" % (
                    partition,
                    metric,
                    np.mean(performances)
                )
            ]
            for j in range(len(feature_labels)):
                lines += [
                    "\t%s %s = %.4f" % (
                        feature_labels[j],
                        metric,
                        np.mean(performances[:,j])
                    )
                ]
                for k in range(len(spatial_labels[i])):
                    lines += [
                        "\t\t%s %4s %s = %.4f" % (
                            spatial_names[i],
                            spatial_labels[i][k],
                            metric,
                            performances[k,j]
                        )
                    ]
    return "\n".join(lines)


# Construct a - b
def list_subtract(a, b):
    if b is None:
        return []
    return [a_i for a_i in a if a_i not in b]


def list_indices(a, items):
    indices = []
    for item in items:
        indices.append(a.index(item))
    return indices


def dict_to_str(a):
    return json.dumps(a, sort_keys=True, indent=4)


def merge_dicts(a, b, overwrite=True):
    if not (isinstance(a, dict) or isinstance(b, dict)):
        raise ValueError("Expected two dictionaries")
    if overwrite:
        return {**a, **b}
    c = copy_dict(a)
    for key, value in b.items():
        if not key in a:
            c[key] = value
    return c


def remap_dict(a, remap_dict, must_exist=False):
    b = copy_dict(a)
    for old_key, new_key in remap_dict.items():
        if old_key in b:
            b[new_key] = b.pop(old_key)
        elif must_exist:
            raise ValueError("Input dict a does not contain key %s" % (str(old_key)))
    return b


def copy_dict(a):
    return {key: value for key, value in a.items()}


def to_key_index_dict(keys, offset=0, stride=1):
    return {key: offset+i for key, i in zip(keys, np.arange(0, len(keys), stride))}


def to_dict(keys, values, repeat=False):
    if repeat:
        return {key: values for key in keys}
    return {key: value for key, value in zip(keys, values)}


def invert_dict(a):
    return {value: key for key, value in a.items()}


def sort_dict(a, by="key"):
    if by == "key":
        return {key: a[key] for key in sorted(a.keys(), key=str.lower)}
    elif by == "value":
        return invert_dict(sort_dict(invert_dict(a)))
    raise ValueError("Unknown sorting option \"%s\"" % (by))


def get_dict_values(a, keys, must_exist=True):
    values = []
    for key in keys:
        if not key in a and not must_exist:
            continue
        values.append(a[key])
#    if isinstance(keys, np.ndarray):
#        values = np.array(values)
    return values


def filter_dict(a, keys, must_exist=False):
    b = {}
    for key in keys:
        if not key in a and not must_exist:
            continue
        b[key] = a[key]
    return b


def to_cache(data, path):
    if os.sep in path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)


def from_cache(path, **kwargs):
    with open(path, "rb") as f:
        data = pickle.load(f, **kwargs)
    return data


def hash_str_to_int(string, n_digits):
    num = int(hashlib.sha256(string.encode("utf-8")).hexdigest(), 16) % 10**n_digits
    cur_digits = len(str(num))
    return num * 10**(n_digits - cur_digits)


def make_msg_block(msg, block_char="#"):
    msg_line = 3*block_char + " " + msg + " " + 3*block_char
    msg_line_len = len(msg_line)
    msg_block = "%s\n%s\n%s" % (
        msg_line_len*block_char,
        msg_line,
        msg_line_len*block_char
    )
    return msg_block


def get_device(use_gpu):
    return torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")


def to_device(items, device):
    if isinstance(items, list): # Multiple items to put onto device
        device_items = []
        for item in items:
            if isinstance(item, torch.Tensor) or isinstance(item, torch.nn.Module):
                device_items += [item.to(device)]
            else:
                device_items += [item]
        return device_items
    else: # Single item to put onto device
        return to_device([items], device)[0]


def to_tensor(items, types):
    if not isinstance(items, list): # Single item
        return to_tensor([items], types)[0]
    # Convert each data in items to a tensor
    tensors = []
    for i in range(len(items)):
        item = items[i]
        if isinstance(types, list):
            _type = types[i]
        else:
            _type = types
        if _type is None: # ignore
            tensors.append(item)
            continue
        if isinstance(item, torch.Tensor): # No need to convert
            tensors.append(item)
        else:
            tensors.append(torch.tensor(item, dtype=_type))
    return tensors


def to_ndarray(items):
    if not isinstance(items, list): # Single item
        return to_ndarray([items])[0]
    # Convert each data in items to a tensor
    ndarrays = []
    for item in items:
        if isinstance(item, torch.Tensor):
            ndarrays += [item.detach().cpu().numpy()]
        elif isinstance(item, np.ndarray):
            ndarrays += [item]
        elif isinstance(item, list):
            ndarrays += [np.array(item)]
        else:
            raise NotImplementedError(
                "Do not know how to convert %s of type %s to a NumPy.ndarray" % (
                    item,
                    type(item)
                )
            )
    return ndarrays


# moveaxis(a, [0, 1, 2], [1, 2, 0]) means we move axis 0 to 1, axis 1 to 2, and axis 2 to 0
def move_axes(items, sources, destinations):
    if not isinstance(sources, list) or not isinstance(destinations, list):
        raise ValueError("Sources and destinations must be type list")
    if isinstance(items, list): # Multiple items to move_axes on
        if isinstance(items[0], np.ndarray):
            if isinstance(sources[0], list) and isinstance(destinations[0], list):
                return [np.moveaxis(item, src, dst) for item, src, dst in zip(items, sources, destinations)]
            else:
                return [np.moveaxis(item, sources, destinations) for item in items]
        elif isinstance(items[0], torch.Tensor):
            if isinstance(sources[0], list) and isinstance(destinations[0], list):
                return [torch.movedim(item, src, dst) for item, src, dst in zip(items, sources, destinations)]
            else:
                return [torch.movedim(item, sources, destinations) for item in items]
        else:
            raise NotImplementedError("Only numpy.ndarray and torch.Tensor types are implemented")
    else: # Single item to move_axes on
        return move_axes([items], sources, destinations)[0]


def minmax_transform(A, minimum, maximum, **kwargs):
    a = kwargs.get("a", -1)
    b = kwargs.get("b", 1)
    clip = kwargs.get("clip", False)
    revert = kwargs.get("revert", False)
    if revert:
        if clip:
            A = np.clip(A, a, b)
        A = ((A - a) / (b - a)) * (maximum - minimum) + minimum
    else:
        A = (b - a) * ((A - minimum) / (maximum - minimum)) + a
        if clip:
            A = np.clip(A, a, b)
    return A


def zscore_transform(A, mean, std, **kwargs):
    revert = kwargs.get("revert", False)
    if isinstance(std, np.ndarray) or isinstance(std, torch.Tensor):
        if std.shape == ():
            if std == 0:
                std = 1.0
        else:
            std[std == 0] = 1.0
    elif isinstance(std, float) or isinstance(std, int):
        if std == 0:
            std = 1
    if revert:
        return A * std + mean
    return (A - mean) / std


def log_transform(A, _1, _2, **kwargs):
    revert = kwargs.get("revert", False)
    if revert:
        return np.exp(A)
    A[A == 0] = 1 + 1e-5
    return np.log(A)


def root_transform(A, _1, _2, **kwargs):
    p = kwargs.get("p", 2)
    revert = kwargs.get("revert", False)
    if revert:
        return A**p
    return A**(1/p)


def identity_transform(A, _1, _2, **kwargs):
    return A


def transform(args, transform, revert=False):
    transform_fn_map = {
        "minmax": minmax_transform, 
        "zscore": zscore_transform, 
        "log": log_transform, 
        "root": root_transform, 
        "identity": identity_transform, 
    }
    name, kwargs = transform, {"revert": revert} # case where transform is str and no kwargs
    if isinstance(transform, dict): # case where transform is dict and specifies kwargs
        _transform = copy_dict(transform)
        name = _transform.pop("name")
        kwargs = _transform
        kwargs["revert"] = revert
    return transform_fn_map[name](*args, **kwargs)


class WabashRiverSubbasinSoil:

    def __init__(self, item, debug=False):
        lines = item.split("\n")
        header = lines[0]
        if "Area [ha]" in header:
            header = header.replace("Area [ha]", "area_ha")
        if "Area[acres]" in header:
            header = header.replace("Area[acres]", "area_acres")
        if "%Wat.Area" in header:
            header = header.replace("%Wat.Area", "watershed_area")
        if "%Sub.Area" in header:
            header = header.replace("%Sub.Area", "subbasin_area")
        self.features = header.split()
        self.data = {}
        i = 0
        while i < len(lines):
            fields = lines[i].split()
            if debug:
                print("FIELDS =", fields)
            if len(fields) < 1:
                i += 1
                continue
            if fields[0] == "SUBBASIN":
                self.data["subbasin"] = fields[2]
            if fields[0] == "LANDUSE:":
                j, k = i + 1, 0
                while j < len(lines):
                    fields = lines[j].split("-->")
                    if debug:
                        print("FIELDS =", fields)
                    if len(fields) > 0 and fields[0] == "SOILS:":
                        break
                    if len(fields) > 1:
                        fields = fields[1].split()
                        for _feature, field in zip(["type"]+self.features, fields):
                            self.data["landuse"+str(k+1)+"_"+_feature] = field
                        k += 1
                    j += 1
                i = j - 1
            if fields[0] == "SOILS:":
                j, k = i + 1, 0
                while j < len(lines):
                    fields = lines[j].split()
                    if debug:
                        print("FIELDS =", fields)
                    if len(fields) > 0 and fields[0] == "SLOPE:":
                        break
                    if len(fields) > 1:
                        for _feature, field in zip(["type"]+self.features, fields):
                            self.data["soil"+str(k+1)+"_"+_feature] = field
                        k += 1
                    j += 1
                i = j - 1
            i += 1

    def __str__(self):
        lines = []
        for key, value in self.__dict__.items():
            lines.append("%s : %s" % (str(key), str(value)))
        return "\n".join(lines)


def convert_wabash_river_spatial():
    import re
    data_dir = os.sep.join(["Data", "WabashRiver"])
    # Convert soil data
    #   Create soil objects
    path = os.sep.join([data_dir, "SpatialData_Type[Elevation,LandUse]", "text", "HRULandUseSoilsReport.txt"])
    with open(path, "r") as f:
        content = f.read()
    items = content.split(122*"_"+"\n")
    max_features = -1
    soils = []
    for i in range(5, len(items), 2):
        soil = WabashRiverSubbasinSoil(items[i])
        soils.append(soil)
        if 0:
            print(122*"_")
            print(soil)
        if len(soil.data.keys()) > max_features:
            max_features = len(soil.data.keys())
            feature_superset = list(soil.data.keys())
    #   Get feature labels
    features = []
    remove = [".*area_ha.*", ".*area_acres.*", ".*watershed_area.*"]
    for feature in feature_superset:
        matches = [not re.match(rem, feature) is None for rem in remove]
        if not any(matches):
            features.append(feature)
    #   Create dataframe
    print(features)
    data = {
        "subbasin": [],
        "landuse_types": [],
        "landuse_proportions": [],
        "soil_types": [],
        "soil_proportions": []
    }
    for i in range(len(soils)):
        soil = soils[i]
        print(soil)
        landuse_types, landuse_proportions = [], []
        soil_types, soil_proportions = [], []
        for feature in soil.data.keys():
            if feature in data:
                data[feature].append(soil.data[feature])
            elif re.match("landuse\d_type", feature):
                landuse_types.append(soil.data[feature])
            elif re.match("landuse\d_subbasin_area", feature):
                landuse_proportions.append(soil.data[feature])
            elif re.match("soil\d_type", feature):
                soil_types.append(soil.data[feature])
            elif re.match("soil\d_subbasin_area", feature):
                soil_proportions.append(soil.data[feature])
        data["landuse_types"].append(";".join(landuse_types))
        data["landuse_proportions"].append(";".join(landuse_proportions))
        data["soil_types"].append(";".join(soil_types))
        data["soil_proportions"].append(";".join(soil_proportions))
    soil_df = pd.DataFrame.from_dict(data)
    print(soil_df)
    # Convert meta data
    path = os.sep.join([data_dir, "WabashSubbasin_AreaLatLon.csv"])
    df = pd.read_csv(path)
    #   Edit feature labels
    columns = [col.lower() for col in df.columns]
    columns[columns.index("lat_1")] = "lat"
    columns[columns.index("long_1")] = "lon"
    df.columns = columns
    #   Get feature labels
    features = []
    remove = [".*fid.*", ".*gridcode.*"]
    for feature in df.columns:
        matches = [not re.match(rem, feature) is None for rem in remove]
        if not any(matches):
            features.append(feature)
    #   Create dataframe
    meta_df = df[features]
    print(meta_df)
    # Convert elevation data
    path = os.sep.join([data_dir, "WabashSubbasinEvalationStatistics.csv"])
    df = pd.read_csv(path)
    #   Edit feature labels
    columns = ["elevation_"+col.lower() for col in df.columns]
    columns[columns.index("elevation_subbasin")] = "subbasin"
    df.columns = columns
    #   Get feature labels
    features = []
    remove = [".*rowid.*", ".*count.*", ".*area.*", ".*range.*", ".*sum.*"]
    for feature in df.columns:
        matches = [not re.match(rem, feature) is None for rem in remove]
        if not any(matches):
            features.append(feature)
    #   Create dataframe
    elev_df = df[features]
    print(elev_df)
    # Convert river data
    path = os.sep.join([data_dir, "WabashSubbasinRivers.csv"])
    df = pd.read_csv(path)
    #   Edit feature labels
    columns = ["river_"+col.lower() for col in df.columns]
    columns[columns.index("river_subbasin")] = "subbasin"
    columns[columns.index("river_len2")] = "river_length"
    columns[columns.index("river_slo2")] = "river_slope"
    columns[columns.index("river_wid2")] = "river_width"
    columns[columns.index("river_dep2")] = "river_depth"
    df.columns = columns
    #   Get feature labels
    features = []
    remove = [".*fid.*", ".*objectid.*", ".*arcid.*", ".*grid_code.*", ".*from_node.*", ".*to_node.*", ".*subbasinr.*", ".*areac.*", ".*minel.*", ".*maxel.*", ".*shape_leng.*", ".*hydroid.*", ".*outletid.*"]
    for feature in df.columns:
        matches = [not re.match(rem, feature) is None for rem in remove]
        if not any(matches):
            features.append(feature)
    #   Create dataframe
    river_df = df[features]
    print(river_df)
    # Join all dataframes and save
    dfs = [soil_df, meta_df, elev_df, river_df]
    df = dfs[0]
    df["subbasin"] = df["subbasin"].astype(int)
    for _df in dfs[1:]:
        _df["subbasin"] = _df["subbasin"].astype(int)
        df = df.merge(_df, "left", "subbasin")
    print(df)
    path = os.sep.join([data_dir, "Spatial.csv"])
    df.to_csv(path, index=False)


def resolution_to_delta(res):
    return dt.timedelta(**{res[1]: res[0]})


def days_in_month(year, month):
    return (dt.datetime(year + month // 12, month % 12 + 1, 1) - dt.datetime(year, month, 1)).days


def datetime_range(start, end, delta):
    datetimes = []
    curr = start
    while curr <= end:
        datetimes.append(curr)
        curr += delta
    return datetimes


def generate_temporal_labels(start, end, delta, frmt="%Y-%m-%d_%H-%M-%S", incl=[True, False]):
    """ Generates a chronologically ordered set of string-formatted time-stamps for a given range

    Arguments
    ---------
    start : str or datetime
    end : str, int, or datetime
    delta : list of [int, str] or timedelta
    frmt : str
    incl : list of [bool, bool]

    Returns
    -------
    temporal_labels : list of str

    """
    # Check arguments
    if not (isinstance(start, str) or isinstance(start, dt.datetime)):
        raise ValueError("Argument \"start\" must be str or datetime. Received %s" % (type(start)))
    if not (isinstance(end, str) or isinstance(end, int) or isinstance(end, dt.datetime)):
        raise ValueError("Argument \"end\" must be str, int, or datetime. Received %s" % (type(end)))
    if not (isinstance(delta, list) or isinstance(delta, dt.timedelta)):
        raise ValueError("Argument \"delta\" must be list or timedelta. Received %s" % (type(delta)))
    elif isinstance(delta, list): # delta given as custom format [int, str] (e.g. [7, "days"])
        if not (isinstance(delta[0], int) and isinstance(delta[1], str)):
            raise ValueError("Argument \"delta\" as list must follow format [int, str]. Received %s" % (str(delta)))
        elif delta[0] < 1:
            raise ValueError("Number of time-steps in delta=[int str] must be positive. Received %d" % (delta[0]))
        elif delta[1] not in ["seconds", "minutes", "hours", "days", "weeks", "months", "years"]:
            raise NotImplementedError("Unknown resolution=%s in argument \"delta\"" % (delta[1]))
    # Start
    temporal_labels = []
    if isinstance(start, str): # start given as date str
        start = dt.datetime.strptime(start, frmt)
    if isinstance(end, str): # end given as date str
        end = dt.datetime.strptime(end, frmt)
    if isinstance(delta, list): # delta given as custom format [int, str] (e.g. [7, "days"])
        time_steps, resolution = delta
        if resolution in ["months", "years"]:
            if isinstance(end, int): # end given as number of time-steps from start
                year, month, day = start.year, start.month, start.day
                hour, minute, second = start.hour, start.minute, start.second
                if resolution == "months":
#                    if day > 28:
#                        raise ValueError("Month denominated delta undefined for day > 28")
                    month += end * time_steps
                    year += (month - 1) // 12
                    month = (month - 1) % 12 + 1
                elif resolution == "years":
                    year += end * time_steps
                day = min(day, days_in_month(year, month))
                end = dt.datetime(year, month, day, hour, minute, second)
            # Start creating temporal labels at k*month/k*year resolution
            curr = start
            while curr <= end:
                temporal_labels.append(curr.strftime(frmt))
                year, month, day = curr.year, curr.month, start.day
                hour, minute, second = start.hour, start.minute, start.second
                if resolution == "months":
                    month += time_steps
                    year += (month - 1) // 12
                    month = (month - 1) % 12 + 1
                elif resolution == "years":
                    year += time_steps
                day = min(day, days_in_month(year, month))
                curr = dt.datetime(year, month, day, hour, minute, second)
        else: # resolution is one of ("seconds", "minutes", "hours", "days", "weeks")
            delta = dt.timedelta(**{resolution: time_steps})
            if isinstance(end, int): # end given as number of time-steps from start
                end = start + end * delta
            temporal_labels = [dt.strftime(frmt) for dt in datetime_range(start, end, delta)]
    elif isinstance(delta, dt.timedelta):
        if isinstance(end, int): # end given as number of time-steps from start
            end = start + end * delta
        temporal_labels = [dt.strftime(frmt) for dt in datetime_range(start, end, delta)]
    if not incl[0]:
        temporal_labels = temporal_labels[1:]
    if not incl[1]:
        temporal_labels = temporal_labels[:-1]
    return temporal_labels


def temporal_labels_to_periodic_indices(labels, period, resolution, frmt="%Y-%m-%d_%H-%M-%S"):
    """ Generates a chronologically ordered set of string-formatted time-stamps for a given range

    Arguments
    ---------
    labels : np.ndarray, list or tuple of str
    period : list of [int, str]
    resolution : list of [int, str]
    frmt : str

    Returns
    -------
    temporal_labels : list of str

    """
    # Check arguments
    supported_units = [
        "microseconds", "milliseconds", "seconds", "minutes", "hours", "days", "weeks", "months", "years"
    ]
    if not period[1] in supported_units:
        raise NotImplementedError(period)
    if not resolution[1] in supported_units:
        raise NotImplementedError(resolution)
    if supported_units.index(resolution[1]) > supported_units.index(period[1]) or (resolution[1] == period[1] and resolution[0] > period[0]):
        raise ValueError("Argument \"period\" must be a time-span that encompasses the time-span of argument \"resolution\". Received period=%s and resolution=%s" % (str(period), str(resolution)))
    # Start conversion
    if isinstance(labels, np.ndarray):
        original_shape = labels.shape
        labels = np.reshape(labels, (-1,))
        indices = np.zeros(labels.shape, dtype=int)
    elif isinstance(labels, list):
        indices = list(0 for _ in labels)
    elif isinstance(labels, tuple):
        indices = tuple(0 for _ in labels)
    elif isinstance(labels, pd.Series):
        indices = pd.Series([0 for _ in labels])
    #   Handle special cases
    if resolution[1] == "days" and period[1] == "days": # day -> day-of-days zero-indexed to first Monday
        unit_delta = dt.timedelta(days=resolution[0])
        period_units = period[0]
        for i in range(len(labels)):
            date = dt.datetime.strptime(labels[i], frmt)
            units_elapsed = (date - dt.datetime.min) // unit_delta
            indices[i] = units_elapsed % period_units
    elif resolution[1] == "months":
        period_units = period[0] # month -> month-of-months
        if period[1] == "years": # month -> month-of-year(s)
            period_units *= 12
        for i in range(len(labels)):
            date = dt.datetime.strptime(labels[i], frmt)
            indices[i] = (12 * date.year + date.month) % period_units
    elif resolution[1] == "years":
        period_units = period[0] # year -> year-of-years
        for i in range(len(labels)):
            date = dt.datetime.strptime(labels[i], frmt)
            indices[i] = date.year % period_units
    elif period[1] == "weeks":
        if resolution[1] == "microseconds": # microsecond -> microsecond-of-the-week(s)
            unit_delta = dt.timedelta(microseconds=resolution[0])
            period_units = 1000 * 1000 * 60 * 60 * 24 * 7 * period[0] / resolution[0]
        elif resolution[1] == "milliseconds": # millisecond -> millisecond-of-the-week(s)
            unit_delta = dt.timedelta(milliseconds=resolution[0])
            period_units = 1000 * 60 * 60 * 24 * 7 * period[0] / resolution[0]
        elif resolution[1] == "seconds": # second -> second-of-the-week(s)
            unit_delta = dt.timedelta(seconds=resolution[0])
            period_units = 60 * 60 * 24 * 7 * period[0] / resolution[0]
        elif resolution[1] == "minutes": # minute -> minute-of-the-week(s)
            unit_delta = dt.timedelta(minutes=resolution[0])
            period_units = 60 * 24 * 7 * period[0] / resolution[0]
        elif resolution[1] == "hours": # hour -> hour-of-the-week(s)
            unit_delta = dt.timedelta(hours=resolution[0])
            period_units = 24 * 7 * period[0] / resolution[0]
        elif resolution[1] == "days": # day -> day-of-the-week(s)
            unit_delta = dt.timedelta(days=resolution[0])
            period_units = 7 * period[0] / resolution[0]
        elif resolution[1] == "weeks": # week -> week-of-the-weeks
            unit_delta = dt.timedelta(weeks=resolution[0])
            period_units = period[0]
        for i in range(len(labels)):
            date = dt.datetime.strptime(labels[i], frmt)
            units_elapsed = (date - dt.datetime.min) // unit_delta
            indices[i] = units_elapsed % period_units
    else:
        unit_delta = dt.timedelta(**{resolution[1]: resolution[0]})
        period_units = period[0]
        period_unit_name = period[1][:-1]
        for i in range(len(labels)):
            date = dt.datetime.strptime(labels[i], frmt)
            period_origin_kwargs = {
                "year": 1, "month": 1, "day": 1, "hour": 0, "minute": 0, "second": 0
            }
            for unit_name in period_origin_kwargs.keys():
                unit_value = getattr(date, unit_name)
                period_origin_kwargs[unit_name] = unit_value
                if unit_name == period_unit_name:
                    unit_origin_value = (unit_value // period_units) * period_units
                    if unit_name in ["day", "month", "year"]: # handle one-based units
                        unit_origin_value = ((unit_value - 1) // period_units) * period_units + 1
                    period_origin_kwargs[unit_name] = unit_origin_value
                    break
            period_origin_date = dt.datetime(**period_origin_kwargs)
            indices[i] = (date - period_origin_date) // unit_delta
    if isinstance(labels, np.ndarray):
        labels = np.reshape(labels, original_shape)
        indices = np.reshape(indices, original_shape)
    return indices


def labels_to_ids(labels):
    orig_shape = labels.shape
    labels = np.reshape(labels, [-1])
    unique_labels = np.sort(np.unique(labels))
    label_index_map = to_key_index_dict(unique_labels)
    return np.reshape(np.array([label_index_map[label] for label in labels]), orig_shape)


# Preconditions:
#   Give: N = n_spatial, T = n_temporal, and F = n_feature
#   df.shape=(N*T, 2+F)
def compute_spatiotemporal_correlations(df, spatial_label_field, temporal_label_field, feature):
    spatial_labels = df[spatial_label_field].unique()
    temporal_labels = df[temporal_label_field].unique()
    n_spatial = len(spatial_labels)
    n_temporal = len(temporal_labels)
    A = np.reshape(df[feature].to_numpy().astype(float), (n_spatial, n_temporal))
    A = np.swapaxes(A, 0, 1)
    df = pd.DataFrame(A, index=temporal_labels, columns=spatial_labels)
    return df.corr()


if __name__ == "__main__":
#    convert_los_loop()
#    convert_sz_taxi()
#    convert_metr_la()
#    convert_pems_bay()
#    convert_solar()
#    convert_electricity()
#    convert_ecg_5000()
#    convert_covid_19()
#    convert_wabash_river_swat()
#    convert_wabash_river_observed()
#    convert_little_river()
#    convert_pems()
    convert_wabash_river_spatial()
    quit()
    method = "topk"
    method = "threshold"
    method = "smart-threshold"
    method = "threshold-topk"
    datasets = ["Solar", "Electricity", "ECG5000", "COVID-19"]
    spatial_label_fields = ["sensor", "sensor", "sensor", "country"]
    temporal_label_fields = ["date", "date", "date", "date"]
    features = ["power_MW", "power_kW", "signal_mV", "confirmed"]
    for i in range(len(datasets))[:]:
        path = os.sep.join(["Data", datasets[i], "Observed", "Spatiotemporal.csv"])
        df = pd.read_csv(path)
        print(df)
        corr = compute_spatiotemporal_correlations(
            df,
            spatial_label_fields[i],
            temporal_label_fields[i],
            features[i]
        )
        print(corr)
        print(datasets[i])
        n_spatial = len(corr)
        if method == "smart-threshold":
            avg_degree = int(n_spatial * 0.05)
            print("Target Average Node Degree =", avg_degree)
            n_expected_edges = avg_degree * n_spatial
            thresh, step_size = 0.5, 0.001
            tmp = corr.abs().to_numpy()
            while np.sum(tmp > thresh) > n_expected_edges:
                print(np.sum(tmp > thresh))
                thresh += step_size
            print("Found Threshold =", thresh)
            corr[corr.abs() < thresh] = 0
            df = corr
        elif method == "threshold-topk":
            avg_degree = int(n_spatial * 0.05)
            min_degree, max_degree = 2, n_spatial
            print("Target Mean Node Degree =", avg_degree)
            n_expected_edges = avg_degree * n_spatial
            thresh, step_size = 0.5, 0.001
            tmp = corr.abs().to_numpy()
            while np.sum(tmp > thresh) > n_expected_edges:
                thresh += step_size
            print("Found Threshold =", thresh)
            ks = np.sum(tmp > thresh, axis=1)
            print("Node Degrees =")
            print(ks)
            ks = np.sum(tmp > thresh, axis=0)
            print("Node Degrees =")
            print(ks)
            ks = np.clip(ks, min_degree, max_degree)
            print("Node Degrees =")
            print(ks)
            adj = np.zeros((n_spatial, n_spatial))
            for j in range(len(ks)):
                indices = np.argsort(tmp[j,:])[-ks[j]:]
                adj[j,indices] = 1
                adj[indices,j] = 1
            print(adj)
            df = pd.DataFrame(adj, columns=corr.columns)
            print(df)
        elif method == "threshold":
            corr[corr.abs() < 0.95] = 0
            df = corr
        elif method == "topk":
            k = int(n_spatial * 0.05) + 1
            print(k)
            df = pd.DataFrame(
                np.where(corr.rank(axis=1,method='min',ascending=False)>k, 0, corr),
                columns=corr.columns
            )
        else:
            raise NotImplementedError(method)
        node_degrees = np.sum(df.to_numpy() > 0, axis=1)
        print("Node Degrees =")
        print(node_degrees)
        node_degrees = np.sum(df.to_numpy() > 0, axis=0)
        print("Node Degrees =")
        print(node_degrees)
        print("Mean Node Degree =", np.mean(node_degrees))
        df = adjacency_to_edgelist(df, np.array(df.columns))
        print(df)
        path = os.sep.join(["Data", datasets[i], "Observed", "Graph.csv"])
        df.to_csv(path, index=False)
