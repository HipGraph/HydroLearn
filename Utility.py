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
import re
import os
import inspect


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


def get_nonroot_process_ranks(root_rank, n_processes):
    return [p for p in range(n_processes) if p != root_rank]


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


# Creates indices that index a set of input and output windows
#   timesteps: number of time-steps in source
#   in_length: time-steps per input window
#   out_length: time-steps per output window
#   horizon: time-steps separating end of input window and start of output window
#   stride: time-steps between window origins
#   offset: time-step where first window begins
def input_output_window_indices(timesteps, in_length, out_length, horizon=1, stride=1, offset=0):
    indices = sliding_window_indices(timesteps, in_length + (horizon - 1) + out_length, stride, offset)
    return indices[:,:in_length], indices[:,-out_length:]


def contiguous_window_indices(n, length, stride=1, offset=0):
    assert length % stride == 0, "Windows cannot be contiguous with stride != 1"
    return np.arange(offset, n, length // stride)


# Calculate mean absolute error (MAE) for each spatial element and feature
# Pre-conditions:
#   Y.shape=(n_samples, n_temporal, n_spatial, n_features)
#   Yhat.shape=(n_samples, n_temporal, n_spatial, n_features)
# Post-conditions:
#   MAE.shape=(n_spatial, n_features)
def MAE(Y, Yhat, **kwargs):
    return np.mean(np.abs(Y - Yhat), axis=(0, 1))


# Calculate mean square error (MSE) for each spatial element and feature
# Pre-conditions:
#   Y.shape=(n_samples, n_temporal, n_spatial, n_features)
#   Yhat.shape=(n_samples, n_temporal, n_spatial, n_features)
# Post-conditions:
#   MSE.shape=(n_spatial, n_features)
def MSE(Y, Yhat, **kwargs):
    return np.mean((Y - Yhat)**2, axis=(0, 1))


# Calculate root mean square error (RMSE) for each spatial element and feature
# Pre-conditions:
#   Y.shape=(n_samples, n_temporal, n_spatial, n_features)
#   Yhat.shape=(n_samples, n_temporal, n_spatial, n_features)
# Post-conditions:
#   MAPE.shape=(n_spatial, n_features)
def MAPE(Y, Yhat, **kwargs):
    eps = 1e-3
    denom = np.copy(Y)
    denom[np.abs(denom) < eps] = 1 # Makes this comp safe
    if 0:
        mins = np.min(denom, axis=(0, 1))
        for i in range(mins.shape[0]):
            print("mins[%d,:]" % (i), mins[i,:])
    return np.mean(np.abs((Y - Yhat) / denom), axis=(0, 1))


# Calculate root mean square error (RMSE) for each spatial element and feature
# Pre-conditions:
#   Y.shape=(n_samples, n_temporal, n_spatial, n_features)
#   Yhat.shape=(n_samples, n_temporal, n_spatial, n_features)
# Post-conditions:
#   RMSE.shape=(n_spatial, n_features)
def RMSE(Y, Yhat, **kwargs):
    return np.sqrt(MSE(Y, Yhat))


# Calculate normalized root mean square error (NRMSE) for each spatial element and feature
# Preconditions:
#   Y.shape=(n_samples, n_temporal, n_spatial, n_features)
#   Yhat.shape=(n_samples, n_temporal, n_spatial, n_features)
#   mins.shape=(n_periodic_samples, n_spatial, n_features)
#   maxes.shape=(n_periodic_samples, n_spatial, n_features)
# Post-conditions:
#   NRMSE.shape=(n_spatial, n_features)
def NRMSE(Y, Yhat, **kwargs):
    mins, maxes = kwargs["mins"], kwargs["maxes"]
    mins = np.min(mins, axis=0)
    maxes = np.max(maxes, axis=0)
    ranges = np.abs(maxes - mins)
    eps = 1e-3
    ranges[np.abs(ranges) < eps] = 1 # Makes this comp safe
    if 0:
        ranges = np.min(denom, axis=(0, 1))
        for i in range(ranges.shape[0]):
            print("ranges[%d,:]" % (i), ranges[i,:])
    return RMSE(Y, Yhat) / ranges


# Preconditions:
#   errors: a list of numpy arrays with shape=(n_spatial, n_features)
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
    }
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
    return [a_i for a_i in a if a_i not in b]


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


def get_dict_values(a, keys, ignore_absent=False):
    values = []
    for key in keys:
        if not key in a and ignore_absent:
            continue
        values.append(a[key])
    if isinstance(keys, np.ndarray):
        values = np.array(values)
    return values


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


def minmax_transform(A, minimum, maximum, revert=False, a=-1, b=1, clamp=False):
    if revert:
        if clamp:
            A[A < a] = a # convert model output to [a, b]
            A[A > b] = b
        A = ((A - a) / (b - a)) * (maximum - minimum) + minimum
    else:
        A = (b - a) * ((A - minimum) / (maximum - minimum)) + a
        if clamp:
            A[A < a] = a # make sure output is in [a, b] (e.g. when min/max are approximations)
            A[A > b] = b
    return A


def zscore_transform(A, mean, standard_deviation, revert=False):
    if revert:
        return A * standard_deviation + mean
    return (A - mean) / standard_deviation


def log_transform(A, _1, _2, revert=False):
    if revert:
#        A[A < -10] = -10
#        A[A > 10] = 10
        return np.exp(A)
    A[A == 0] = 1 + 1e-5
    return np.log(A)


def identity_transform(A, _1, _2, revert=False):
    return A


def to_spatiotemporal_format(df, temporal_labels):
    spatial_labels = np.array(df.columns)
    features = df.to_numpy()
    n_temporal_labels = temporal_labels.shape[0]
    n_spatial_labels = spatial_labels.shape[0]
    temporal_labels = np.tile(temporal_labels, n_spatial_labels)
    spatial_labels = np.repeat(spatial_labels, n_temporal_labels)
    features = np.concatenate([features[:,i] for i in range(features.shape[1])])
    return pd.DataFrame({"temporal_label": temporal_labels, "spatial_label": spatial_labels, "feature": features})


def adjacency_to_edgelist(df):
    adj, node_labels = df.to_numpy(), df.columns
    srcs, dsts, ws = [], [], []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i,j] > 0:
                srcs += [node_labels[i]]
                dsts += [node_labels[j]]
                ws += [adj[i,j]]
    data_map = {"source": srcs, "destination": dsts}
    if len(np.unique(np.array(ws))) > 1:
        data_map["weight"] = ws
    return pd.DataFrame(data_map)
    

def convert_los_loop():
    os.makedirs("LosLoop", exist_ok=True)
    path= os.sep.join(["Models", "T-GCN", "data", "los_speed.csv"])
    spatmp_df = pd.read_csv(path)
    # Get spatial labels
    spatial_labels = np.array(spatmp_df.columns)
    df = pd.DataFrame({"sensor": spatial_labels})
    path = os.sep.join(["LosLoop", "SpatialLabels.csv"])
    df.to_csv(path, index=False)
    # Create temporal labels
    temporal_labels = generate_temporal_labels(
        dt.datetime(2012, 3, 1),
        dt.datetime(2012, 3, 8),
        dt.timedelta(minutes=5),
        bound_inclusion=[True, False]
    )
    temporal_labels = np.array(temporal_labels)
    df = pd.DataFrame({"date": temporal_labels})
    path = os.sep.join(["LosLoop", "TemporalLabels.csv"])
    df.to_csv(path, index=False)
    # Convert spatiotemporal data
    df = to_spatiotemporal_format(spatmp_df, temporal_labels)
    df = df.rename({"temporal_label":"date", "spatial_label":"sensor", "feature":"speedmph"}, axis=1)
    path = os.sep.join(["LosLoop", "Spatiotemporal.csv"])
    df.to_csv(path, index=False)
    # Convert adjacency data
    path= os.sep.join(["Models", "T-GCN", "data", "los_adj.csv"])
    df = pd.read_csv(path, names=spatial_labels)
    df = adjacency_to_edgelist(df, spatial_labels)
    path = os.sep.join(["LosLoop", "Graph.csv"])
    df.to_csv(path, index=False)


def convert_sz_taxi():
    os.makedirs("SZTaxi", exist_ok=True)
    path= os.sep.join(["Models", "T-GCN", "data", "sz_speed.csv"])
    spatmp_df = pd.read_csv(path)
    # Get spatial labels
    spatial_labels = np.array(spatmp_df.columns)
    df = pd.DataFrame({"sensor": spatial_labels})
    path = os.sep.join(["SZTaxi", "SpatialLabels.csv"])
    df.to_csv(path, index=False)
    # Create temporal labels
    temporal_labels = generate_temporal_labels(
        dt.datetime(2015, 1, 1),
        dt.datetime(2015, 2, 1),
        dt.timedelta(minutes=15),
        bound_inclusion=[True, False]
    )
    temporal_labels = np.array(temporal_labels)
    df = pd.DataFrame({"date": temporal_labels})
    path = os.sep.join(["SZTaxi", "TemporalLabels.csv"])
    df.to_csv(path, index=False)
    # Convert spatiotemporal data
    df = to_spatiotemporal_format(spatmp_df, temporal_labels)
    df = df.rename({"temporal_label":"date", "spatial_label":"sensor", "feature":"speedmph"}, axis=1)
    path = os.sep.join(["SZTaxi", "Spatiotemporal.csv"])
    df.to_csv(path, index=False)
    # Convert adjacency data
    path= os.sep.join(["Models", "T-GCN", "data", "sz_adj.csv"])
    df = pd.read_csv(path, names=spatial_labels)
    df = adjacency_to_edgelist(df, spatial_labels)
    path = os.sep.join(["SZTaxi", "Graph.csv"])
    df.to_csv(path, index=False)


def convert_solar():
    os.makedirs("Solar", exist_ok=True)
    import glob
    path = os.sep.join(["Data", "al-pv-2006", "Actual_*"])
    paths = glob.glob(path)
    # Convert spatial labels
    spatial_labels = []
    for path in paths:
        fname = path.split(os.sep)[-1]
        spatial_labels += [fname.replace("Actual_", "").replace("2006_", "").replace("_5_Min.csv", "")]
    spatial_labels = np.array(spatial_labels)
    path = os.sep.join(["Solar", "SpatialLabels.csv"])
    df = pd.DataFrame({"sensor": spatial_labels})
    df.to_csv(path, index=False)
    # Convert temporal labels
    df = pd.read_csv(paths[0])
    df["LocalTime"] = pd.to_datetime(df["LocalTime"])
    df["LocalTime"] = df["LocalTime"].dt.strftime("%Y-%m-%d_%H-%M-%S")
    temporal_labels = df["LocalTime"].to_numpy()
    df = pd.DataFrame({"date": temporal_labels})
    path = os.sep.join(["Solar", "TemporalLabels.csv"])
    df.to_csv(path, index=False)
    # Convert spatiotemporal
    label_data_map = {}
    for label, path in zip(spatial_labels, paths):
        df = pd.read_csv(path)
        label_data_map[label] = df["Power(MW)"].to_numpy()
    spatmp_df = pd.DataFrame(label_data_map)
    df = to_spatiotemporal_format(spatmp_df, temporal_labels)
    df = df.rename({"temporal_label":"date", "spatial_label":"sensor", "feature":"power_MW"}, axis=1)
    print(df)
    path = os.sep.join(["Solar", "Spatiotemporal.csv"])
    df.to_csv(path, index=False)


def convert_electricity():
    os.makedirs("Electricity", exist_ok=True)
    path = os.sep.join(["Data", "LD2011_2014.txt", "LD2011_2014.txt"])
    df = pd.read_csv(path, sep=";")
    print(df)
    # Convert temporal labels
    df["Unnamed: 0"] = pd.to_datetime(df["Unnamed: 0"])
    df["Unnamed: 0"] = df["Unnamed: 0"].dt.strftime("%Y-%m-%d_%H-%M-%S")
    temporal_labels = df["Unnamed: 0"].to_numpy()
    path = os.sep.join(["Electricity", "TemporalLabels.csv"])
    pd.DataFrame({"date": temporal_labels}).to_csv(path, index=False)
    # Convert spatiotemporal
    spatmp_df = df.drop("Unnamed: 0", axis=1)
    df = to_spatiotemporal_format(spatmp_df, temporal_labels)
    df["feature"] = df["feature"].str.replace(",", ".").fillna(0)
    df = df.rename({"temporal_label":"date", "spatial_label":"sensor", "feature":"power_kW"}, axis=1)
    path = os.sep.join(["Electricity", "Spatiotemporal.csv"])
    df.to_csv(path, index=False)
    # Convert spatial labels
    spatial_labels = spatmp_df.columns.to_numpy()
    path = os.sep.join(["Electricity", "SpatialLabels.csv"])
    df = pd.DataFrame({"sensor": spatial_labels})
    df.to_csv(path, index=False)
    

def convert_ecg_5000():
    os.makedirs("ECG5000", exist_ok=true)
    # convert spatial labels
    spatial_labels = np.array(["ECG%d"%(i) for i in range(1, 141)])
    path = os.sep.join(["ECG5000", "spatiallabels.csv"])
    df = pd.DataFrame({"sensor": spatial_labels})
    df.to_csv(path, index=False)
    # Convert temporal labels
    temporal_labels = generate_temporal_labels(
        dt.datetime(1, 1, 1, 0, 0, 0),
        dt.datetime(1, 1, 1, 1, 23, 20),
        dt.timedelta(seconds=1),
        bound_inclusion=[True, False]
    )
    temporal_labels = np.array(temporal_labels)
    path = os.sep.join(["ECG5000", "TemporalLabels.csv"])
    df = pd.DataFrame({"date": temporal_labels})
    df.to_csv(path, index=False)
    # Convert spatiotemporal
    path = os.sep.join(["Data", "ECG5000", "ECG5000_TRAIN.txt"])
    df1 = pd.read_csv(path, sep="  ", names=spatial_labels)
    path = os.sep.join(["Data", "ECG5000", "ECG5000_TEST.txt"])
    df2 = pd.read_csv(path, sep="  ", names=spatial_labels)
    spatmp_df = pd.concat([df1, df2])
    df = to_spatiotemporal_format(spatmp_df, temporal_labels)
    df = df.rename({"temporal_label":"date", "spatial_label":"sensor", "feature":"signal_mV"}, axis=1)
    path = os.sep.join(["ECG5000", "Spatiotemporal.csv"])
    df.to_csv(path, index=False)


def convert_covid_19():
    os.makedirs("COVID-19", exist_ok=True)
    regions = ['US','Canada','Mexico','Russia','United Kingdom','Italy','Germany','France','Belarus','Brazil','Peru','Ecuador','Chile','India','Turkey','Saudi Arabia','Pakistan','Iran','Singapore','Qatar','Bangladesh','Arab','China','Japan','Korea']
    path = os.sep.join(["Data", "COVID-19", "covid19.csv"])
    df = pd.read_csv(path)
    # Convert spatiotemporal
    temporal_labels = df.columns.to_list()[4:]
    fields = ["Country/Region"] + temporal_labels
    df = df[fields]
    #   Keep only regions of interest
    pat = '|'.join(r"\b{}\b".format(x) for x in regions)
    df = df[df["Country/Region"].str.contains(pat)]
    #   Sum up when there are multiple records for a given region
    df[temporal_labels] = df.groupby(["Country/Region"])[temporal_labels].transform("sum")
    #   Keep only the first (though all entries are the same now since they are summed)
    df = df.drop_duplicates(subset=["Country/Region"], keep="first")
    spatial_labels = df["Country/Region"].to_list()
    df = df.transpose()
    df = df.drop("Country/Region", axis=0)
    df.columns = spatial_labels
    temporal_labels = np.array(temporal_labels)
    spatmp_df = to_spatiotemporal_format(df, temporal_labels)
    spatmp_df = spatmp_df.rename({"temporal_label":"date", "spatial_label":"country", "feature":"confirmed"}, axis=1)
    spatmp_df["date"] = pd.to_datetime(spatmp_df["date"])
    spatmp_df["date"] = spatmp_df["date"].dt.strftime("%Y-%m-%d")
    path = os.sep.join(["COVID-19", "Spatiotemporal.csv"])
    spatmp_df.to_csv(path, index=False)
    # Convert spatial labels
    path = os.sep.join(["COVID-19", "SpatialLabels.csv"])
    df = pd.DataFrame({"country": spatial_labels})
    df.to_csv(path, index=False)
    # Convert temporal labels
    df = pd.DataFrame({"date": temporal_labels})
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    path = os.sep.join(["COVID-19", "TemporalLabels.csv"])
    df.to_csv(path, index=False)


def convert_wabash_river_swat():
    out_dir = os.sep.join(["WabashRiver", "SWAT"])
    os.makedirs(out_dir, exist_ok=True)
    spatial_labels = np.arange(1276) + 1
    path = os.sep.join([out_dir, "SpatialLabels.csv"])
    pd.DataFrame({"subbasin": spatial_labels}).to_csv(path, index=False)
    temporal_labels = generate_temporal_labels(
        dt.datetime(1929, 1, 1),
        dt.datetime(2013, 12, 31),
        dt.timedelta(days=1),
        "%Y-%m-%d", 
        bound_inclusion=[True, True]
    )
    path = os.sep.join([out_dir, "TemporalLabels.csv"])
    pd.DataFrame({"date": temporal_labels}).to_csv(path, index=False)


def convert_wabash_river_observed():
    out_dir = os.sep.join(["WabashRiver", "Observed"])
    os.makedirs(out_dir, exist_ok=True)
    spatial_labels = [43, 757, 169, 348, 529]
    path = os.sep.join([out_dir, "SpatialLabels.csv"])
    pd.DataFrame({"subbasin": spatial_labels}).to_csv(path, index=False)
    temporal_labels = generate_temporal_labels(
        dt.datetime(1985, 1, 1),
        dt.datetime(2019, 1, 6),
        dt.timedelta(days=1),
        "%Y-%m-%d", 
        bound_inclusion=[True, True]
    )
    path = os.sep.join([out_dir, "TemporalLabels.csv"])
    pd.DataFrame({"date": temporal_labels}).to_csv(path, index=False)


def convert_little_river():
    out_dir = os.sep.join(["LittleRiver", "Observed"])
    os.makedirs(out_dir, exist_ok=True)
    spatial_labels = ["B", "F", "I", "J", "K", "M", "N", "O"]
    path = os.sep.join([out_dir, "SpatialLabels.csv"])
    pd.DataFrame({"subbasin": spatial_labels}).to_csv(path, index=False)
    temporal_labels = generate_temporal_labels(
        dt.datetime(1968, 1, 1),
        dt.datetime(2004, 12, 31),
        dt.timedelta(days=1),
        "%Y-%m-%d", 
        bound_inclusion=[True, True]
    )
    path = os.sep.join([out_dir, "TemporalLabels.csv"])
    pd.DataFrame({"date": temporal_labels}).to_csv(path, index=False)


def convert_pems():
    import glob
    import gzip
    import pandas as pd
    os.makedirs("PEMS", exist_ok=True)
    dirs = glob.glob(os.sep.join(["Downloads", "d??"]))
    print(dirs)
    header = ["timestamp", "station", "district", "freeway", "direction_of_travel", "lane_type", "station_length", "total_samples", "prop_observed", "total_flow", "avg_occupancy", "avg_speed"]
    for i in range(8):
        header += ["lane_%d_samples"%(i), "lane%d_flow"%(i), "lane%d_avg_occupancy"%(i), "lane%d_avg_speed"%(i), "lane%d_observed"%(i)]
    cols_kept = ["timestamp", "station", "total_samples", "prop_observed", "total_flow", "avg_occupancy", "avg_speed"]
    for _dir in dirs:
        district = os.path.basename(_dir)
        paths = glob.glob(os.sep.join([_dir, "*.txt.gz"]))
        dfs = []
        for path in paths[:3]:
            start = time.time()
            with gzip.open(path, "rb") as f:
#                print(time.time() - start)
                start = time.time()
                df = pd.read_csv(f, names=header)
                df = df[cols_kept]
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d_%H-%M-%S")
#                print(time.time() - start)
                dfs.append(df)
            print(df)
        df = pd.concat(dfs)
        path = os.sep.join(["PEMS", "%s.csv" % (district)])
        df.to_csv(path, index=False)


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


def generate_temporal_labels(start, end, delta, frmt="%Y-%m-%d_%H-%M-%S", bound_inclusion=[True, True]):
    temporal_labels = []
    curr = start
    if not bound_inclusion[0]:
        curr += delta
    while (curr <= end if bound_inclusion[1] else curr < end):
        temporal_labels += [curr.strftime(frmt)]
        curr += delta
    return temporal_labels


def dates_to_daysofyear(dates, date_format):
    if isinstance(dates, list) and isinstance(dates[0], np.ndarray):
        return [dates_to_daysofyear(dates_array) for dates_array in dates]
    elif isinstance(dates, np.ndarray):
        dates_shape = dates.shape
        dates = dates.reshape(-1)
        days_of_year = np.zeros(dates.shape, dtype=np.int16)
        for d in range(dates.shape[0]):
            if dates[d] == "":
                continue
            date = dt.datetime.strptime(dates[d], date_format)
            new_year_day = dt.datetime(year=date.year, month=1, day=1)
            day_of_year = (date - new_year_day).days + 1
            days_of_year[d] = day_of_year
        dates = dates.reshape(dates_shape)
        days_of_year = days_of_year.reshape(dates.shape)
    else:
        raise NotImplementedError("Can only convert dates to days of year for ndarray or list of ndarrays")
    return days_of_year


# Purpose:
#   Converts a set of time-stamps (datetime strings) into a set of indices that repeat according to a period.
#   For example, given 10 years of daily time-stamps and period of one year, this function would produce 10 years
#       (~3650) of indices in [0, 365].
#   These indices, for example, allow us to treat time-stamps Januray 1st 2012 and Januray 1st 2013 as two samples
#       from the same source. 
#   This allows us to compute, for example, seasonal mean at daily resolution which would consist of 365 daily 
#       mean feature (streamflow, vehicle speed, etc.) values.
# Arguments:
#   temporal_labels: time-stamps as datetime strings
#   period: 2-tuple specifying duration of the period (e.g. [2, "years"])
#   timestep: 2-tuple specifying duration between time-stamps (e.g. [10, "minutes"]) with specifal input "infer" 
#       telling the algorithm to infer time-step duration from adjacent samples.
#       Note: timestep is bin-width in common sampling regimes. Most cases will use a timestep equivalent to 
#           the duration between adjacent samples in temporal_labels but a greater timestep (bin-width) may be 
#           given to treat multiple time-stamps as belonging to same unit.
#           For example, actual time-stamps are 5 minutes apart but a timestep [10, "minutes"] is given. In this
#               case, time-stamps 00:00:00 and 00:05:00 are mapped to index 0 since they both fall into the 
#               interval [00:00:00, 00:10:00].
#   temporal_label_format: format of datetime strings
def temporal_labels_to_periodic_indices(temporal_labels, temporal_period, temporal_resolution, temporal_label_format="%Y-%m-%d_%H-%M-%S"):
    # Check for macro op.
    if isinstance(temporal_labels, list) \
        and len(temporal_labels) > 0 \
        and isinstance(temporal_labels[0], np.ndarray):
        results = []
        for labels in temporal_labels:
            results += [temporal_labels_to_periodic_indices(
                labels, 
                temporal_period, 
                temporal_resolution, 
                temporal_label_format
            )]
        return results
    # Start
    orig_shape = temporal_labels.shape
    temporal_labels = np.reshape(temporal_labels, [-1])
    # Check that present and stated temporal resolution are consistent
    a = dt.datetime.strptime(temporal_labels[0], temporal_label_format)
    b = dt.datetime.strptime(temporal_labels[1], temporal_label_format)
    present_res_delta = b - a
    stated_res_delta = dt.timedelta(**{temporal_resolution[1]: temporal_resolution[0]})
    if False and present_res_delta != stated_res_delta:
        raise ValueError("Temporal resolution of data (%s) differs from stated temporal resolution (%s)" % (
            str(present_res_delta), 
            str(stated_res_delta)
        ))
    res_delta = stated_res_delta
    periodic_indices = np.zeros(temporal_labels.shape, dtype=np.int)
    for i in range(len(temporal_labels)):
        if temporal_labels[i] == "":
            continue
        date = dt.datetime.strptime(temporal_labels[i], temporal_label_format)
        # construct base_date (starting date of the current period) from date
        base_date_args = {"year": 1, "month": 1, "day": 1, "hour": 0, "minute": 0, "second": 0}
        period_key = temporal_period[1][:-1] if temporal_period[1].endswith("s") else temporal_period[1]
        period_value = temporal_period[0]
        for key in base_date_args.keys():
            value = getattr(date, key)
            base_date_args[key] = value
            if period_key == key:
                # round down to nearest period start
                start_value = (value // period_value) * period_value
                if start_value == 0 and key in ["year", "month", "day"]: # Not zero-based, need to increment
                    start_value += 1
                base_date_args[key] = start_value
                break
        base_date = dt.datetime(**base_date_args)
        # compute timedelta from base_date (date - base_date) and convert this to index (// timestep_delta)
        periodic_indices[i] = (date - base_date) // res_delta
#        print(base_date, date, periodic_indices[i])
    temporal_labels = np.reshape(temporal_labels, orig_shape)
    periodic_indices = np.reshape(periodic_indices, orig_shape)
    return periodic_indices


def labels_to_ids(labels):
    orig_shape = labels.shape
    labels = np.reshape(labels, [-1])
    unique_labels = np.sort(np.unique(labels))
    label_index_map = to_key_index_dict(unique_labels)
    return np.reshape(np.array([label_index_map[label] for label in labels]), orig_shape)


# Preconditions:
#   Give: N = n_spatial, T = n_temporal, and F = n_features
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
