import torch
import math
import numpy as np
import datetime as dt
import pickle
import time
import sys
import hashlib
from fastdtw import fastdtw
from progressbar import ProgressBar
from scipy.special import comb
import networkx as nx
import itertools
from os import sep as os_sep
#from Variables import Variables
from inspect import getargspec, signature


np.set_printoptions(precision=3, suppress=True, linewidth=200)


def merge_dicts(a, b):
    if not (isinstance(a, dict) or isinstance(b, dict)):
        raise ValueError("Expected two dictionaries")
    return {**a, **b}


def is_anything(item):
    return True

def is_none(item):
    return item is None

def is_string(item):
    return isinstance(item, str)


def is_list_of_strings(item):
    if isinstance(item, list) and len(item) > 0:
        return all(isinstance(i, str) for i in item)
    return False


def get_func_args(func):
    args, var_args, kw_args, def_vals = getargspec(func)
    if "self" in args:
        args.remove("self")
    req_args = args[:-len(def_vals)]
    def_args = dict(zip(list_subtract(list(args), req_args), def_vals))
    return req_args, var_args, kw_args, def_args


def path(items):
    return os_sep.join(map(str, items))


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


def invoked_me(argv, script_name):
    return argv[0] == script_name


def get_feature_index_map(data_source):
    var = Variables()
    return var.get(data_source).get("feature_index_map")


def sort_dictionary(a, by="key"):
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


def compute_n_timesteps(dates_interval):
    start_date = dt.datetime.strptime(dates_interval[0], "%Y-%m-%d")
    end_date = dt.datetime.strptime(dates_interval[1], "%Y-%m-%d")
    return (end_date - start_date).days + 1


def compute_n_contiguous_sets(reduction_factor, reduction_stride):
    return reduction_factor // reduction_stride


def compute_n_temporal(n_temporal, reduction_factor, reduction_stride):
    n_reduced_sets = self.compute_n_reduced_sets(reduction_factor, reduction_stride)
    n_reduced_units = np.zeros((n_reduced_sets), dtype=np.int)
    for set_idx in range(n_reduced_sets):
        n_reduced_units[set_idx] =  (n_temporal - set_idx - 1) // n_reduced_sets + 1
    return n_reduced_units


def filter_dates(historical, historical_dates, dates_interval):
    start_idx = np.where(historical_dates == dates_interval[0])[0][0]
    end_idx = np.where(historical_dates == dates_interval[1])[0][0]
    return historical[start_idx:end_idx+1], historical_dates[start_idx:end_idx+1]


def format_memory(n_bytes):
    if n_bytes == 0:
        return "0B"
    mem_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(n_bytes, 1024)))
    p = math.pow(1024, i)
    s = int(round(n_bytes / p, 2))
    return "%s%s" % (s, mem_name[i])


def get_memory_of(var):
    n_bytes = sys.getsizeof(var)
    return format_memory(n_bytes)


def convert_dates_to_daysofyear(dates):
    if isinstance(dates, list) and isinstance(dates[0], np.ndarray):
        return [convert_dates_to_daysofyear(dates_array) for dates_array in dates]
    elif isinstance(dates, np.ndarray):
        dates_shape = dates.shape
        dates = dates.reshape(-1)
        days_of_year = np.zeros(dates.shape, dtype=np.int16)
        for d in range(dates.shape[0]):
            if dates[d] == "":
                continue
            date = dt.datetime.strptime(dates[d], "%Y-%m-%d")
            new_year_day = dt.datetime(year=date.year, month=1, day=1)
            day_of_year = (date - new_year_day).days + 1
            days_of_year[d] = day_of_year
        dates = dates.reshape(dates_shape)
        days_of_year = days_of_year.reshape(dates.shape)
    else:
        raise NotImplementedError("Can only convert dates to days of year for ndarray or list of ndarrays")
    return days_of_year


def compute_dynamic_time_warping(historical, variables):
    n_subbasins = historical.shape[-1]
    dtw = np.zeros((n_subbasins, n_subbasins))
    pb = ProgressBar()
    for i in pb(range(n_subbasins-1)):
        for j in range(i+1, n_subbasins):
            distance, path = fastdtw(historical[:,i], historical[:,j])
            dtw[i,j] = distance
            dtw[j,i] = distance
    return dtw


def compute_correlations(historical):
    return np.corrcoef(np.swapaxes(historical, 0, 1))


def compute_temporal_differencing(reduced_historical, iterations, variables):
    pb = ProgressBar()
    for i in pb(range(iterations)):
        tmp = np.copy(reduced_historical)
        for j in range(1, reduced_historical.shape[1]):
            reduced_historical[:,j,:,:] = reduced_historical[:,j,:,:] - tmp[:,j-1,:,:]
    reduced_historical[:,0,:,:] = 0
    return reduced_historical


# Calculate NRMSE for each subbasin
# Preconditions:
#   prediction.shape=(n_samples, n_temporal, n_spatial, n_features),
def compute_nrmse(prediction, groundtruth, minimums, maximums):
    """
    print(prediction.shape)
    print(groundtruth.shape)
    print("Prediction =", prediction[0,:,:,:])
    print("Groundtruth =", groundtruth[0,:,:,:])
    print(minimums.shape)
    print(maximums.shape)
    """
    minimums = np.min(minimums, axis=0)
    maximums = np.max(maximums, axis=0)
    ranges = np.abs(maximums - minimums)
    """
    print("Min =", minimums)
    print("Max =", maximums)
    print(ranges.shape)
    """
    nrmses = np.sqrt(np.mean((prediction - groundtruth)**2, axis=(0, 1))) / ranges
    return nrmses


# Preconditions:
#   errors: a list of numpy arrays with shape=(n_subbasins, n_features)
def curate_error_report(errors, spatial_labels, feature_labels, partitions, spatial_name="Subbasin", error_name="NRMSE"):
    error_report = []
    error_report += ["########################################"]
    for i in range(len(partitions)):
        error_report += ["%s %s = %.4f" % (partitions[i], error_name, np.mean(errors[i][:,:]))]
        for j in range(len(feature_labels)):
            error_report += ["\t%s %s = %.4f" % (feature_labels[j], error_name, np.mean(errors[i][:,j], axis=0))]
            for k in range(len(spatial_labels[i])):
                error_report += [
                    "\t\t%s %4s %s = %.4f" % (spatial_name, spatial_labels[i][k], error_name, errors[i][k,j])
                ]
    error_report += [""]
    error_report += ["########################################"]
    return "\n".join(error_report)


# Construct a - b
def list_subtract(a, b):
    return [a_i for a_i in a if a_i not in b]


def to_dictionary(keys, values):
    if not isinstance(keys, list) and not isinstance(values, list):
        raise ValueError("Keys and values must be type list")
    return {key: value for key, value in zip(keys, values)}


def invert_dictionary(a):
    return {value: key for key, value in a.items()}


def to_cache(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)


def from_cache(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def hash_str_to_int(string, n_digits):
    num = int(hashlib.sha256(string.encode("utf-8")).hexdigest(), 16) % 10**n_digits
    cur_digits = len(str(num))
    return num * 10**(n_digits - cur_digits)


def make_msg_block(msg, block_char):
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
    device_items = []
    for item in items:
        device_items += [item.to(device)]
    return device_items


def to_tensor(items, types):
    tensors = []
    for i in range(len(items)):
        if isinstance(items[i], torch.Tensor):
            tensors += items[i]
        elif isinstance(items[i], np.ndarray):
            if isinstance(types, list):
                tensors += [torch.tensor(items[i], dtype=types[i])]
            else:
                tensors += [torch.tensor(items[i], dtype=types)]
        else:
            raise NotImplementedError("Only numpy.ndarray and torch.Tensor types are implemented")
    return tensors


def to_ndarray(items):
    ndarrays = []
    for item in items:
        if isinstance(item, torch.Tensor):
            ndarrays += [item.detach().cpu().numpy()]
        elif isinstance(item, np.ndarray):
            ndarrays += [item]
        elif isinstance(item, list):
            ndarrays += [np.array(item)]
        else:
            raise NotImplementedError()
    return ndarrays


# moveaxis(a, [0, 1, 2], [1, 2, 0]) means we move axis 0 to 1, axis 1 to 2, and axis 2 to 0
def move_axes(items, sources, destinations):
    if not isinstance(items, list) or not isinstance(sources, list) or not isinstance(destinations, list):
        raise ValueError("Items, sources, and destinations must be type list")
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


def get_chkpt_dir(chkpt_dir, model_name):
    return chkpt_dir + os.sep + model_name


def get_eval_dir(eval_dir, model_name):
    return eval_dir + os.sep + model_name


def minmax_transform(A, minimum, maximum, revert=False):
    a, b = -1, 1
    clamp = True
    clamp = False
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


def tanh_transform(A, mean, standard_deviation, revert=False):
    if revert:
        return np.arctanh(A / 0.5 - 1.0) * standard_deviation / 0.01 + mean
    return 0.5 * (np.tanh(0.01 * (A - mean) / standard_deviation) + 1.0)


def log_transform(A, _1, _2, revert=False):
    if revert:
#        A[A < -10] = -10
#        A[A > 10] = 10
        return np.exp(A)
    A[A == 0] = 1 + 1e-5
    return np.log(A)


def identity_transform(A, _1, _2, revert=False):
    return A


def test():
    corr = from_cache("Data\\Correlations_Source[historical].pkl")
    src = 0
    k = 2
    least_similar = get_least_similar(corr[10], src, k)
    n = len(str(src)) + 4 * (k-1) + k - 1
    for path, similarity in least_similar.items():
        print("%-*s : %.4f" % (n, path, similarity))


if len(sys.argv) > 1 and sys.argv[0] == "Utility.py" and sys.argv[1] == "test":
    test()
