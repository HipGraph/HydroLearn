import glob
import os
import sys
import numpy as np
import Utility as util
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sbn
from Plotting import Plotting


def get_subbasin_errors(lines, i):
    subbasin_errors = []
    while not "Subbasin" in lines[i]:
        i += 1
    while "Subbasin" in lines[i]:
        fields = lines[i].split()
        subbasin_errors += [[int(fields[1]), float(fields[-1])]]
        i += 1
    return subbasin_errors, i

def get_errors_from_report(report_str):
    lines = report_str.split("\n")
    train_subbasin_errors = []
    valid_subbasin_errors = []
    test_subbasin_errors = []
    i = 0
    train_errors, i = get_subbasin_errors(lines, i)
    valid_errors, i = get_subbasin_errors(lines, i)
    test_errors, i = get_subbasin_errors(lines, i)
    return train_errors, valid_errors, test_errors


def get_all_errors(directory):
    if directory[-1] == os.sep:
        directory = directory[:-1]
    paths = glob.glob(directory + os.sep + "*" + os.sep + "*Errors.txt")
    train_errors, valid_errors, test_errors = [], [], []
    for path in paths:
        fields = path.split(os.sep)
        model_id = fields[-2]
        with open(path, "r") as f:
            error_report_str = f.read()
        train_err, valid_err, test_err = get_errors_from_report(error_report_str)
        train_errors += [train_err]
        valid_errors += [valid_err]
        test_errors += [test_err]
    return train_errors, valid_errors, test_errors


if len(sys.argv) < 2:
    print("Give me checkpoint or evaluation directory!")
    quit()
train_errors, valid_errors, test_errors = [], [], []
directories = sys.argv[1:]
for directory in directories:
    train_errs, valid_errs, test_errs = get_all_errors(directory)
    train_errors += [train_errs]
    valid_errors += [valid_errs]
    test_errors += [test_errs]
configurations = []
print("TRAINING ERRORS =")
for err in train_errors:
    print(err)
print("VALIDATION ERRORS =")
for err in valid_errors:
    print(err)
print("TESTING ERRORS =")
for err in test_errors:
    print(err)
print("#######################")
plt = Plotting()
if len(directories) > 1:
    if "TemporalInduction" in directories[0]:
        test_errs = np.array([err[-1] for errors in test_errors[0] for err in errors])
        plt.plot_density(test_errs, "g", "Soil Moisture", band_width=3e-3, plt_mean=True, plt_stddev=True)
        test_errs = np.array([err[-1] for errors in test_errors[1] for err in errors])
        plt.plot_density(test_errs, "b", "Streamflow", band_width=3e-3, plt_mean=True, plt_stddev=True)
    elif "SpatiotemporalInduction" in directories[0]:
        test_errs = np.array([err[-1] for errors in test_errors[0] for err in errors])
        plt.plot_density(test_errs, "g", "Soil Moisture", band_width=3.3e-3, plt_mean=True, plt_stddev=True)
        test_errs = np.array([err[-1] for errors in test_errors[1] for err in errors])
        plt.plot_density(test_errs, "b", "Streamflow", band_width=9e-3, plt_mean=True, plt_stddev=True)
    plt.legend(9)
    plt.ticks(yticks=[[], []])
    plt.labels(xlabel="NRMSE")
    experiment = directories[0].split(os.sep)[-3]
    path = plt.get("plot_dir") + os.sep + "PredictionErrorDensity_Features[%s]_Experiment[%s].png" % (
        "SWmm,FLOW_OUTcms",
        experiment
    )
    plt.save_figure(path)
    quit()
elif "Transduction" in directories[0]: # Transductive results
    train_errors, valid_errors, test_errors = train_errors[0], valid_errors[0], test_errors[0]
    n = len(train_errors)
    train, valid, test = np.zeros([n]), np.zeros([n]), np.zeros([n])
    subbasins = []
    for train_err, valid_err, test_err in zip(train_errors, valid_errors, test_errors):
        train[train_err[0][0] - 1] = train_err[0][-1]
        valid[valid_err[0][0] - 1] = valid_err[0][-1]
        test[test_err[0][0] - 1] = test_err[0][-1]
elif "Induction" in directories[0]: # Inductive results
    train_errors, valid_errors, test_errors = train_errors[0], valid_errors[0], test_errors[0]
    train_subbasin_index_map, valid_subbasin_index_map, test_subbasin_index_map = {}, {}, {}
    i = 0
    for train_err in train_errors:
        subbasin, err = train_err[0][0], train_err[0][-1]
        if subbasin not in train_subbasin_index_map:
            train_subbasin_index_map[subbasin] = i
            i += 1
    train_subbasin_index_map = util.sort_dictionary(train_subbasin_index_map, "key")
    i = 0
    for key in train_subbasin_index_map.keys():
        train_subbasin_index_map[key] = i
        i += 1
    i = 0
    for valid_err in valid_errors:
        subbasin, err = valid_err[0][0], valid_err[0][-1]
        if subbasin not in valid_subbasin_index_map:
            valid_subbasin_index_map[subbasin] = i
            i += 1
    valid_subbasin_index_map = util.sort_dictionary(valid_subbasin_index_map, "key")
    i = 0
    for key in valid_subbasin_index_map.keys():
        valid_subbasin_index_map[key] = i
        i += 1
    i = 0
    for test_err in test_errors:
        for subbasin_err in test_err:
            subbasin, err = subbasin_err[0], subbasin_err[-1]
            if subbasin not in test_subbasin_index_map:
                test_subbasin_index_map[subbasin] = i
                i += 1
    test_subbasin_index_map = util.sort_dictionary(test_subbasin_index_map, "key")
    i = 0
    for key in test_subbasin_index_map.keys():
        test_subbasin_index_map[key] = i
        i += 1
    print(train_subbasin_index_map)
    print(valid_subbasin_index_map)
    print(test_subbasin_index_map)
    print(list(train_subbasin_index_map.keys()))
    print(list(valid_subbasin_index_map.keys()))
    print(list(test_subbasin_index_map.keys()))
    train_n = len(train_subbasin_index_map.keys())
    valid_n = len(valid_subbasin_index_map.keys())
    test_n = len(test_subbasin_index_map.keys())
    train, valid, test = np.zeros([train_n]), np.zeros([valid_n]), np.zeros([train_n, test_n])
    for train_err, valid_err, test_err in zip(train_errors, valid_errors, test_errors):
        train_idx = train_subbasin_index_map[train_err[0][0]]
        train[train_idx] = train_err[0][-1]
        valid_idx = valid_subbasin_index_map[valid_err[0][0]]
        valid[valid_idx] = valid_err[0][-1]
        print("###########################################")
        for subbasin_err in test_err:
            subbasin, err = subbasin_err[0], subbasin_err[-1]
            test_idx = test_subbasin_index_map[subbasin]
            print(train_err[0][0], subbasin, "|", train_idx, test_idx, "|", err)
            test[train_idx,test_idx] = err
    mat = test
    xlabel, ylabel = "Testing Subbasin", "Training Subbasin"
    xtick_labels, ytick_labels = list(test_subbasin_index_map.keys()), list(train_subbasin_index_map.keys())
    path = directory + os.sep + "InductionHeatmap.png"
    plt.plot_heatmap(mat, xtick_labels, ytick_labels, xlabel, ylabel, "Testing Set NRMSE", path=path, transpose=False)
    path = directory + os.sep + "InductionClusterHeatmap.png"
    plt.plot_cluster_heatmap(mat, xtick_labels, ytick_labels, xlabel, ylabel, path=path, plot_numbers=False)
    path = directory + os.sep + "InductionHistogram.png"
print("Training Errors")
print(pd.DataFrame(train[train > 0]).describe())
print("Validation Errors")
print(pd.DataFrame(valid[valid > 0]).describe())
print("Testing Errors")
print(pd.DataFrame(test[test > 0]).describe())
feature = directory.split(os.sep)[-2]
experiment = directory.split(os.sep)[-3]
print(feature, experiment)
path = directory + os.sep + "ErrorDensity_Feature[%s]_Experiment[%s].png" % (feature, experiment)
plt.plot_density(test, source="NRMSE")#, plt_median=False)
plt.lims(y_lim=[0, None])
plt.save_figure(path)
quit()
path = directory + os.sep + "Errors_Partition[%s].pkl" % ("train")
util.to_cache(train, path)
path = directory + os.sep + "Errors_Partition[%s].pkl" % ("valid")
util.to_cache(valid, path)
path = directory + os.sep + "Errors_Partition[%s].pkl" % ("test")
util.to_cache(test, path)
