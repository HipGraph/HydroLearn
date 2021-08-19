from Container import Container
from Variables import Variables
from Data import Data
from Plotting import Plotting
import os
import sys
import numpy as np
from scipy.special import kl_div
import LSTM
import Utility as util

def plot_partition_densities(spatmp, features, partitions):
    partition_color_map = {"train": "k", "valid": "r", "test": "g"}
    for feature in features:
        idx = spatmp.get("feature_index_map")[feature]
        channel_indices = spatmp.get_reduced_channel_indices(spatmp.get("reduced_n_temporal"), 1)
        reduced = spatmp.get("reduced")
        reduced = spatmp.filter_axes(
            reduced, 
            [0, 1, 2, 3], 
            [[channel_indices[0]], channel_indices[1], [0], [idx]]
        )
        n_bins = 100
        bins = np.linspace(np.min(reduced), np.max(reduced), n_bins+1, endpoint=True)
        partition_probabilities_map = {}
        for partition in partitions:
            idx = spatmp.get("feature_index_map")[feature]
            channel_indices = spatmp.get_reduced_channel_indices(spatmp.get("reduced_n_temporal", partition), 1)
            reduced = spatmp.get("reduced", partition)
            reduced = spatmp.filter_axes(
                reduced, 
                [0, 1, 2, 3], 
                [[channel_indices[0]], channel_indices[1], [0], [idx]]
            )
            freqs, _ = np.histogram(reduced, bins)
            partition_probabilities_map[partition] = freqs / np.sum(freqs)
            color = partition_color_map[partition]
            ax = plt.plot_density(reduced, color, partition, plt_median=False)
        plot_dir = var.get("common").get("plot_dir")
        path = plot_dir + os.sep + "GroundTruthDensity_Feature[%s]_Partition[%s].png" % (feature, ",".join(partitions))
        plt.save_figure(path)
        train_probs, test_probs = partition_probabilities_map["train"], partition_probabilities_map["test"]
        eps = 1e-7
        train_probs, test_probs = np.clip(train_probs, eps, 1), np.clip(test_probs, eps, 1)
        test_probs[test_probs == 0] = eps
        divergences = kl_div(train_probs, test_probs)
        print("%s KL(Train||Test) = %.4f" % (feature, np.sum(divergences)))


# Plot the distribution of features in one plot
def plot_1():
    plt = Plotting()
    var = Variables()
    var.set("spatial_selection", "literal,1,2".split(","), "train", ["historical"])
    var.set("temporal_selection", "interval,1929-01-01,2013-12-31".split(","), "train", ["historical"])
    data = Data(var)
    plt = Plotting()
    spatmp = data.get("spatiotemporal", "train")
    features = ["SWmm", "FLOW_OUTcms"]
    partition = "train"
    Y = spatmp.get("transformed_output_windowed", partition)
    print(Y.shape)
    contiguous_window_indices = spatmp.get_contiguous_window_indices(
        spatmp.get("n_temporal_out"),
        spatmp.get("n_windows", partition),
        1
    )
    spatial_labels = spatmp.get("original_spatial_labels", partition)
    response_features = spatmp.get("response_features")
    response_indices = spatmp.get("response_indices")
    feature_color_map = {"SWmm": "g", "FLOW_OUTcms": "b"}
    y_max = -sys.float_info.max
    for i in range(len(response_features)):
        color = feature_color_map[response_features[i]]
        feature_fullname = plt.feature_fullname_map[response_features[i]]
        plt.plot_density(Y[contiguous_window_indices,:,0,i], color, feature_fullname, plt_mean=True, plt_stddev=True)
    plt.legend(9)
    plt.labels(xlabel="Normalized Value")
    plt.ticks(yticks=[[], []])
    path = plt.get("plot_dir") + os.sep + "GroundTruthDensity_Features[%s].png" % (",".join(response_features))
    plt.save_figure(path)


# Plotting train, test
def plot_2():
    var = Variables()
#    data = Data(var)
#    spatmp = data.get("spatiotemporal", "train")
    features = ["SWmm", "FLOW_OUTcms"]
    partitions = ["train", "test"]
    feature_evaldir_map = {
        "SWmm": "Evaluations\\LSTM\\2370015156",
        "FLOW_OUTcms": "Evaluations\\LSTM\\5946508066",
        "SWmm": "Evaluations\\LSTM\\6158861809",
        "FLOW_OUTcms": "Evaluations\\LSTM\\4309267947",
    }
    model_name = "LSTM"
    for feature in features:
        eval_dir = feature_evaldir_map[feature]
        path = eval_dir + os.sep + "Evaluation_Partition[%s]_Subbasins[%s]_Response[%s]_Datatype[Groundtruth].pkl" % (
            "train",
            "1",
            feature
        )
        train_gt = util.from_cache(path)
        path = eval_dir + os.sep + "Evaluation_Partition[%s]_Subbasins[%s]_Response[%s]_Datatype[Prediction].pkl" % (
            "train",
            "1",
            feature
        )
        train_pred = util.from_cache(path)
        path = eval_dir + os.sep + "Evaluation_Partition[%s]_Subbasins[%s]_Response[%s]_Datatype[Groundtruth].pkl" % (
            "test",
            "1,2",
            feature
        )
        test_gt = util.from_cache(path)
        path = eval_dir + os.sep + "Evaluation_Partition[%s]_Subbasins[%s]_Response[%s]_Datatype[Prediction].pkl" % (
            "test",
            "1,2",
            feature
        )
        test_pred = util.from_cache(path)
        path = "Data" + os.sep + "SpatiotemporalMetric_Type[Minimums]_TemporalInterval[1929-01-01,1997-12-31]_TemporalReduction[avg,7,1].pkl_Source[Historical].pkl"
        mins = util.from_cache(path)
        path = "Data" + os.sep + "SpatiotemporalMetric_Type[Maximums]_TemporalInterval[1929-01-01,1997-12-31]_TemporalReduction[avg,7,1].pkl_Source[Historical].pkl"
        maxes = util.from_cache(path)
        spatial_indices = np.array([1, 2], dtype=int) - 1
        f = var.get("historical").get("feature_index_map")[feature]
        n_temporal_in = var.get("common").get("n_temporal_in")
        print("Original Shapes")
        print(train_gt.shape)
        print(train_pred.shape)
        print(test_gt.shape)
        print(test_gt.shape, np.min(test_gt[:,1]), np.max(test_gt[:,1]))
        print(test_pred.shape, np.min(test_pred[:,0,1]), np.max(test_pred[:,0,1]))
        if 0:
            train_pred = util.minmax_transform(train_pred, np.min(train_gt, axis=0), np.max(train_gt, axis=0))
            test_pred = util.minmax_transform(test_pred, np.min(test_gt, axis=0), np.max(test_gt, axis=0))
            train_gt = util.minmax_transform(train_gt, np.min(train_gt), np.max(train_gt))
            test_gt = util.minmax_transform(test_gt, np.min(test_gt, axis=0), np.max(test_gt, axis=0))
        else:
            print("Transform")
            train_gt = util.minmax_transform(
                np.squeeze(train_gt[n_temporal_in:]), 
                np.min(mins[:,0,f], axis=0), 
                np.max(maxes[:,0,f], axis=0)
            )
            train_pred = util.minmax_transform(
                np.squeeze(train_pred), 
                np.min(mins[:,0,f], axis=0), 
                np.max(maxes[:,0,f], axis=0)
            )
            test_gt = util.minmax_transform(
                np.squeeze(test_gt[n_temporal_in:]), 
                np.min(mins[:,spatial_indices,f], axis=0), 
                np.max(maxes[:,spatial_indices,f], axis=0)
            )
            test_pred = util.minmax_transform(
                np.squeeze(test_pred), 
                np.min(mins[:,spatial_indices,f], axis=0), 
                np.max(maxes[:,spatial_indices,f], axis=0)
            )
        print("New Shapes")
        print(train_gt.shape)
        print(train_pred.shape)
        print(test_gt.shape)
        print(test_pred.shape)
        plt = Plotting()
        print("Plotting")
        plt.plot_density(train_gt, "k", "Subbasin 1 - GroundTruth(Training)", plt_mean=True)
#        plt.plot_density(train_pred, "b", "Subbasin 1 - Prediction(Training)", plt_mean=True)
        plt.plot_density(test_gt[:,1], "g", "Subbasin 2 - GroundTruth(Testing)", marker="", plt_mean=True)
        plt.plot_density(test_pred[:,1], "r", "Subbasin 2 - Prediction(Testing)", marker="", plt_mean=True)
        plt.legend(9)
        plt.ticks(yticks=[[], []])
        plt.labels(xlabel="Normalized "+plt.feature_fullname_map[feature])
        path = plt.get("plot_dir") + os.sep + "PredictionVsGroundTruthDensity_Features[%s]_Model[%s].png" % (feature, model_name)
        plt.save_figure(path)



#plot_1()
plot_2()
quit()

# Temporal induction model misspecification plots
partition_gt_map, partition_pred_map = {}, {}
feature_evaldir_map = {
    "SWmm": "Evaluations\\SwatTransduction\\SW\\LSTM\\3114374638",
    "FLOW_OUTcms": "Evaluations\\SwatTransduction\\SF\\LSTM\\1025069110",
}
model_name = "LSTM"
for feature in features:
    eval_dir = feature_evaldir_map[feature]
    for partition in partitions:
        path = eval_dir + os.sep + "Evaluation_Partition[%s]_Subbasins[%s]_Response[%s]_Datatype[Groundtruth].pkl" % (
            partition,
            str(1),
            feature
        )
        partition_gt_map[partition] = util.from_cache(path)
        path = eval_dir + os.sep + "Evaluation_Partition[%s]_Subbasins[%s]_Response[%s]_Datatype[Prediction].pkl" % (
            partition,
            str(1),
            feature
        )
        partition_pred_map[partition] = util.from_cache(path)
    n_tmp_in = var.get("common").get("n_temporal_in")
    train_gt, train_pred = partition_gt_map["train"][n_tmp_in:], partition_pred_map["train"]
    test_gt, test_pred = partition_gt_map["test"][n_tmp_in:], partition_pred_map["test"]
    ax = plt.plot_density(train_gt, "k", "train", plt_median=False)
    ax = plt.plot_density(test_gt, "g", "test", plt_median=False)
#    ax = plt.plot_density(train_pred, "r", "Pred", plt_median=False)
    ax = plt.plot_density(test_pred, "r", "Pred", plt_median=False)
    plot_dir = var.get("common").get("plot_dir")
    path = plot_dir + os.sep + "ModelVsGroundTruthDensity_Model[%s]_Feature[%s]_Partition[%s].png" % (
        model_name, 
        feature, 
        ",".join(partitions)
    )
    plt.save_figure(path)
quit()
# Spatiotemporal induction model misspecification plots
partition_gt_map, partition_pred_map = {}, {}
feature_evaldir_map = {
    "SWmm": "Evaluations\\SwatTransduction\\SW\\LSTM\\3114374638",
    "FLOW_OUTcms": "Evaluations\\SwatTransduction\\SF\\LSTM\\1025069110",
}
model_name = "LSTM"
for feature in features:
    eval_dir = feature_evaldir_map[feature]
    for partition in partitions:
        path = eval_dir + os.sep + "Evaluation_Partition[%s]_Subbasins[%s]_Response[%s]_Datatype[Groundtruth].pkl" % (
            partition,
            str(1),
            feature
        )
        partition_gt_map[partition] = util.from_cache(path)
        path = eval_dir + os.sep + "Evaluation_Partition[%s]_Subbasins[%s]_Response[%s]_Datatype[Prediction].pkl" % (
            partition,
            str(1),
            feature
        )
        partition_pred_map[partition] = util.from_cache(path)
    n_tmp_in = var.get("common").get("n_temporal_in")
    train_gt, train_pred = partition_gt_map["train"][n_tmp_in:], partition_pred_map["train"]
    test_gt, test_pred = partition_gt_map["test"][n_tmp_in:], partition_pred_map["test"]
    ax = plt.plot_density(train_gt, "k", "train", plt_median=False)
    ax = plt.plot_density(test_gt, "g", "test", plt_median=False)
#    ax = plt.plot_density(train_pred, "r", "%s^{Train}" % (model_name), plt_median=False)
    ax = plt.plot_density(test_pred, "r", "%s^{Test}" % (model_name), plt_median=False)
    plot_dir = var.get("common").get("plot_dir")
    path = plot_dir + os.sep + "ModelVsGroundTruthDensity_Model[%s]_Feature[%s]_Partition[%s].png" % (
        model_name, 
        feature, 
        ",".join(partitions)
    )
    plt.save_figure(path)
