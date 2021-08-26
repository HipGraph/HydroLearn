import torch
import torch.distributed as torch_dist
import NerscDistributed as nerscdist
import numpy as np
import datetime as dt
import pickle
import sys
import os
from progressbar import ProgressBar
import re
import math
import time
import Utility as util
from Container import Container
from Plotting import Plotting


np.set_printoptions(precision=3, suppress=True, linewidth=200)
torch.set_printoptions(precision=3, sci_mode=False)
np.seterr(all="raise")


class SpatiotemporalData(Container):

    def __init__(self, var):
        print(util.make_msg_block(" Spatiotemporal Data Initialization : Started ", "#"))
        start = time.time()
        con = self.init_miscellaneous(Container(), var)
        self.set("misc", con)
        print("    Initialized Miscellaneous: %.3fs" % ((time.time() - start)))
        start = time.time()
        con = self.init_original(Container(), var)
        self.set("original", con)
        print("    Initialized Original: %.3fs" % ((time.time() - start)))
        start = time.time()
        con = self.init_reduced(Container(), var)
        self.set("reduced", con)
        print("    Initialized Reduced: %.3fs" % ((time.time() - start)))
        start = time.time()
        con = self.init_metrics(Container(), var)
        self.set("metrics", con)
        print("    Initialized Metrics: %.3fs" % ((time.time() - start)))
        start = time.time()
        con = self.init_and_distribute_windowed(Container(), var)
        self.set("windowed", con)
        print("    Initialized Windowed: %.3fs" % ((time.time() - start)))
        self.set(["partitioning", "mapping"], [var.get("partitioning"), var.get("mapping")])
        print(util.make_msg_block("Spatiotemporal Data Initialization : Completed", "#"))
        return
        plt = Plotting()
        if 0:
            for partition in self.get("partitions"):
                plt._plot_spatiotemporal("original", self, partition)
        if 1:
            for partition in self.get("partitions"):
                plt._plot_spatiotemporal("reduced", self, partition)
        if 0:
            for partition in self.get("partitions"):
                plt._plot_spatiotemporal("input_windowed", self, partition)
        if 0:
            for partition in self.get("partitions"):
                plt._plot_spatiotemporal("output_windowed", self, partition)
        if 0:
            for partition in self.get("partitions"):
                plt._plot_spatiotemporal("transformed_input_windowed", self, partition)
        if 1:
            for partition in self.get("partitions"):
                plt._plot_spatiotemporal("transformed_output_windowed", self, partition)
        quit()

    def init_miscellaneous(self, con, var):
        map_var = var.get("mapping")
        load_var = var.get("loading")
        con.set("predictor_features", map_var.get("predictor_features"))
        con.set(
            "predictor_indices", 
            util.get_dict_values(load_var.get("feature_index_map"), con.get("predictor_features"))
        )
        con.set("n_predictors", len(map_var.get("predictor_features")))
        con.set("response_features", map_var.get("response_features"))
        con.set(
            "response_indices", 
            util.get_dict_values(load_var.get("feature_index_map"), con.get("response_features"))
        )
        con.set("n_responses", len(map_var.get("response_features")))
        con.set("features", list(load_var.get("feature_index_map").keys()))
        con.set(
            "feature_indices", 
            util.get_dict_values(load_var.get("feature_index_map"), con.get("features"))
        )
        con.set("n_features", len(con.get("features")))
        con.set(
            ["feature_index_map", "index_feature_map"], 
            load_var.get(["feature_index_map", "index_feature_map"])
        )
        return con

    def init_original(self, con, var):
        load_var = var.get("loading")
        cache_var = var.get("caching")
        part_var = var.get("partitioning")
        struct_var = var.get("structure")
        dist_var = var.get("distribution")
        tmp_var = Container().copy([load_var, cache_var, struct_var])
        if dist_var.get("process_rank") == dist_var.get("root_process_rank"):
            con.set(
                "temporal_interval",
                util.get_interval_from_selection(
                    part_var.get("temporal_selection")
                )
            )
            con.set("original", self.load_original(tmp_var))
            con.set("original_temporal_labels", self.load_original_temporal_labels(tmp_var))
            con.set(
                "original_temporal_indices", 
                self.get_original_temporal_indices(
                    part_var.get("temporal_selection"),
                    con.get("original_temporal_labels")
                )
            )
            con.set("original_n_temporal", len(con.get("original_temporal_labels")))
            con.set("original_spatial_labels", self.load_original_spatial_labels(tmp_var))
            con.set(
                "original_spatial_indices", 
                self.get_original_spatial_indices(
                    part_var.get("spatial_selection"),
                    con.get("original_spatial_labels")
                )
            )
            con.set("original_n_spatial", len(con.get("original_spatial_labels")))
            for partition in part_var.get("partitions"):
                con.set(
                    "temporal_interval",
                    util.get_interval_from_selection(
                        part_var.get("temporal_selection", partition)
                    ),
                    partition
                )
                con.set(
                    "original_temporal_indices", 
                    self.get_original_temporal_indices(
                        part_var.get("temporal_selection", partition),
                        con.get("original_temporal_labels")
                    ),
                    partition
                )
                con.set(
                    "original_temporal_labels",
                    self.filter_axis(
                        con.get("original_temporal_labels"),
                        0,
                        con.get("original_temporal_indices", partition)
                    ),
                    partition
                )
                con.set(
                    "original_n_temporal",
                    len(con.get("original_temporal_labels", partition)),
                    partition
                )
                con.set(
                    "original_spatial_indices",
                    self.get_original_spatial_indices(
                        part_var.get("spatial_selection", partition),
                        con.get("original_spatial_labels")
                    ),
                    partition
                )
                con.set(
                    "original_spatial_labels",
                    self.filter_axis(
                        con.get("original_spatial_labels"),
                        0,
                        con.get("original_spatial_indices", partition)
                    ),
                    partition
                )
                con.set(
                    "original_n_spatial",
                    len(con.get("original_spatial_labels", partition)),
                    partition
                )
                con.set(
                    "original",
                    self.filter_axes(
                        con.get("original"),
                        [0, 1],
                        [
                            con.get("original_temporal_indices", partition),
                            con.get("original_spatial_indices", partition)
                        ]
                    ),
                    partition
                )
        return con

    def init_reduced(self, con, var):
        misc_var = self.get("misc")
        orig_var = self.get("original")
        load_var = var.get("loading")
        cache_var = var.get("caching")
        part_var = var.get("partitioning")
        struct_var = var.get("structure")
        dist_var = var.get("distribution")
        proc_var = var.get("processing")
        tmp_var = Container().copy([load_var, cache_var, struct_var])
        if dist_var.get("process_rank") == dist_var.get("root_process_rank"):
            con.set(
                "reduced_n_temporal_channels",
                self.compute_reduced_n_temporal_channels(proc_var.get("temporal_reduction"))
            )
            con.set(
                "reduced_n_temporal",
                self.compute_reduced_n_temporal(
                    orig_var.get("original_n_temporal"),
                    proc_var.get("temporal_reduction")
                )
            )
            con.set(
                "reduced",
                self.load_reduced(
                    orig_var.get("original"),
                    orig_var.get("original_temporal_labels"),
                    orig_var.get("temporal_interval"),
                    proc_var.get("temporal_reduction"),
                    tmp_var
                )
            )
            con.set(
                "reduced_temporal_labels",
                self.load_reduced_temporal_labels(
                    orig_var.get("original"),
                    orig_var.get("original_temporal_labels"),
                    orig_var.get("temporal_interval"),
                    proc_var.get("temporal_reduction"),
                    tmp_var
                )
            )
            con.set(
                "reduced_spatial_labels",
                np.tile(orig_var.get("original_spatial_labels"), con.get("reduced_n_temporal_channels"))
            )
            for partition in part_var.get("partitions"):
                con.set("temporal_interval", orig_var.get("temporal_interval", partition), partition)
                con.set("reduced", con.get("reduced"), partition)
                con.set(
                    "reduced_temporal_indices", 
                    self.get_reduced_temporal_indices(
                        part_var.get("temporal_selection", partition),
                        con.get("reduced_temporal_labels"),
                        con.get("reduced_n_temporal")
                    ),
                    partition
                )
                con.set(
                    "reduced_n_temporal",
                    self.compute_reduced_n_temporal_from_temporal_indices(
                        con.get("reduced_temporal_indices", partition)
                    ),
                    partition
                )
                con.set(
                    "reduced",
                    self.filter_axis(
                        con.get("reduced", partition),
                        2,
                        orig_var.get("original_spatial_indices", partition)
                    ),
                    partition
                )
                con.set(
                    "reduced",
                    self.filter_reduced(
                        con.get("reduced", partition),
                        con.get("reduced_temporal_indices", partition),
                        "temporal"
                    ),
                    partition
                )
                con.set(
                    "reduced_temporal_labels",
                    self.filter_reduced_temporal_labels(
                        con.get("reduced_temporal_labels"),
                        con.get("reduced_temporal_indices", partition),
                        "temporal"
                    ),
                    partition
                )
                con.set(
                    "reduced_spatial_labels",
                    np.tile(
                        orig_var.get("original_spatial_labels", partition), 
                        con.get("reduced_n_temporal_channels")
                    ),
                    partition
                )
        return con

    def init_metrics(self, con, var):
        misc_var = self.get("misc")
        orig_var = self.get("original")
        red_var = self.get("reduced")
        load_var = var.get("loading")
        cache_var = var.get("caching")
        proc_var = var.get("processing")
        struct_var = var.get("structure")
        dist_var = var.get("distribution")
        proc_var = var.get("processing")
        tmp_var = Container().copy([proc_var, load_var, cache_var, struct_var])
        metric_function_map = {
            "reduced_minimums": self.load_reduced_minimums,
            "reduced_maximums": self.load_reduced_maximums,
            "reduced_medians": self.load_reduced_medians,
            "reduced_means": self.load_reduced_means,
            "reduced_standard_deviations": self.load_reduced_standard_deviations
        }
        reduced, reduced_not_filtered = None, True
        if dist_var.get("process_rank") == dist_var.get("root_process_rank"):
            metric_names = list(metric_function_map.keys()) 
            i = 0
            while i < len(metric_names):
                # Try to pull from cache (since reduced is none)
                metric = metric_function_map[metric_names[i]](
                    reduced,
                    red_var.get("reduced_temporal_labels", proc_var.get("metric_source_partition")),
                    red_var.get("temporal_interval", proc_var.get("metric_source_partition")),
                    proc_var.get("temporal_reduction"),
                    tmp_var
                )
                # If not pulled from cache, get the filtered "reduced" to compute the metric.
                #   However, only get the filter reduced once since it is expensive
                if metric is None:
                    if reduced_not_filtered:
                        reduced = self.filter_reduced(
                            red_var.get("reduced"),
                            red_var.get("reduced_temporal_indices", proc_var.get("metric_source_partition")),
                            "temporal"
                        )
                        reduced_not_filtered = False
                    else:
                        raise ValueError()
                else:
                    con.set(metric_names[i], metric)
                    i += 1
        return con

    def init_and_distribute_windowed(self, con, var):
        misc_var = self.get("misc")
        orig_var = self.get("original")
        red_var = self.get("reduced")
        met_var = self.get("metrics")
        load_var = var.get("loading")
        cache_var = var.get("caching")
        part_var = var.get("partitioning")
        proc_var = var.get("processing")
        struct_var = var.get("structure")
        dist_var = var.get("distribution")
        proc_var = var.get("processing")
        map_var = var.get("mapping")
        tmp_var = Container().copy([misc_var, met_var, proc_var])
        if dist_var.get("process_rank") == dist_var.get("root_process_rank"):
            # create and distribute all windowed elements for each partition
            con.set(["n_temporal_in", "n_temporal_out"], map_var.get(["n_temporal_in", "n_temporal_out"]))
            for partition in part_var.get("partitions"):
                con.set(
                    "n_windows",
                    self.compute_n_windows(
                        red_var.get("reduced_n_temporal_channels"),
                        red_var.get("reduced_n_temporal", partition),
                        map_var.get("n_temporal_in"),
                        map_var.get("n_temporal_out")
                    ),
                    partition
                )
                # Get number of windows to send to each process. If n_processes is 1 for a partition, then all windows of that partition are assigned to just 1 process
                con.set(
                    "n_windows_per_process",
                    np.sum(con.get("n_windows", partition)) / dist_var.get("n_processes", partition),
                    partition
                )
                process_ranks = dist_var.get(
                    "nonroot_process_ranks",
                    partition
                ) + [dist_var.get("root_process_rank")]
                # Create windowed inputs + ouputs and then send them to all non-root processes.
                for process_rank in process_ranks:
                    # Create windowed inputs for current process
                    windowed, windowed_temporal_labels = self.window_reduced(
                        self.filter_axes(
                            red_var.get("reduced", partition),
                            [3],
                            [misc_var.get("predictor_indices")]
                        ),
                        red_var.get("reduced_temporal_labels", partition),
                        map_var.get("n_temporal_in"),
                        1,
                        0,
                        dist_var.get("n_processes", partition),
                        process_rank,
                        con.get("n_windows", partition),
                        con.get("n_windows_per_process", partition)
                    )
                    con.set("input_windowed", windowed, partition)
                    con.set("input_windowed_temporal_labels", windowed_temporal_labels, partition)
                    con.set(
                        "transformed_input_windowed",
                        self.transform_windowed(
                            np.copy(con.get("input_windowed", partition)),
                            [1, 2, 3],
                            util.convert_dates_to_daysofyear(
                                con.get("input_windowed_temporal_labels", partition)
                            ) - 1,
                            orig_var.get("original_spatial_indices", partition),
                            misc_var.get("predictor_indices"),
                            tmp_var
                        ),
                        partition
                    )
                    # Create windowed outputs for current process
                    windowed, windowed_temporal_labels = self.window_reduced(
                        self.filter_axes(
                            red_var.get("reduced", partition),
                            [3],
                            [misc_var.get("response_indices")]
                        ),
                        red_var.get("reduced_temporal_labels", partition),
                        map_var.get("n_temporal_out"),
                        1,
                        map_var.get("n_temporal_in"),
                        dist_var.get("n_processes", partition),
                        process_rank,
                        con.get("n_windows", partition),
                        con.get("n_windows_per_process", partition)
                    )
                    con.set("output_windowed", windowed, partition)
                    con.set("output_windowed_temporal_labels", windowed_temporal_labels, partition)
                    con.set(
                        "transformed_output_windowed",
                        self.transform_windowed(
                            np.copy(con.get("output_windowed", partition)),
                            [1, 2, 3],
                            util.convert_dates_to_daysofyear(
                                con.get("output_windowed_temporal_labels", partition)
                            ) - 1,
                            orig_var.get("original_spatial_indices", partition),
                            misc_var.get("response_indices"),
                            tmp_var
                        ),
                        partition
                    )
                    # Send results because the target process is a non-root processes
                    if process_rank != dist_var.get("root_process_rank"):
                        n_dimensions = torch.tensor([con.get("transformed_input_windowed", partition).ndim])
                        input_shape = torch.tensor(con.get("transformed_input_windowed", partition).shape)
                        output_shape = torch.tensor(con.get("transformed_output_windowed", partition).shape)
                        torch_dist.send(n_dimensions, dst=process_rank)
                        torch_dist.send(input_shape, dst=process_rank)
                        torch_dist.send(con.get("transformed_input_windowed"), dst=process_rank)
                        torch_dist.send(output_shape, dst=process_rank)
                        torch_dist.send(con.get("transformed_output_windowed"), dst=process_rank)
        else: # Receive results because I am a non-root process
            n_dimensions = torch.tensor([0])
            torch_dist.recv(n_dimensions, src=dist_var.get("root_process_rank"))
            input_shape = torch.tensor([n_dimensions[0]])
            output_shape = torch.tensor([n_dimensions[0]])
            torch_dist.recv(input_shape, src=dist_var.get("root_process_rank"))
            torch_dist.recv(output_shape, src=dist_var.get("root_process_rank"))
            transformed_input_windowed = torch.Tensor(input_shape)
            transformed_output_windowed = torch.Tensor(output_shape)
            torch_dist.recv(transformed_input_windowed, src=dist_var.get("root_process_rank"))
            torch_dist.recv(transformed_output_windowed, src=dist_var.get("root_process_rank"))
            con.set("transformed_input_windowed", util.to_ndarray(transformed_input_windowed))
            con.set("transformed_output_windowed", util.to_ndarray(transformed_output_windowed))
        return con

    def filter_axes(self, A, target_axes, filter_indices):
        if not (isinstance(target_axes, list) and isinstance(filter_indices, list)):
            raise ValueError("Target axes and filter indices must be lists")
        else:
            for filter_index in filter_indices:
                if not (isinstance(filter_index, list) or isinstance(filter_index, np.ndarray)):
                    raise ValueError("Items of filter indices must be lists")
        if len(target_axes) != len(filter_indices):
            raise ValueError("Target axes must have equal number of element in filter indices")
        for axis, indices in zip(target_axes, filter_indices):
            A = self.filter_axis(A, axis, indices)
        return A

    def filter_axis(self, A, target_axis, filter_indices):
        if not isinstance(filter_indices, int) and len(filter_indices) == 0:
            return A
        return np.swapaxes(np.swapaxes(A, 0, target_axis)[filter_indices], 0, target_axis)

    def filter_reduced(self, reduced, filter_indices, axis):
        if axis == "temporal":
            n_temporal_channels = len(filter_indices)
            max_n_temporal = -1
            n_temporal = self.compute_reduced_n_temporal_from_temporal_indices(filter_indices)
            filtered_reduced = np.ones(
                (n_temporal_channels, np.max(n_temporal), reduced.shape[2], reduced.shape[3])
            ) * -sys.float_info.max
            for i in range(n_temporal_channels):
                filtered_reduced[i,:n_temporal[i],:,:] = self.filter_axis(
                    reduced[i,:,:,:],
                    0,
                    filter_indices[i]
                )
        elif axis == "spatial":
            raise NotImplementedError()
        return filtered_reduced

    def filter_reduced_temporal_labels(self, reduced_temporal_labels, filter_indices, axis):
        n_temporal_channels = len(filter_indices)
        n_temporal = self.compute_reduced_n_temporal_from_temporal_indices(filter_indices)
        filtered_reduced_temporal_labels = np.ones(
            (n_temporal_channels, np.max(n_temporal)),
            dtype=np.object
        ) * -sys.float_info.max
        for i in range(n_temporal_channels):
            filtered_reduced_temporal_labels[i,:n_temporal[i]] = reduced_temporal_labels[i,filter_indices[i]]
        filtered_reduced_temporal_labels[filtered_reduced_temporal_labels == -sys.float_info.max] = ""
        return filtered_reduced_temporal_labels

    def compute_reduced_n_temporal_from_temporal_indices(self, reduced_temporal_indices):
        n_temporal_channels = len(reduced_temporal_indices)
        n_temporal = np.zeros((n_temporal_channels), dtype=np.int)
        for i in range(n_temporal_channels):
            n_temporal[i] = reduced_temporal_indices[i].shape[0]
        return n_temporal

    def compute_reduced_n_temporal_channels(self, temporal_reduction):
        return temporal_reduction[1] // temporal_reduction[2]

    def compute_reduced_n_temporal(self, original_n_temporal, temporal_reduction):
        reduced_n_temporal_channels = self.compute_reduced_n_temporal_channels(temporal_reduction)
        reduced_n_temporal = np.zeros((reduced_n_temporal_channels), dtype=np.int)
        for i in range(reduced_n_temporal_channels):
            reduced_n_temporal[i] = (original_n_temporal - i - 1) // reduced_n_temporal_channels + 1
        return reduced_n_temporal

    def transform_reduced(self, reduced, reduced_n_temporal, axes, reduced_temporal_indices, spatial_indices, feature_indices, var, revert=False):
        if len(reduced_temporal_indices) == 0 or len(spatial_indices) == 0 or len(feature_indices) == 0:
            return reduced
        chnl_ax, tmp_ax, spa_ax, ftr_ax = axes[0], axes[1], axes[2], axes[3]
        n_chnl, n_tmp = reduced.shape[chnl_ax], reduced.shape[tmp_ax]
        n_spa, n_ftr = reduced.shape[spa_ax], reduced.shape[ftr_ax]
        features = self.get_features_from_indices(feature_indices, var.get("index_feature_map"))
        transformation_resolution = var.get("transformation_resolution")
        feature_transformation_map = var.get("feature_transformation_map")
        mins = var.get("reduced_minimums")
        maxes = var.get("reduced_maximums")
        meds = var.get("reduced_medians")
        means = var.get("reduced_means")
        stds = var.get("reduced_standard_deviations")
        mins = self.reduce_metric_to_resolution(mins, transformation_resolution, np.min)
        maxes = self.reduce_metric_to_resolution(maxes, transformation_resolution, np.max)
        meds = self.reduce_metric_to_resolution(meds, transformation_resolution, np.median)
        means = self.reduce_metric_to_resolution(means, transformation_resolution, np.mean)
        stds = self.reduce_metric_to_resolution(stds, transformation_resolution, np.std)
        transformation_function_map = {
            "min_max": util.minmax_transform,
            "z_score": util.zscore_transform,
            "tanh": util.tanh_transform,
            "log": util.log_transform,
            "none": util.identity_transform
        }
        empty_arg = np.zeros(mins.shape)
        transformation_metric_map = {
            "min_max": [mins, maxes],
            "z_score": [means, stds],
            "tanh": [means, stds],
            "log": [empty_arg, empty_arg],
            "none": [empty_arg, empty_arg]
        }
        for feature, transformations in feature_transformation_map.items():
            for transformation in transformations:
                if transformation not in transformation_function_map:
                    raise NotImplementedError("Received unsupported transformation \"%s\" for feature \"%s\"" % (
                            transformation,
                            feature
                        )
                    )
        reduced = util.move_axes([reduced], [chnl_ax, tmp_ax, spa_ax, ftr_ax], [0, 1, 2, 3])[0]
        if "temporal" in transformation_resolution and "spatial" in transformation_resolution:
            for i in range(n_chnl):
                n = reduced_n_temporal[i]
                temporal_indices = reduced_temporal_indices[i][:n]
                for j in range(n_ftr):
                    f = feature_indices[j]
                    transformations = feature_transformation_map[features[j]]
                    if revert:
                        transformations.reverse()
                    for transformation in transformations:
                        reduced[i,:n,:,j] = transformation_function_map[transformation](
                            reduced[i,:n,:,j],
                            transformation_metric_map[transformation][0][temporal_indices,:,:][:,spatial_indices,f],
                            transformation_metric_map[transformation][1][temporal_indices,:,:][:,spatial_indices,f],
                            revert
                        )
        elif "temporal" in transformation_resolution:
            for i in range(n_chnl):
                n = reduced_n_temporal[i]
                temporal_indices = reduced_temporal_indices[i][:n]
                for j in range(n_ftr):
                    f = feature_indices[j]
                    transformations = feature_transformation_map[features[j]]
                    if revert:
                        transformations.reverse()
                    for transformation in transformations:
                        reduced[i,:n,:,j] = transformation_function_map[transformation](
                            reduced[i,:n,:,j],
                            transformation_metric_map[transformation][0][temporal_indices,f],
                            transformation_metric_map[transformation][1][temporal_indices,f],
                            revert
                        )
        elif "spatial" in transformation_resolution:
            for i in range(n_chnl):
                n = reduced_n_temporal[i]
                for j in range(n_ftr):
                    f = feature_indices[j]
                    transformations = feature_transformation_map[features[j]]
                    if revert:
                        transformations.reverse()
                    for transformation in transformations:
                        reduced[i,:n,:,j] = transformation_function_map[transformation](
                            reduced[i,:n,:,j],
                            transformation_metric_map[transformation][0][spatial_indices,f],
                            transformation_metric_map[transformation][1][spatial_indices,f],
                            revert
                        )
        else:
            for i in range(n_chnl):
                n = reduced_n_temporal[i]
                for j in range(n_ftr):
                    f = feature_indices[j]
                    transformations = feature_transformation_map[features[j]]
                    if revert:
                        transformations.reverse()
                    for transformation in transformations:
                        reduced[i,:n,:,j] = transformation_function_map[transformation](
                            reduced[i,:n,:,j],
                            transformation_metric_map[transformation][0][f],
                            transformation_metric_map[transformation][1][f],
                            revert
                        )
        reduced = util.move_axes([reduced], [0, 1, 2, 3], [chnl_ax, tmp_ax, spa_ax, ftr_ax])[0]
        return reduced

    def transform_windowed(self, windowed, axes, windowed_temporal_indices, spatial_indices, feature_indices, var, revert=False):
        # Edge case that no temporal, spatial, or feature selection was made/possible
        if len(windowed_temporal_indices) == 0 or len(spatial_indices) == 0 or len(feature_indices) == 0:
            print("I AM NOT TRANSFORMING THE WINDOWED!")
            return windowed
        temporal_axis, spatial_axis, feature_axis = axes[0], axes[1], axes[2]
        n_windows, n_features = windowed.shape[0], windowed.shape[3]
        features = self.get_features_from_indices(feature_indices, var.get("index_feature_map"))
        transformation_resolution = var.get("transformation_resolution")
        feature_transformation_map = var.get("feature_transformation_map")
        mins = var.get("reduced_minimums")
        maxes = var.get("reduced_maximums")
        meds = var.get("reduced_medians")
        means = var.get("reduced_means")
        stds = var.get("reduced_standard_deviations")
        mins = self.reduce_metric_to_resolution(mins, transformation_resolution, np.min)
        maxes = self.reduce_metric_to_resolution(maxes, transformation_resolution, np.max)
        meds = self.reduce_metric_to_resolution(meds, transformation_resolution, np.median)
        means = self.reduce_metric_to_resolution(means, transformation_resolution, np.mean)
        stds = self.reduce_metric_to_resolution(stds, transformation_resolution, np.std)
        transformation_function_map = {
            "min_max": util.minmax_transform,
            "z_score": util.zscore_transform,
            "tanh": util.tanh_transform,
            "log": util.log_transform,
            "none": util.identity_transform
        }
        empty_arg = np.zeros(mins.shape)
        transformation_metric_map = {
            "min_max": [mins, maxes],
            "z_score": [means, stds],
            "tanh": [means, stds],
            "log": [empty_arg, empty_arg],
            "none": [empty_arg, empty_arg]
        }
        for feature, transformations in feature_transformation_map.items():
            for transformation in transformations:
                if transformation not in transformation_function_map:
                    raise NotImplementedError("Received unsupported transformation \"%s\" for feature \"%s\"" % (
                            transformation,
                            feature
                        )
                    )
        windowed = np.moveaxis(windowed, [temporal_axis,spatial_axis,feature_axis], [1,2,3])
        if "temporal" in transformation_resolution and "spatial" in transformation_resolution:
            for i in range(n_windows):
                temporal_indices = windowed_temporal_indices[i,:]
                for j in range(n_features):
                    f = feature_indices[j]
                    transformations = feature_transformation_map[features[j]]
                    if revert:
                        transformations.reverse()
                    for transformation in transformations:
                        windowed[i,:,:,j] = transformation_function_map[transformation](
                            windowed[i,:,:,j],
                            transformation_metric_map[transformation][0][temporal_indices,:,:][:,spatial_indices,f],
                            transformation_metric_map[transformation][1][temporal_indices,:,:][:,spatial_indices,f],
                            revert
                        )
        elif "temporal" in transformation_resolution:
            for i in range(n_windows):
                temporal_indices = windowed_temporal_indices[i,:]
                for j in range(n_features):
                    f = feature_indices[j]
                    transformations = feature_transformation_map[features[j]]
                    if revert:
                        transformations.reverse()
                    for transformation in transformations:
                        windowed[i,:,:,j] = transformation_function_map[transformation](
                            windowed[w,:,:,j],
                            transformation_metric_map[transformation][0][temporal_indices,f],
                            transformation_metric_map[transformation][1][temporal_indices,f],
                            revert
                        )
        elif "spatial" in transformation_resolution:
            for j in range(n_features):
                f = feature_indices[j]
                transformations = feature_transformation_map[features[j]]
                if revert:
                    transformations.reverse()
                for transformation in transformations:
                    windowed[:,:,:,j] = transformation_function_map[transformation](
                        windowed[:,:,:,j],
                        transformation_metric_map[transformation][0][spatial_indices,f],
                        transformation_metric_map[transformation][1][spatial_indices,f],
                        revert
                    )
        else:
            for j in range(n_features):
                f = feature_indices[j]
                transformations = feature_transformation_map[features[j]]
                if revert:
                    transformations.reverse()
                for transformation in transformations:
                    windowed[:,:,:,j] = transformation_function_map[transformation](
                        windowed[:,:,:,j],
                        transformation_metric_map[transformation][0][f],
                        transformation_metric_map[transformation][1][f],
                        revert
                    )
        windowed = np.moveaxis(windowed, [1,2,3], [temporal_axis,spatial_axis,feature_axis])
        return windowed

    def reduce_metric_to_resolution(self, metric, transformation_resolution, reduction):
        if "temporal" in transformation_resolution and "spatial" in transformation_resolution:
            pass # reduce out no dimensions and proceed
        elif "temporal" in transformation_resolution: # reduce out the subbasin dimension
            metric = reduction(metric, axis=1)
        elif "spatial" in transformation_resolution: # reduce out the temporal dimension
            metric = reduction(metric, axis=0)
        else: # reduce out both temporal and spatial dimensions
            metric = reduction(metric, axis=(0,1))
        if reduction == np.std:
            metric[metric == 0] = 1
        return metric

    def compute_n_windows(self, reduced_n_temporal_channels, reduced_n_temporal, n_temporal_in, n_temporal_out):
        n_windows = np.zeros((reduced_n_temporal_channels), dtype=np.int)
        for set_idx in range(reduced_n_temporal_channels):
            n_windows[set_idx] = reduced_n_temporal[set_idx] - n_temporal_in - n_temporal_out + 1
        return n_windows

    # Create from a reduced "partition" a set of input and output (X, Y) model windows distributed
    #   across n_processes and map the current chunk to target_process_rank.
    #   windowed.shape=(*_n_windows_pprc, n_input_timesteps, *_n_spatial, n_predictors)
    #   windowed_temporal_labels.shape=(*_n_windows_pprc, n_output_timesteps, *_n_spatial, n_responses)
    def window_reduced(self, reduced, reduced_temporal_labels, n_window_timesteps, window_stride, window_offset, n_processes, target_process_rank, n_windows, n_windows_pprc):
        reduced_n_temporal_channels = reduced.shape[0]
        max_reduced_n_temporal = reduced.shape[1]
        n_spatial = reduced.shape[2]
        n_features = reduced.shape[3]
        p = target_process_rank
        n_windows_here = int(n_windows_pprc*(p+1)+0.5) - int(n_windows_pprc*p+0.5)
        n_windows_here = 0
        for set_idx in range(reduced_n_temporal_channels):
            set_n_windows_pprc = n_windows[set_idx] / n_processes
            set_n_windows_here = int(set_n_windows_pprc*(p+1)+0.5) - int(set_n_windows_pprc*p+0.5)
            n_windows_here += set_n_windows_here
        windowed = np.zeros((n_windows_here, n_window_timesteps, n_spatial, n_features), dtype=np.float32)
        windowed_temporal_labels = np.empty((n_windows_here, n_window_timesteps), dtype=object)
        i = 0
        for set_idx in range(reduced_n_temporal_channels):
            set_n_windows_pprc = n_windows[set_idx] / n_processes
            set_n_windows_here = int(set_n_windows_pprc*(p+1)+0.5) - int(set_n_windows_pprc*p+0.5)
            start = int(set_n_windows_pprc*p+0.5)
            end = int(set_n_windows_pprc*(p+1)+0.5)
            for w in range(int(set_n_windows_pprc*p+0.5), int(set_n_windows_pprc*(p+1)+0.5), window_stride):
                win_idx = w + window_offset
                windowed[i] = reduced[set_idx,win_idx:win_idx+n_window_timesteps,:,:]
                windowed_temporal_labels[i] = reduced_temporal_labels[set_idx,win_idx:win_idx+n_window_timesteps]
                i += 1
        return windowed, windowed_temporal_labels

    def get_temporal_interval_from_selection(self, temporal_selection):
        mode = temporal_selection[0]
        implemented_modes = ["interval", "range", "literal"]
        if mode not in implemented_modes:
            raise NotImplementedError("Given mode \"%s\" but only modes \"%s\" are implemented" % (
                mode,
                ",".join(implemented_modes))
            )
        if mode == "interval":
            temporal_interval = temporal_selection[1:]
        elif mode == "range":
            temporal_interval = temporal_selection[1:2]
        elif mode == "literal":
            sorted_temporals = sort(temporal_selection[1:])
            temporal_interval = [sorted_temporals[1], sorted_temporals[-1]]
        return temporal_interval

    def get_indices_from_features(self, features, feature_index_map):
        return np.array([feature_index_map[feature] for feature in features], dtype=np.int)

    def get_features_from_indices(self, feature_indices, index_feature_map):
        return np.array([index_feature_map[i] for i in feature_indices], dtype=np.object)

    def get_original_temporal_indices(self, temporal_selection, original_temporal_labels):
        mode = temporal_selection[0]
        implemented_modes = ["interval", "range", "literal"]
        if mode not in implemented_modes:
            raise NotImplementedError("Given mode \"%s\" but only modes \"%s\" are implemented" % (
                mode,
                ",".join(implemented_modes))
            )
        if mode == "interval":
            start_idx = np.where(original_temporal_labels == temporal_selection[1])[0][0]
            end_idx = np.where(original_temporal_labels == temporal_selection[2])[0][0]
            temporal_indices = np.arange(start_idx, end_idx+1)
        elif mode == "range":
            raise NotImplementedError()
        elif mode == "literal":
            raise NotImplementedError()
        return temporal_indices

    def get_original_spatial_indices(self, spatial_selection, original_spatial_labels):
        mode = spatial_selection[0]
        implemented_modes = ["interval", "range", "literal"]
        if mode not in implemented_modes:
            raise NotImplementedError("Given mode \"%s\" but only modes \"%s\" are implemented" % (mode, ",".join(implemented_modes)))
        if mode == "interval":
            selected_spatial_labels = np.arange(int(spatial_selection[1]), int(spatial_selection[2])+1, 1)
        elif mode == "range":
            selected_spatial_labels = np.arange(
                int(spatial_selection[1]),
                int(spatial_selection[2])+1,
                int(spatial_selection[3])
            )
        elif mode == "literal":
            selected_spatial_labels = np.array([str(i) for i in spatial_selection[1:]])
        for spatial_label in selected_spatial_labels:
            if not spatial_label in selected_spatial_labels:
                raise ValueError("Spatial label \"%s\" does not exist in original spatial labels" % (str(spatial_label)))
        spatial_indices = []
        for s in selected_spatial_labels:
            indices = np.where(original_spatial_labels == s)
            if len(indices[0]) > 0:
                spatial_indices += [indices[0][0]]
        return np.array(spatial_indices)

    def get_reduced_temporal_indices(self, temporal_selection, reduced_temporal_labels, reduced_n_temporal):
        mode = temporal_selection[0]
        implemented_modes = ["interval", "range", "literal"]
        if mode not in implemented_modes:
            raise NotImplementedError("Given mode \"%s\" but only modes \"%s\" are implemented" % (
                mode,
                ",".join(implemented_modes))
            )
        n_temporal_channels = reduced_temporal_labels.shape[0]
        temporal_indices = []
        if mode == "interval":
            for i in range(n_temporal_channels):
                start_idx = self.get_temporal_index(
                    temporal_selection[1],
                    reduced_temporal_labels[i,:reduced_n_temporal[i]]
                )
                end_idx = self.get_temporal_index(
                    temporal_selection[2],
                    reduced_temporal_labels[i,:reduced_n_temporal[i]]
                )
                temporal_indices.append(np.arange(start_idx, end_idx+1))
        elif mode == "range":
            raise NotImplementedError()
        elif mode == "literal":
            raise NotImplementedError()
        return temporal_indices

    def get_temporal_index(self, target_temporal_label, temporal_labels):
        start = 0
        end = temporal_labels.shape[0] - 1
        while end > start:
            mid = (start + end) // 2
            start_distance = self.get_temporal_distance(
                temporal_labels[start], 
                target_temporal_label,
                "date"
            )
            end_distance = self.get_temporal_distance(
                temporal_labels[end], 
                target_temporal_label,
                "date"
            )
            mid_distance = self.get_temporal_distance(
                temporal_labels[mid], 
                target_temporal_label,
                "date"
            )
            if start_distance <= end_distance:
                end = mid
            else:
                start = mid + 1
        return start

    def get_temporal_distance(self, a, b, temporal_format):
        if temporal_format is "date":
            if a is "" or b is "":
                return sys.float_info.max
            date_a = dt.datetime.strptime(a, "%Y-%m-%d")
            date_b = dt.datetime.strptime(b, "%Y-%m-%d")
            delta = np.abs(date_a - date_b)
            distance = delta.days
        else:
            raise NotImplementedError()
        return distance
    
    def get_original(self, partition, masks=["dates interval", "subbasins"]):
        original = self.get("original")
        if original is None:
            raise ValueError("Historical was None: Initialize with load_historical(...).")
        if "dates interval" in masks:
            dates_indices = self.get("dates_indices", partition)
            original = original[dates_indices,:,:]
        if "subbasins" in masks:
            subbasin_indices = self.get("subbasin_indices", partition)
            original = original[:,subbasin_indices,:]
        predictor_indices = self.get("predictor_indices")
        response_indices = self.get("response_indices")
        if "predictors + responses" in masks:
            feature_indices = np.concatenate((predictor_indices, response_indices))
            original = original[:,:,feature_indices]
        elif "predictors" in masks:
            original = original[:,:,predictor_indices]
        elif "responses" in masks:
            original = original[:,:,response_indices]
        return original

    def get_reduced_channel_indices(self, reduced_n_temporal, channel):
        channel_idx = channel - 1
        n_temporal = reduced_n_temporal[channel_idx]
        indices = np.arange(0, n_temporal)
        return (channel_idx, indices)

    def get_contiguous_window_indices(self, window_n_temporal, n_windows, channel):
        channel_idx = channel - 1
        if channel_idx == 0:
            start = 0
        else:
            start = n_windows[channel_idx-1]
        end = n_windows[channel_idx]
        step = window_n_temporal
        return np.arange(start, end, step)

    def get_XY_indices(self, partition, masks=[]):
        n_model_windows = self.get("n_model_windows", partition)
        result = re.findall("contiguous window set \d+", ",".join(masks))
        if len(result) > 0:
            set_idx = int(result[0].split()[-1]) - 1
            n_contiguous_window_sets = self.get("n_contiguous_window_sets")
            if set_idx < 0 or set_idx >= n_contiguous_window_sets:
                raise ValueError("Contiguous Window Set ID was out of range: must be in [%d,%d]." % (1,n_contiguous_window_sets))
            n_contiguous_model_windows = n_model_windows // n_contiguous_window_sets + 1
            start = n_contiguous_model_windows * set_idx
            end = n_contiguous_model_windows * (set_idx + 1)
            step = self.get("n_output_timesteps")
            contiguous_model_window_set_indices = np.arange(start, end, step)
            indices = contiguous_model_window_set_indices
        else:
            indices = np.arange(0, n_model_windows)
        return indices

    def load_original(self, var):
        from_cache, to_cache = var.get("from_cache"), var.get("to_cache")
        data_dir = var.get("data_dir")
        cache_dir = var.get("cache_dir")
        n_spatial = var.get("original_n_spatial")
        n_temporal = var.get("original_n_temporal")
        spatial_feature = var.get("spatial_feature")
        temporal_feature = var.get("temporal_feature")
        header_feature_fields = var.get("header_feature_fields")
        header_field_index_map = var.get("header_field_index_map")
        feature_index_map = var.get("feature_index_map")
        missing_value_code = var.get("missing_value_code")
        text_filename = var.get("original_text_filename")
        cache_filename = var.get("original_cache_filename")
        text_path = os.sep.join([data_dir, text_filename])
        cache_path = os.sep.join([cache_dir, cache_filename])
        if os.path.exists(cache_path) and from_cache:
            original = util.from_cache(cache_path)
        elif os.path.exists(text_path):
            with open(text_path) as f:
                lines = f.read().split("\n")[1:-1]
                n_features = len(header_feature_fields)
                original = np.zeros((n_temporal, n_spatial, n_features), dtype=np.float)
                spatial_idx = header_field_index_map[spatial_feature]
                spatial_label = None
                last_values = ["0.0" for i in range(len(lines[0].split(",")))]
                i = 0
                j = -1
                pb = ProgressBar()
                for line in pb(lines):
                    values = line.split(",")
                    if values[spatial_idx] != spatial_label:
                        spatial_label = values[spatial_idx]
                        i = 0
                        j += 1
                    for feature in header_feature_fields:
                        idx = header_field_index_map[feature]
                        k = feature_index_map[feature]
                        if feature == temporal_feature:
                            date = dt.datetime.strptime(values[idx], "%Y-%m-%d")
                            new_year_day = dt.datetime(year=date.year, month=1, day=1)
                            original[i,j,k] = (date - new_year_day).days + 1
                        else:
                            if values[idx] == missing_value_code:
                                values[idx] = last_values[idx]
                            original[i,j,k] = float(values[idx])
                    last_values = values
                    i += 1
            if to_cache:
                util.to_cache(original, cache_path)
        else:
            raise FileNotFoundError("Text file \"%s\" nor cache file \"%s\" exist" % (text_path, cache_path))
        return original

    def load_original_spatial_labels(self, var):
        from_cache, to_cache = var.get("from_cache"), var.get("to_cache")
        data_dir = var.get("data_dir")
        cache_dir = var.get("cache_dir")
        n_temporal = var.get("original_n_temporal")
        n_spatial = var.get("original_n_spatial")
        spatial_feature = var.get("spatial_feature")
        header_field_index_map = var.get("header_field_index_map")
        feature_index_map = var.get("feature_index_map")
        text_filename = var.get("original_spatial_labels_text_filename")
        cache_filename = var.get("original_spatial_labels_cache_filename")
        text_path = os.sep.join([data_dir, text_filename])
        cache_path = os.sep.join([cache_dir, cache_filename])
        if os.path.exists(cache_path) and from_cache:
            original_spatial_labels = util.from_cache(cache_path)
        elif os.path.exists(text_path):
            with open(text_path) as f:
                original_spatial_labels = np.empty((n_spatial), dtype=object)
                lines = f.read().split("\n")[1:-1]
                spatial_idx = header_field_index_map[spatial_feature]
                spatial_label = None
                j = 0
                pb = ProgressBar()
                for i in pb(range(0, len(lines), n_temporal)):
                    values = lines[i].split(",")
                    if values[spatial_idx] != spatial_label:
                        spatial_label = values[spatial_idx]
                        original_spatial_labels[j] = values[spatial_idx]
                        j += 1
                    else:
                        raise ValueError("Error in load_original_spatial_labels")
            if to_cache:
                util.to_cache(original_spatial_labels, cache_path)
        else:
            raise FileNotFoundError("Text file \"%s\" nor cache file \"%s\" exist" % (text_path, cache_path))
        return original_spatial_labels

    def load_original_temporal_labels(self, var):
        from_cache, to_cache = var.get("from_cache"), var.get("to_cache")
        data_dir = var.get("data_dir")
        cache_dir = var.get("cache_dir")
        n_temporal = var.get("original_n_temporal")
        temporal_feature = var.get("temporal_feature")
        header_field_index_map = var.get("header_field_index_map")
        feature_index_map = var.get("feature_index_map")
        text_filename = var.get("original_temporal_labels_text_filename")
        cache_filename = var.get("original_temporal_labels_cache_filename")
        text_path = os.sep.join([data_dir, text_filename])
        cache_path = os.sep.join([cache_dir, cache_filename])
        if os.path.exists(cache_path) and from_cache:
            original_temporal_labels = util.from_cache(cache_path)
        elif os.path.exists(text_path):
            with open(text_path) as f:
                original_temporal_labels = np.empty((n_temporal,), dtype=object)
                lines = f.read().split("\n")[1:n_temporal+1]
                i = 0
                pb = ProgressBar()
                for line in pb(lines):
                    idx = header_field_index_map[temporal_feature]
                    values = line.split(",")
                    original_temporal_labels[i] = values[idx]
                    i += 1
            if to_cache:
                util.to_cache(original_temporal_labels, cache_path)
        else:
            raise FileNotFoundError("Text file \"%s\" nor cache file \"%s\" exist" % (text_path, cache_path))
        return original_temporal_labels

    def load_reduced(self, original, original_temporal_labels, temporal_interval, temporal_reduction, var):
        from_cache, to_cache = var.get("from_cache"), var.get("to_cache")
        cache_dir = var.get("cache_dir")
        cache_filename = var.get("reduced_cache_filename") % (
            temporal_interval[0],
            temporal_interval[1],
            temporal_reduction[0],
            temporal_reduction[1],
            temporal_reduction[2]
        )
        cache_path = os.sep.join([cache_dir, cache_filename])
        if os.path.exists(cache_path) and from_cache:
            reduced = util.from_cache(cache_path)
        else:
            if original is None or original_temporal_labels is None:
                raise ValueError("Cannot compute reduced because original and/or original_temporal_labels was None.")
                reduced, reduced_temporal_labels = None, None
            else:
                reduced, reduced_temporal_labels = self.reduce_original(
                    original,
                    original_temporal_labels,
                    temporal_reduction,
                    var
                )
                if to_cache:
                    util.to_cache(reduced, cache_path)
        return reduced

    def load_reduced_temporal_labels(self, original, original_temporal_labels, temporal_interval, temporal_reduction, var):
        from_cache, to_cache = var.get("from_cache"), var.get("to_cache")
        cache_dir = var.get("cache_dir")
        cache_filename = var.get("reduced_temporal_labels_cache_filename") % (
            temporal_interval[0],
            temporal_interval[1],
            temporal_reduction[0],
            temporal_reduction[1],
            temporal_reduction[2]
        )
        cache_path = os.sep.join([cache_dir, cache_filename])
        if os.path.exists(cache_path) and from_cache:
            reduced_temporal_labels = util.from_cache(cache_path)
        else:
            if original is None or original_temporal_labels is None:
                raise ValueError("Cannot compute reduced_temporal_labels because original and/or original_temporal_labels was None.")
                reduced, reduced_temporal_labels = None, None
            else:
                reduced, reduced_temporal_labels = self.reduce_original(
                    original,
                    original_temporal_labels,
                    temporal_reduction,
                    var
                )
                if to_cache:
                    util.to_cache(reduced_temporal_labels, cache_path)
        return reduced_temporal_labels

    def load_reduced_minimums(self, reduced, reduced_temporal_labels, temporal_interval, timestep_reduction, var):
        from_cache, to_cache = var.get("from_cache"), var.get("to_cache")
        cache_dir = var.get("cache_dir")
        cache_filename = var.get("minimums_cache_filename") % (
            temporal_interval[0],
            temporal_interval[1],
            timestep_reduction[0],
            timestep_reduction[1],
            timestep_reduction[2]
        )
        cache_path = os.sep.join([cache_dir, cache_filename])
        if os.path.exists(cache_path) and from_cache:
            minimums = util.from_cache(cache_path)
        else:
            if reduced is None or reduced_temporal_labels is None:
                minimums = None
            else:
                minimums = self.compute_reduced_minimums(reduced, reduced_temporal_labels, var)
                if to_cache:
                    util.to_cache(minimums, cache_path)
        return minimums
    
    # Load or calculate standard deviation for each feature of each sub-basin
    def load_reduced_maximums(self, reduced, reduced_temporal_labels, temporal_interval, timestep_reduction, var):
        from_cache, to_cache = var.get("from_cache"), var.get("to_cache")
        cache_dir = var.get("cache_dir")
        cache_filename = var.get("maximums_cache_filename") % (
            temporal_interval[0],
            temporal_interval[1],
            timestep_reduction[0],
            timestep_reduction[1],
            timestep_reduction[2]
        )
        cache_path = os.sep.join([cache_dir, cache_filename])
        if os.path.exists(cache_path) and from_cache:
            maximums = util.from_cache(cache_path)
        else:
            if reduced is None or reduced_temporal_labels is None:
                maximums = None
            else:
                maximums = self.compute_reduced_maximums(reduced, reduced_temporal_labels, var)
                if to_cache:
                    util.to_cache(maximums, cache_path)
        return maximums

    # Load or calculate standard deviation for each feature of each sub-basin
    def load_reduced_medians(self, reduced, reduced_temporal_labels, temporal_interval, timestep_reduction, var):
        from_cache, to_cache = var.get("from_cache"), var.get("to_cache")
        cache_dir = var.get("cache_dir")
        cache_filename = var.get("medians_cache_filename") % (
            temporal_interval[0],
            temporal_interval[1],
            timestep_reduction[0],
            timestep_reduction[1],
            timestep_reduction[2]
        )
        cache_path = os.sep.join([cache_dir, cache_filename])
        if os.path.exists(cache_path) and from_cache:
            medians = util.from_cache(cache_path)
        else:
            if reduced is None or reduced_temporal_labels is None:
                medians = None
            else:
                medians = self.compute_reduced_medians(reduced, reduced_temporal_labels, var)
                if to_cache:
                    util.to_cache(medians, cache_path)
        return medians

    # Load or calculate mean for each feature of each sub-basin
    def load_reduced_means(self, reduced, reduced_temporal_labels, temporal_interval, timestep_reduction, var):
        from_cache, to_cache = var.get("from_cache"), var.get("to_cache")
        cache_dir = var.get("cache_dir")
        cache_filename = var.get("means_cache_filename") % (
            temporal_interval[0],
            temporal_interval[1],
            timestep_reduction[0],
            timestep_reduction[1],
            timestep_reduction[2]
        )
        cache_path = os.sep.join([cache_dir, cache_filename])
        if os.path.exists(cache_path) and from_cache:
            means = util.from_cache(cache_path)
        else:
            if reduced is None or reduced_temporal_labels is None:
                means = None
            else:
                means = self.compute_reduced_means(reduced, reduced_temporal_labels, var)
                if to_cache:
                    util.to_cache(means, cache_path)
        return means
    
    # Load or calculate standard deviation for each feature of each sub-basin
    def load_reduced_standard_deviations(self, reduced, reduced_temporal_labels, temporal_interval, timestep_reduction, var):
        from_cache, to_cache = var.get("from_cache"), var.get("to_cache")
        cache_dir = var.get("cache_dir")
        cache_filename = var.get("standard_deviations_cache_filename") % (
            temporal_interval[0],
            temporal_interval[1],
            timestep_reduction[0],
            timestep_reduction[1],
            timestep_reduction[2]
        )
        cache_path = os.sep.join([cache_dir, cache_filename])
        if os.path.exists(cache_path) and from_cache:
            standard_deviations = util.from_cache(cache_path)
        else:
            if reduced is None or reduced_temporal_labels is None:
                standard_deviations = None
            else:
                standard_deviations = self.compute_reduced_standard_deviations(
                    reduced, 
                    reduced_temporal_labels, 
                    var
                )
                if to_cache:
                    util.to_cache(standard_deviations, cache_path)
        return standard_deviations

    # Reduce original times-steps (number of days) by a factor of temporal_reduction_factor
    def reduce_original(self, original, original_temporal_labels, temporal_reduction, var):
        n_spatial = original.shape[1]
        n_features = original.shape[2]
        method = temporal_reduction[0]
        factor = temporal_reduction[1]
        stride = temporal_reduction[2]
        feature_index_map = var.get("feature_index_map")
        reduced_n_temporal_channels = self.compute_reduced_n_temporal_channels(temporal_reduction)
        reduced_n_temporal = self.compute_reduced_n_temporal(original.shape[0], temporal_reduction)
        max_reduced_n_temporal = np.max(reduced_n_temporal)
        reduced = np.ones((reduced_n_temporal_channels, max_reduced_n_temporal, n_spatial, n_features)) * -sys.float_info.max
        reduced_temporal_labels = np.ones(
            (reduced_n_temporal_channels, max_reduced_n_temporal),
            dtype=np.object
        ) * -sys.float_info.max
        method_function_map = {
            "min": np.min,
            "max": np.max,
            "avg": np.mean
        }
        pb = ProgressBar()
        for set_idx in pb(range(reduced_n_temporal_channels)):
            for t in range(reduced_n_temporal[set_idx]):
                window_start = factor * t + set_idx
                window_end = window_start + factor
                try:
                    reduced[set_idx,t] = method_function_map[method](original[window_start:window_end], axis=0)
                    idx = feature_index_map["date"]
                    reduced[set_idx,t,:,idx] = method_function_map["min"](
                        original[window_start:window_end,:,idx],
                        axis=0
                    )
                except:
                    pass
                reduced_temporal_labels[set_idx,t] = original_temporal_labels[window_start]
        reduced_temporal_labels[reduced_temporal_labels == -sys.float_info.max] = ""
        return reduced, reduced_temporal_labels
    
    # Preconditions:
    #   reduced.shape = (reduced_n_temporal_channels, max_reduced_n_temporal, n_spatial, n_features)
    #   reduced_dates.shape = (reduced_n_temporal_channels, max_reduced_n_temporal)
    def compute_reduced_minimums(self, reduced, reduced_dates, var):
        n_daysofyear = var.get("n_daysofyear")
        feature_index_map = var.get("feature_index_map")
        n_spatial = reduced.shape[2]
        n_features = reduced.shape[3]
        reduced_daysofyear = util.convert_dates_to_daysofyear(reduced_dates)
        minimums = np.ones((n_daysofyear, n_spatial, n_features))
        pb = ProgressBar()
        for t in pb(range(n_daysofyear)):
            dayofyear_mask = reduced_daysofyear == t+1
            minimums[t,:,:] = np.min(reduced[dayofyear_mask], axis=0)
        idx = feature_index_map["date"]
        minimums[:,:,idx] = 1
        return minimums

    # Preconditions:
    #   reduced.shape = (reduced_n_temporal_channels, max_reduced_n_temporal, n_spatial, n_features)
    #   reduced_dates.shape = (reduced_n_temporal_channels, max_reduced_n_temporal)
    def compute_reduced_maximums(self, reduced, reduced_dates, var):
        n_daysofyear = var.get("n_daysofyear")
        feature_index_map = var.get("feature_index_map")
        n_spatial = reduced.shape[2]
        n_features = reduced.shape[3]
        reduced_daysofyear = util.convert_dates_to_daysofyear(reduced_dates)
        maximums = np.ones((n_daysofyear, n_spatial, n_features))
        pb = ProgressBar()
        for t in pb(range(n_daysofyear)):
            dayofyear_mask = reduced_daysofyear == t+1
            maximums[t,:,:] = np.max(reduced[dayofyear_mask], axis=0)
        idx = feature_index_map["date"]
        maximums[:,:,idx] = n_daysofyear
        return maximums
    
    # Preconditions:
    #   reduced.shape = (reduced_n_temporal_channels, max_reduced_n_temporal, n_spatial, n_features)
    #   reduced_dates.shape = (reduced_n_temporal_channels, max_reduced_n_temporal)
    def compute_reduced_medians(self, reduced, reduced_dates, var):
        n_daysofyear = var.get("n_daysofyear")
        feature_index_map = var.get("feature_index_map")
        n_spatial = reduced.shape[2]
        n_features = reduced.shape[3]
        reduced_daysofyear = util.convert_dates_to_daysofyear(reduced_dates)
        medians = np.ones((n_daysofyear, n_spatial, n_features))
        pb = ProgressBar()
        for t in pb(range(n_daysofyear)):
            dayofyear_mask = reduced_daysofyear == t+1
            medians[t,:,:] = np.median(reduced[dayofyear_mask], axis=0)
        idx = feature_index_map["date"]
#        medians[:,:,idx] = 365.2425 / 2
        return medians
    
    # Preconditions:
    #   reduced.shape = (reduced_n_temporal_channels, max_reduced_n_temporal, n_spatial, n_features)
    #   reduced_dates.shape = (reduced_n_temporal_channels, max_reduced_n_temporal)
    def compute_reduced_means(self, reduced, reduced_dates, var):
        n_daysofyear = var.get("n_daysofyear")
        feature_index_map = var.get("feature_index_map")
        n_spatial = reduced.shape[2]
        n_features = reduced.shape[3]
        reduced_daysofyear = util.convert_dates_to_daysofyear(reduced_dates)
        means = np.ones((n_daysofyear, n_spatial, n_features))
        pb = ProgressBar()
        for t in pb(range(n_daysofyear)):
            dayofyear_mask = reduced_daysofyear == t+1
            means[t,:,:] = np.mean(reduced[dayofyear_mask], axis=0)
        idx = feature_index_map["date"]
#        means[:,:,idx] = 365.2425 / 2
        return means

    # Preconditions:
    #   reduced.shape = (reduced_n_temporal_channels, max_reduced_n_temporal, n_spatial, n_features)
    #   reduced_dates.shape = (reduced_n_temporal_channels, max_reduced_n_temporal)
    def compute_reduced_standard_deviations(self, reduced, reduced_dates, var):
        n_daysofyear = var.get("n_daysofyear")
        feature_index_map = var.get("feature_index_map")
        n_spatial = reduced.shape[2]
        n_features = reduced.shape[3]
        reduced_daysofyear = util.convert_dates_to_daysofyear(reduced_dates)
        standard_deviations = np.ones((n_daysofyear, n_spatial, n_features))
        pb = ProgressBar()
        for t in pb(range(n_daysofyear)):
            dayofyear_mask = reduced_daysofyear == t+1
            standard_deviations[t,:,:] = np.std(reduced[dayofyear_mask], axis=0)
        idx = feature_index_map["date"]
#        standard_deviations[:,:,idx] = np.std(np.arange(n_daysofyear) + 1)
        standard_deviations[standard_deviations == 0] = 1
        return standard_deviations
