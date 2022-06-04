import numpy as np
import datetime as dt
import sys
import os
import time
import pandas as pd
from progressbar import ProgressBar
import Utility as util
from Container import Container
from Data.DataSelection import DataSelection


class SpatiotemporalData(Container, DataSelection):

    debug = 0

    def __init__(self, var):
        print(util.make_msg_block(" Spatiotemporal Data Initialization : Started ", "#"))
        start = time.time()
        tmp_var = Container().copy(var)
        self.misc = Miscellaneous(tmp_var)
        print("    Initialized Miscellaneous: %.3fs" % ((time.time() - start)))
        start = time.time()
        tmp_var.misc = self.misc
        self.original = Original(tmp_var)
        tmp_var.original = self.original
        print("    Initialized Original: %.3fs" % ((time.time() - start)))
        start = time.time()
        self.metrics = Metrics(tmp_var)
        tmp_var.metrics = self.metrics
        print("    Initialized Metrics: %.3fs" % ((time.time() - start)))
        start = time.time()
        tmp_var.original = self.original
        self.reduced = Reduced(tmp_var)
        tmp_var.reduced = self.reduced
        print("    Initialized Reduced: %.3fs" % ((time.time() - start)))
        start = time.time()
        self.reduced_metrics = ReducedMetrics(tmp_var)
        tmp_var.reduced_metrics = self.reduced_metrics
        print("    Initialized Reduced Metrics: %.3fs" % ((time.time() - start)))
        start = time.time()
        self.windowed = Windowed(tmp_var)
        tmp_var.windowed = self.windowed
        print("    Initialized Windowed: %.3fs" % ((time.time() - start)))
        start = time.time()
        self.transformed = Transformed(tmp_var)
        tmp_var.transformed = self.transformed
        print("    Initialized Transformed: %.3fs" % ((time.time() - start)))
        self.partitioning = var.partitioning
        self.mapping = var.mapping.merge(self.misc)
        self.loading = var.loading
        print(util.make_msg_block("Spatiotemporal Data Initialization : Completed", "#"))


class Miscellaneous(Container, DataSelection):

    def __init__(self, var):
        map_var = var.get("mapping")
        load_var = var.get("loading")
        proc_var = var.get("processing")
        self.set("feature_index_map", util.to_key_index_dict(load_var.get("feature_fields")))
        self.set("index_feature_map", util.invert_dict(self.get("feature_index_map")))
        # Init predictor features data
        for feature in map_var.get("predictor_features"):
            if not feature in self.get("feature_index_map"):
                raise ValueError("Predictor feature \"%s\" does not exist in this dataset" % (feature))
        self.set("predictor_features", map_var.get("predictor_features"))
        self.set(
            "predictor_indices", 
            np.array(util.get_dict_values(self.get("feature_index_map"), self.get("predictor_features")))
        )
        self.set("n_predictors", len(map_var.get("predictor_features")))
        # Init response features data
        for feature in map_var.get("response_features"):
            if not feature in self.get("feature_index_map"):
                raise ValueError("Response feature \"%s\" does not exist in this dataset" % (feature))
        self.set("response_features", map_var.get("response_features"))
        self.set(
            "response_indices", 
            np.array(util.get_dict_values(self.get("feature_index_map"), self.get("response_features")))
        )
        self.set("n_responses", len(map_var.get("response_features")))
        self.set("features", list(self.get("feature_index_map").keys()))
        self.set(
            "feature_indices", 
            np.array(util.get_dict_values(self.get("feature_index_map"), self.get("features")))
        )
        self.set("n_features", len(self.get("features")))
        # Set precision
        self.set("int_dtype", {16: np.int16, 32: np.int32, 64: np.int64}[load_var.get("precision")])
        self.set("float_dtype", {16: np.float16, 32: np.float32, 64: np.float64}[load_var.get("precision")])


class Original(Container, DataSelection):

    def __init__(self, var):
        misc_var = var.get("misc")
        load_var = var.get("loading")
        cache_var = var.get("caching")
        part_var = var.get("partitioning")
        struct_var = var.get("structure")
        dist_var = var.get("distribution")
        tmp_var = Container().copy([load_var, cache_var, struct_var])
        if dist_var.get("process_rank") == dist_var.get("root_process_rank"):
            self.set(
                "temporal_interval",
                self.interval_from_selection(
                    part_var.get("temporal_selection")
                )
            )
            self.set("features", self.load_features(tmp_var))
            self.set("features", self.get("features").astype(misc_var.get("float_dtype")))
            self.set("n_features", self.get("features").shape[-1])
            self.set("spatial_labels", self.load_spatial_labels(tmp_var))
            self.set(
                "spatial_indices", 
                self.indices_from_selection(
                    self.get("spatial_labels"), 
                    part_var.get("spatial_selection")
                )
            )
            self.set("n_spatial", len(self.get("spatial_labels")))
            self.set("temporal_labels", self.load_temporal_labels(tmp_var))
            self.set(
                "temporal_indices", 
                self.indices_from_selection(
                    self.get("temporal_labels"), 
                    part_var.get("temporal_selection")
                )
            )
            self.set("n_temporal", len(self.get("temporal_labels")))
            for partition in part_var.get("partitions"):
                self.set(
                    "temporal_interval",
                    self.interval_from_selection(
                        part_var.get("temporal_selection", partition)
                    ),
                    partition
                )
                self.set(
                    "temporal_indices", 
                    self.indices_from_selection(
                        self.get("temporal_labels"), 
                        part_var.get("temporal_selection", partition)
                    ),
                    partition
                )
                self.set(
                    "temporal_labels",
                    self.filter_axis(
                        self.get("temporal_labels"),
                        0,
                        self.get("temporal_indices", partition)
                    ),
                    partition
                )
                self.set(
                    "n_temporal",
                    len(self.get("temporal_labels", partition)),
                    partition
                )
                self.set(
                    "spatial_indices",
                    self.indices_from_selection(
                        self.get("spatial_labels"), 
                        part_var.get("spatial_selection", partition)
                    ),
                    partition
                )
                self.set(
                    "spatial_labels",
                    self.filter_axis(
                        self.get("spatial_labels"),
                        0,
                        self.get("spatial_indices", partition)
                    ),
                    partition
                )
                self.set(
                    "n_spatial",
                    len(self.get("spatial_labels", partition)),
                    partition
                )
                self.set(
                    "features",
                    self.filter_axis(
                        self.get("features"),
                        [0, 1],
                        [
                            self.get("temporal_indices", partition),
                            self.get("spatial_indices", partition)
                        ]
                    ),
                    partition
                )
                self.set("n_features", self.get("features", partition).shape[-1], partition)
                self.set(
                    "predictor_features", 
                    self.filter_axis(
                        self.get("features", partition), 
                        -1, 
                        misc_var.predictor_indices
                    ), 
                    partition
                )
                self.set(
                    "response_features", 
                    self.filter_axis(
                        self.get("features", partition), 
                        -1, 
                        misc_var.response_indices
                    ), 
                    partition
                )
                self.set(
                    "periodic_indices", 
                    util.temporal_labels_to_periodic_indices(
                        self.get("temporal_labels", partition),
                        load_var.get("temporal_seasonality_period"),
                        load_var.get("temporal_resolution"),
                        load_var.get("temporal_label_format")
                    ), 
                    partition
                )

    def load_features(self, var):
        from_cache, to_cache = var.get("from_cache"), var.get("to_cache")
        feature_fields = var.get("feature_fields")
        spatial_label_field = var.get("spatial_label_field")
        temporal_label_field = var.get("temporal_label_field")
        missing_value_code = var.get("missing_value_code")
        shape = var.get("shape")
        text_path = os.sep.join([var.get("data_dir"), var.get("original_text_filename")])
        cache_path = os.sep.join([var.get("cache_dir"), var.get("original_cache_filename")])
        if from_cache or to_cache:
            os.makedirs(var.get("cache_dir"), exist_ok=True)
        if os.path.exists(cache_path) and from_cache:
            features = util.from_cache(cache_path)
        elif os.path.exists(text_path):
            # Read data and determine its dimensions
            if not missing_value_code is None:
                df = pd.read_csv(text_path, na_values=missing_value_code)
            else:
                df = pd.read_csv(text_path)
            n_spatial = len(df[spatial_label_field].unique())
            n_temporal = len(df[temporal_label_field].unique())
            n_features = len(feature_fields)
            missing_values_df = df.drop(spatial_label_field, 1).isna().groupby(
                df[spatial_label_field], 
                sort=False
            ).sum().reset_index()
            path = os.sep.join([var.get("cache_dir"), "MissingValuesSummary.csv"])
            missing_values_df.to_csv(path, index=False)
            # Attempt to remove missing values
            if not all(df[feature_fields].isna().sum() == 0): # Missing data is present
                df[feature_fields] = df.groupby(spatial_label_field)[feature_fields].fillna(method="ffill")
            if not all(df[feature_fields].isna().sum() == 0): # Missing data still present
                df[feature_fields] = df.groupby(spatial_label_field)[feature_fields].fillna(method="bfill")
            if False and not all(df[feature_fields].isna().sum() == 0): # Missing data still present
                df[feature_fields] = df[feature_fields].fillna(
                    df.groupby(spatial_label_field)[feature_fields].transform("mean")
                )
            if not all(df[feature_fields].isna().sum() == 0): # Missing data still present
                df[feature_fields] = df[feature_fields].fillna(0)
            if not all(df[feature_fields].isna().sum() == 0): # Missing data still present
                raise ValueError("Could not remove all NaN values from %s" % (text_path))
            # Convert to NumPy.ndarray with shape=(n_temporal, n_spatial, n_features)
            features = df[feature_fields].to_numpy()
            if shape == ["spatial", "temporal", "feature"]:
                features = np.reshape(features, [n_spatial, n_temporal, n_features])
                features = np.swapaxes(features, 0, 1)
            elif shape == ["temporal", "spatial", "feature"]:
                features = np.reshape(features, [n_temporal, n_spatial, n_features])
            else:
                raise ValueError("Unknown shape %s given for layout of original spatiotemporal data" % (shape))
            # Handle special features: derive numerical representation for temporal and/or spatial labels
            if temporal_label_field in feature_fields:
                idx = feature_fields.index(temporal_label_field)
                features[:,:,idx] = util.temporal_labels_to_periodic_indices(
                    features[:,:,idx], 
                    var.get("temporal_seasonality_period"), 
                    var.get("temporal_resolution"), 
                    var.get("temporal_label_format")
                )
            if spatial_label_field in feature_fields:
                idx = feature_fields.index(spatial_label_field)
                features[:,:,idx] = util.labels_to_ids(features[:,:,idx])
            if to_cache:
                util.to_cache(features, cache_path)
        else:
            raise FileNotFoundError("Text file \"%s\" nor cache file \"%s\" exist" % (text_path, cache_path))
        return features

    def load_spatial_labels(self, var):
        from_cache, to_cache = var.get("from_cache"), var.get("to_cache")
        spatial_label_field = var.get("spatial_label_field")
        text_path = os.sep.join([var.get("data_dir"), var.get("original_spatial_labels_text_filename")])
        cache_path = os.sep.join([var.get("cache_dir"), var.get("original_spatial_labels_cache_filename")])
        if os.path.exists(cache_path) and from_cache:
            spatial_labels = util.from_cache(cache_path)
        elif os.path.exists(text_path):
            if "missing_value_code" in var:
                df = pd.read_csv(text_path, na_values=var.get("missing_value_code"))
            else:
                df = pd.read_csv(text_path)
            spatial_labels = df[spatial_label_field].unique().astype(str)
            if to_cache:
                util.to_cache(spatial_labels, cache_path)
        else:
            raise filenotfounderror("text file \"%s\" nor cache file \"%s\" exist" % (text_path, cache_path))
        return spatial_labels

    def load_temporal_labels(self, var):
        from_cache, to_cache = var.get("from_cache"), var.get("to_cache")
        temporal_label_field = var.get("temporal_label_field")
        text_path = os.sep.join([var.get("data_dir"), var.get("original_temporal_labels_text_filename")])
        cache_path = os.sep.join([var.get("cache_dir"), var.get("original_temporal_labels_cache_filename")])
        if os.path.exists(cache_path) and from_cache:
            temporal_labels = util.from_cache(cache_path)
        elif os.path.exists(text_path):
            if "missing_value_code" in var:
                df = pd.read_csv(text_path, na_values=var.get("missing_value_code"))
            else:
                df = pd.read_csv(text_path)
            temporal_labels = df[temporal_label_field].unique().astype(str)
            if to_cache:
                util.to_cache(temporal_labels, cache_path)
        else:
            raise filenotfounderror("text file \"%s\" nor cache file \"%s\" exist" % (text_path, cache_path))
        return temporal_labels

    def to_windows(self, A, in_length, out_length, horizon=1, stride=1, offset=0):
        in_indices, out_indices = util.input_output_window_indices(
            A.shape[0], 
            in_length, 
            out_length, 
            horizon, 
            stride, 
            offset
        )
        return self.filter_axis(A, 0, in_indices), self.filter_axis(A, 0, out_indices)

    def transform(self, features, periodic_indices, spatial_indices, feature_indices, var, revert=False):
        n_temporal, n_spatial, n_features = features.shape
        feature_labels = util.get_dict_values(var.get("index_feature_map"), feature_indices)
        transformation_resolution = var.get("transformation_resolution")
        met_var = var.get("metrics")
        mins = met_var.get("minimums")
        maxes = met_var.get("maximums")
        meds = met_var.get("medians")
        means = met_var.get("means")
        stds = met_var.get("standard_deviations")
        mins = met_var.reduce_metric_to_resolution(mins, transformation_resolution, np.min)
        maxes = met_var.reduce_metric_to_resolution(maxes, transformation_resolution, np.max)
        meds = met_var.reduce_metric_to_resolution(meds, transformation_resolution, np.median)
        means = met_var.reduce_metric_to_resolution(means, transformation_resolution, np.mean)
        stds = met_var.reduce_metric_to_resolution(stds, transformation_resolution, np.std)
        transformation_function_map = {
            "min_max": util.minmax_transform,
            "z_score": util.zscore_transform,
            "log": util.log_transform,
            "identity": util.identity_transform, 
            "none": util.identity_transform, 
        }
        empty_arg = np.zeros(mins.shape)
        transformation_metric_map = {
            "min_max": [mins, maxes],
            "z_score": [means, stds],
            "log": [empty_arg, empty_arg],
            "identity": [empty_arg, empty_arg],
            "none": [empty_arg, empty_arg], 
        }
        feature_transformation_map = util.merge_dicts(
            util.to_dict(feature_labels, var.default_feature_transformation, True), 
            var.feature_transformation_map
        )
        if "temporal" in transformation_resolution and "spatial" in transformation_resolution:
            for j in range(n_features):
                f = feature_indices[j]
                transformations = feature_transformation_map[feature_labels[j]]
                if revert:
                    transformations.reverse()
                for transformation in transformations:
                    features[:,:,j] = transformation_function_map[transformation](
                        features[:,:,j],
                        transformation_metric_map[transformation][0][periodic_indices,:,:][:,spatial_indices,f],
                        transformation_metric_map[transformation][1][periodic_indices,:,:][:,spatial_indices,f],
                        revert
                    )
        elif "temporal" in transformation_resolution:
            for j in range(n_features):
                f = feature_indices[j]
                transformations = feature_transformation_map[feature_labels[j]]
                if revert:
                    transformations.reverse()
                for transformation in transformations:
                    features[:,:,j] = transformation_function_map[transformation](
                        features[:,:,j],
                        transformation_metric_map[transformation][0][periodic_indices,f],
                        transformation_metric_map[transformation][1][periodic_indices,f],
                        revert
                    )
        elif "spatial" in transformation_resolution:
            for j in range(n_features):
                f = feature_indices[j]
                transformations = feature_transformation_map[feature_labels[j]]
                if revert:
                    transformations.reverse()
                for transformation in transformations:
                    features[:,:,j] = transformation_function_map[transformation](
                        features[:,:,j],
                        transformation_metric_map[transformation][0][spatial_indices,f],
                        transformation_metric_map[transformation][1][spatial_indices,f],
                        revert
                    )
        else:
            for j in range(n_features):
                f = feature_indices[j]
                transformations = feature_transformation_map[feature_labels[j]]
                if revert:
                    transformations.reverse()
                for transformation in transformations:
                    features[:,:,j] = transformation_function_map[transformation](
                        features[:,:,j],
                        transformation_metric_map[transformation][0][f],
                        transformation_metric_map[transformation][1][f],
                        revert
                    )
        return features


class Metrics(Container, DataSelection):

    def __init__(self, var):
        misc_var = var.get("misc")
        orig_var = var.get("original")
        load_var = var.get("loading")
        cache_var = var.get("caching")
        part_var = var.get("partitioning")
        struct_var = var.get("structure")
        dist_var = var.get("distribution")
        proc_var = var.get("processing")
        tmp_var = Container().copy([proc_var, load_var, cache_var, struct_var, misc_var])
        metric_func_map = {
            "minimums": np.min, 
            "maximums": np.max, 
            "medians": np.median, 
            "means": np.mean, 
            "standard_deviations": np.std, 
        }
        for metric_name, metric_func in metric_func_map.items():
            self.set(
                metric_name, 
                self.load_metric(
                    orig_var.filter_axis(
                        orig_var.get("features"), 
                        0, 
                        orig_var.get("temporal_indices", proc_var.metric_source_partition)
                    ), 
                    orig_var.get("temporal_labels", proc_var.metric_source_partition), 
                    orig_var.get("temporal_interval", proc_var.metric_source_partition),
                    metric_name, 
                    metric_func, 
                    tmp_var
                )
            )

    def load_metric(self, features, temporal_labels, temporal_interval, metric_name, metric_func, var):
        from_cache, to_cache = var.get("from_cache"), var.get("to_cache")
        cache_dir = var.get("cache_dir")
        cache_filename = "SpatiotemporalMetric_Type[%s]_TemporalInterval[%s].pkl" % (
            util.convert_name_convention(metric_name, "snake", "Pascal"), 
            ",".join(temporal_interval)
        )
        cache_path = os.sep.join([cache_dir, cache_filename])
        if os.path.exists(cache_path) and from_cache:
            metric = util.from_cache(cache_path)
        else:
            metric = self.compute_metric(features, temporal_labels, metric_func, var)
            if to_cache:
                util.to_cache(metric, cache_path)
        return metric

    def compute_metric(self, features, temporal_labels, metric_func, var):
        n_temporal, n_spatial, n_features = features.shape
        periodic_indices = util.temporal_labels_to_periodic_indices(
            temporal_labels, 
            var.temporal_seasonality_period, 
            var.temporal_resolution, 
            var.temporal_label_format
        )
        period_size = len(np.unique(np.reshape(periodic_indices, -1)))
        metric = np.ones((period_size, n_spatial, n_features))
        pb = ProgressBar()
        for t in pb(range(period_size)):
            periodic_mask = periodic_indices == t
            metric[t,:,:] = metric_func(features[periodic_mask], 0)
        # Handle special features
        if var.spatial_label_field in var.feature_index_map:
            idx = var.feature_index_map[var.spatial_label_field]
            if metric_func == np.min:
                metric[:,:,idx] = 0
            elif metric_func == np.max:
                metric[:,:,idx] = metric.shape[1] - 1
            elif metric_func == np.median:
                metric[:,:,idx] = metric.shape[1] // 2
            elif metric_func == np.mean:
                metric[:,:,idx] = metric.shape[1] // 2
            elif metric_func == np.std:
                metric[:,:,idx] = np.std(np.arange(metric.shape[1]))
        if var.temporal_label_field in var.feature_index_map:
            idx = var.feature_index_map[var.temporal_label_field]
            if metric_func == np.min:
                metric[:,:,idx] = 0
            elif metric_func == np.max:
                metric[:,:,idx] = metric.shape[0] - 1
            elif metric_func == np.median:
                metric[:,:,idx] = metric.shape[0] // 2
            elif metric_func == np.mean:
                metric[:,:,idx] = metric.shape[0] // 2
            elif metric_func == np.std:
                metric[:,:,idx] = np.std(np.arange(metric.shape[0]))
        if metric_func == np.std:
            metric[metric == 0] = 1
        return metric

    def reduce_metric_to_resolution(self, metric, transformation_resolution, reduction):
        if "temporal" in transformation_resolution and "spatial" in transformation_resolution:
            pass # reduce out no dimensions and proceed
        elif "temporal" in transformation_resolution: # reduce out the subbasin dimension
            metric = reduction(metric, 1)
        elif "spatial" in transformation_resolution: # reduce out the temporal dimension
            metric = reduction(metric, 0)
        else: # reduce out both temporal and spatial dimensions
            metric = reduction(metric, (0,1))
        if reduction == np.std:
            metric[metric == 0] = 1
        return metric


class Reduced(Container, DataSelection):

    def __init__(self, var):
        misc_var = var.get("misc")
        orig_var = var.get("original")
        load_var = var.get("loading")
        cache_var = var.get("caching")
        part_var = var.get("partitioning")
        struct_var = var.get("structure")
        dist_var = var.get("distribution")
        proc_var = var.get("processing")
        tmp_var = Container().copy([load_var, cache_var, struct_var, misc_var])
        if dist_var.get("process_rank") == dist_var.get("root_process_rank"):
            self.set(
                "features",
                self.load_features(
                    orig_var.get("features"),
                    orig_var.get("temporal_labels"),
                    orig_var.get("temporal_interval"),
                    proc_var.get("temporal_reduction"),
                    tmp_var
                )
            )
            self.set(
                ["n_channel", "n_temporal", "n_spatial", "n_feature"], 
                list(self.get("features").shape), 
                multi_value=True
            )
            self.set("features", self.get("features").astype(misc_var.get("float_dtype")))
            self.set("spatial_indices", orig_var.get("spatial_indices"))
            self.set("spatial_labels", orig_var.get("spatial_labels"))
            self.set("n_spatial", orig_var.get("n_spatial"))
            self.set(
                "temporal_labels",
                self.load_temporal_labels(
                    orig_var.get("features"),
                    orig_var.get("temporal_labels"),
                    orig_var.get("temporal_interval"),
                    proc_var.get("temporal_reduction"),
                    tmp_var
                )
            )
            self.set(
                "periodic_indices", 
                util.temporal_labels_to_periodic_indices(
                    self.get("temporal_labels"),
                    load_var.get("temporal_seasonality_period"),
                    load_var.get("temporal_resolution"),
                    load_var.get("temporal_label_format")
                )
            )
            for partition in part_var.get("partitions"):
                self.set("temporal_interval", orig_var.get("temporal_interval", partition), partition)
                self.set("spatial_indices", orig_var.get("spatial_indices", partition), partition)
                self.set("spatial_labels", orig_var.get("spatial_labels", partition), partition)
                self.set("n_spatial", orig_var.get("n_spatial", partition), partition)
                self.set(
                    "temporal_indices", 
                    self.get_reduced_temporal_indices(
                        part_var.get("temporal_selection", partition),
                        self.get("temporal_labels"),
                        load_var.get("temporal_resolution"), 
                        load_var.get("temporal_label_format")
                    ),
                    partition
                )
                self.set(
                    "temporal_labels",
                    self.filter_axes(
                        self.get("temporal_labels"), 
                        [0, 1], 
                        self.get("temporal_indices", partition) 
                    ), 
                    partition
                )
                self.set(
                    "features",
                    self.filter_axes(
                        self.filter_axis(
                            self.get("features"),
                            2,
                            self.get("spatial_indices", partition)
                        ),
                        [0, 1], 
                        self.get("temporal_indices", partition),
                    ),
                    partition
                )
                self.set(
                    "predictor_features", 
                    self.filter_axis(
                        self.get("features", partition), 
                        -1, 
                        misc_var.predictor_indices
                    ), 
                    partition
                )
                self.set(
                    "response_features", 
                    self.filter_axis(
                        self.get("features", partition), 
                        -1, 
                        misc_var.response_indices
                    ), 
                    partition
                )
                self.set(
                    "periodic_indices", 
                    util.temporal_labels_to_periodic_indices(
                        self.get("temporal_labels", partition),
                        load_var.get("temporal_seasonality_period"),
                        load_var.get("temporal_resolution"),
                        load_var.get("temporal_label_format")
                    ), 
                    partition
                )
                self.set(
                    ["n_channel", "n_temporal", "n_spatial", "n_feature"], 
                    list(self.get("features", partition).shape), 
                    partition, 
                    multi_value=True
                )

    def load_features(self, original, original_temporal_labels, temporal_interval, temporal_reduction, var):
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

    def load_temporal_labels(self, original, original_temporal_labels, temporal_interval, temporal_reduction, var):
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

    def reduce_original(self, original, original_temporal_labels, temporal_reduction, var):
        n_temporal, n_spatial, n_feature = original.shape
        method, factor, stride = temporal_reduction
        temporal_label_field = var.get("temporal_label_field")
        feature_index_map = var.get("feature_index_map")
        method_function_map = {
            "min": np.min, 
            "max": np.max, 
            "avg": np.mean, 
            "sum": np.sum, 
        }
        indices = self.get_temporal_reduction_indices(n_temporal, temporal_reduction)
        reduced = method_function_map[method](original[indices,:,:], 2)
        if temporal_label_field in feature_index_map:
            idx = feature_index_map[temporal_label_field]
            reduced[:,:,:,idx] = original[indices[:,:,0],:,idx]
        reduced_temporal_labels = original_temporal_labels[indices[:,:,0]]
        return reduced, reduced_temporal_labels

    def get_temporal_reduction_indices(self, n_temporal, temporal_reduction):
        factor, stride = temporal_reduction[1:]
        assert factor % stride == 0, "Temporal reduction %s cannot produce contiguous results. Factor must be divisible by stride." % (map(str, temporal_reduction))
        n_channel = temporal_reduction[1] // temporal_reduction[2]
        indices = [util.sliding_window_indices(n_temporal, factor, factor, stride*i) for i in range(n_channel)]
        lens = [len(_indices) for _indices in indices]
        truncate_idx = min(lens)
        indices = np.stack([_indices[:truncate_idx,:] for _indices in indices])
        return indices

    def get_reduced_temporal_indices(self, temporal_selection, reduced_temporal_labels, temporal_resolution, temporal_label_format):
        mode = temporal_selection[0]
        implemented_modes = ["interval", "range", "literal"]
        if mode not in implemented_modes:
            raise NotImplementedError("Given mode \"%s\" but only modes \"%s\" are implemented" % (
                mode,
                ",".join(implemented_modes))
            )
        n_channel, n_temporal = reduced_temporal_labels.shape
        temporal_indices = []
        if mode == "interval":
            for i in range(n_channel):
                start_idx = self.get_approximate_temporal_index(
                    temporal_selection[1], 
                    reduced_temporal_labels[i], 
                    temporal_resolution, 
                    temporal_label_format
                )
                end_idx = self.get_approximate_temporal_index(
                    temporal_selection[2], 
                    reduced_temporal_labels[i], 
                    temporal_resolution, 
                    temporal_label_format
                )
                temporal_indices.append(np.arange(start_idx, end_idx+1))
            lens = [len(_indices) for _indices in temporal_indices]
            truncate_idx = min(lens)
            temporal_indices = np.stack([_indices[:truncate_idx] for _indices in temporal_indices])
        elif mode == "range":
            raise NotImplementedError()
        elif mode == "literal":
            raise NotImplementedError()
        return temporal_indices

    def get_approximate_temporal_index(self, target_temporal_label, temporal_labels, temporal_resolution, temporal_label_format):
        start = 0
        end = temporal_labels.shape[0] - 1
        while end > start:
            mid = (start + end) // 2
            start_distance = self.get_temporal_distance(
                temporal_labels[start], 
                target_temporal_label,
                temporal_resolution, 
                temporal_label_format
            )
            end_distance = self.get_temporal_distance(
                temporal_labels[end], 
                target_temporal_label,
                temporal_resolution, 
                temporal_label_format
            )
            mid_distance = self.get_temporal_distance(
                temporal_labels[mid], 
                target_temporal_label,
                temporal_resolution, 
                temporal_label_format
            )
            if start_distance <= end_distance:
                end = mid
            else:
                start = mid + 1
        return start

    def get_temporal_distance(self, a, b, temporal_resolution, temporal_label_format):
        if a is "" or b is "":
            return sys.float_info.max
        date_a = dt.datetime.strptime(a, temporal_label_format)
        date_b = dt.datetime.strptime(b, temporal_label_format)
        res_delta = dt.timedelta(**{temporal_resolution[1]: temporal_resolution[0]})
        delta = np.abs(date_a - date_b)
        distance = delta // res_delta
        return distance

    def to_windows(self, A, in_length, out_length, horizon=1, stride=1, offset=0):
        in_indices, out_indices = util.input_output_window_indices(
            A.shape[1], 
            in_length, 
            out_length, 
            horizon, 
            stride, 
            offset
        )
        return self.filter_axis(A, 1, in_indices), self.filter_axis(A, 1, out_indices)

    def transform(self, features, periodic_indices, spatial_indices, feature_indices, var, revert=False):
        n_channel, n_temporal, n_spatial, n_feature = features.shape
        feature_labels = util.get_dict_values(var.get("index_feature_map"), feature_indices)
        transformation_resolution = var.get("transformation_resolution")
        met_var = var.get("metrics")
        mins = met_var.get("minimums")
        maxes = met_var.get("maximums")
        meds = met_var.get("medians")
        means = met_var.get("means")
        stds = met_var.get("standard_deviations")
        mins = met_var.reduce_metric_to_resolution(mins, transformation_resolution, np.min)
        maxes = met_var.reduce_metric_to_resolution(maxes, transformation_resolution, np.max)
        meds = met_var.reduce_metric_to_resolution(meds, transformation_resolution, np.median)
        means = met_var.reduce_metric_to_resolution(means, transformation_resolution, np.mean)
        stds = met_var.reduce_metric_to_resolution(stds, transformation_resolution, np.std)
        transformation_function_map = {
            "min_max": util.minmax_transform,
            "z_score": util.zscore_transform,
            "log": util.log_transform,
            "none": util.identity_transform
        }
        empty_arg = np.zeros(mins.shape)
        transformation_metric_map = {
            "min_max": [mins, maxes],
            "z_score": [means, stds],
            "log": [empty_arg, empty_arg],
            "none": [empty_arg, empty_arg]
        }
        feature_transformation_map = util.merge_dicts(
            util.to_dict(feature_labels, var.default_feature_transformation, True), 
            var.feature_transformation_map
        )
        if "temporal" in transformation_resolution and "spatial" in transformation_resolution:
            for j in range(n_feature):
                f = feature_indices[j]
                transformations = feature_transformation_map[feature_labels[j]]
                if revert:
                    transformations.reverse()
                for transformation in transformations:
                    features[:,:,:,j] = transformation_function_map[transformation](
                        features[:,:,:,j],
                        transformation_metric_map[transformation][0][periodic_indices,:,:][:,spatial_indices,f],
                        transformation_metric_map[transformation][1][periodic_indices,:,:][:,spatial_indices,f],
                        revert
                    )
        elif "temporal" in transformation_resolution:
            for j in range(n_feature):
                f = feature_indices[j]
                transformations = feature_transformation_map[feature_labels[j]]
                if revert:
                    transformations.reverse()
                for transformation in transformations:
                    features[:,:,:,j] = transformation_function_map[transformation](
                        features[:,:,:,j],
                        transformation_metric_map[transformation][0][periodic_indices,f],
                        transformation_metric_map[transformation][1][periodic_indices,f],
                        revert
                    )
        elif "spatial" in transformation_resolution:
            for j in range(n_feature):
                f = feature_indices[j]
                transformations = feature_transformation_map[feature_labels[j]]
                if revert:
                    transformations.reverse()
                for transformation in transformations:
                    features[:,:,:,j] = transformation_function_map[transformation](
                        features[:,:,:,j],
                        transformation_metric_map[transformation][0][spatial_indices,f],
                        transformation_metric_map[transformation][1][spatial_indices,f],
                        revert
                    )
        else:
            for j in range(n_feature):
                f = feature_indices[j]
                transformations = feature_transformation_map[feature_labels[j]]
                if revert:
                    transformations.reverse()
                for transformation in transformations:
                    features[:,:,:,j] = transformation_function_map[transformation](
                        features[:,:,:,j],
                        transformation_metric_map[transformation][0][f],
                        transformation_metric_map[transformation][1][f],
                        revert
                    )
        return features


class ReducedMetrics(Container, DataSelection):

    def __init__(self, var):
        misc_var = var.get("misc")
        orig_var = var.get("original")
        red_var = var.get("reduced")
        load_var = var.get("loading")
        cache_var = var.get("caching")
        part_var = var.get("partitioning")
        struct_var = var.get("structure")
        dist_var = var.get("distribution")
        proc_var = var.get("processing")
        tmp_var = Container().copy([proc_var, load_var, cache_var, struct_var, misc_var])
        metric_func_map = {
            "minimums": np.min, 
            "maximums": np.max, 
            "medians": np.median, 
            "means": np.mean, 
            "standard_deviations": np.std, 
        }
        for metric_name, metric_func in metric_func_map.items():
            self.set(
                metric_name, 
                self.load_metric(
                    red_var.filter_axes(
                        red_var.get("features"), 
                        [0, 1], 
                        red_var.get("temporal_indices", proc_var.metric_source_partition)
                    ), 
                    red_var.get("temporal_labels", proc_var.metric_source_partition), 
                    red_var.get("temporal_interval", proc_var.metric_source_partition),
                    proc_var.get("temporal_reduction"),
                    metric_name, 
                    metric_func, 
                    tmp_var
                )
            )

    def load_metric(self, features, temporal_labels, temporal_interval, temporal_reduction, metric_name, metric_func, var):
        from_cache, to_cache = var.get("from_cache"), var.get("to_cache")
        cache_dir = var.get("cache_dir")
        cache_filename = "SpatiotemporalMetric_Type[%s]_TemporalInterval[%s]_TemporalReduction[%s].pkl" % (
            util.convert_name_convention(metric_name, "snake", "Pascal"), 
            ",".join(temporal_interval),
            ",".join(map(str, temporal_reduction))
        )
        cache_path = os.sep.join([cache_dir, cache_filename])
        if os.path.exists(cache_path) and from_cache:
            metric = util.from_cache(cache_path)
        else:
            metric = self.compute_metric(
                features, 
                temporal_labels, 
                metric_func, 
                var
            )
            if to_cache:
                util.to_cache(metric, cache_path)
        return metric

    def compute_metric(self, features, temporal_labels, metric_func, var):
        n_channel, n_temporal, n_spatial, n_feature = features.shape
        periodic_indices = util.temporal_labels_to_periodic_indices(
            temporal_labels, 
            var.temporal_seasonality_period, 
            var.temporal_resolution, 
            var.temporal_label_format
        )
        period_size = len(np.unique(np.reshape(periodic_indices, -1)))
        metric = np.ones((period_size, n_spatial, n_feature))
        pb = ProgressBar()
        for t in pb(range(period_size)):
            periodic_mask = periodic_indices == t
            metric[t,:,:] = metric_func(features[periodic_mask], 0)
        # Handle special features
        if var.spatial_label_field in var.feature_index_map:
            idx = var.feature_index_map[var.spatial_label_field]
            if metric_func == np.min:
                metric[:,:,idx] = 0
            elif metric_func == np.max:
                metric[:,:,idx] = metric.shape[1] - 1
            elif metric_func == np.median:
                metric[:,:,idx] = metric.shape[1] // 2
            elif metric_func == np.mean:
                metric[:,:,idx] = metric.shape[1] // 2
            elif metric_func == np.std:
                metric[:,:,idx] = np.std(np.arange(metric.shape[1]))
        if var.temporal_label_field in var.feature_index_map:
            idx = var.feature_index_map[var.temporal_label_field]
            if metric_func == np.min:
                metric[:,:,idx] = 0
            elif metric_func == np.max:
                metric[:,:,idx] = metric.shape[0] - 1
            elif metric_func == np.median:
                metric[:,:,idx] = metric.shape[0] // 2
            elif metric_func == np.mean:
                metric[:,:,idx] = metric.shape[0] // 2
            elif metric_func == np.std:
                metric[:,:,idx] = np.std(np.arange(metric.shape[0]))
        if metric_func == np.std:
            metric[metric == 0] = 1
        return metric

    def reduce_metric_to_resolution(self, metric, transformation_resolution, reduction):
        if "temporal" in transformation_resolution and "spatial" in transformation_resolution:
            pass # reduce out no dimensions and proceed
        elif "temporal" in transformation_resolution: # reduce out the subbasin dimension
            metric = reduction(metric, 1)
        elif "spatial" in transformation_resolution: # reduce out the temporal dimension
            metric = reduction(metric, 0)
        else: # reduce out both temporal and spatial dimensions
            metric = reduction(metric, (0,1))
        if reduction == np.std:
            metric[metric == 0] = 1
        return metric


class Windowed(Container, DataSelection):

    def __init__(self, var):
        misc_var = var.get("misc")
        orig_var = var.get("original")
        red_var = var.get("reduced")
        met_var = var.get("metrics")
        load_var = var.get("loading")
        cache_var = var.get("caching")
        part_var = var.get("partitioning")
        proc_var = var.get("processing")
        struct_var = var.get("structure")
        dist_var = var.get("distribution")
        proc_var = var.get("processing")
        map_var = var.get("mapping")
        tmp_var = Container().copy([misc_var, proc_var])
        tmp_var.set("metrics", met_var)

    def transform(self, features, periodic_indices, spatial_indices, feature_indices, var, revert=False):
        n_windows, n_temporal, n_spatial, n_features = features.shape
        feature_labels = util.get_dict_values(var.get("index_feature_map"), feature_indices)
        transformation_resolution = var.get("transformation_resolution")
        feature_transformation_map = var.get("feature_transformation_map")
        default_feature_transformation = var.get("default_feature_transformation")
        met_var = var.get("metrics")
        mins = met_var.get("minimums")
        maxes = met_var.get("maximums")
        meds = met_var.get("medians")
        means = met_var.get("means")
        stds = met_var.get("standard_deviations")
        mins = met_var.reduce_metric_to_resolution(mins, transformation_resolution, np.min)
        maxes = met_var.reduce_metric_to_resolution(maxes, transformation_resolution, np.max)
        meds = met_var.reduce_metric_to_resolution(meds, transformation_resolution, np.median)
        means = met_var.reduce_metric_to_resolution(means, transformation_resolution, np.mean)
        stds = met_var.reduce_metric_to_resolution(stds, transformation_resolution, np.std)
        transformation_function_map = {
            "min_max": util.minmax_transform,
            "z_score": util.zscore_transform,
            "log": util.log_transform,
            "identity": util.identity_transform, 
            "none": util.identity_transform, 
        }
        empty_arg = np.zeros(mins.shape)
        transformation_metric_map = {
            "min_max": [mins, maxes],
            "z_score": [means, stds],
            "log": [empty_arg, empty_arg],
            "identity": [empty_arg, empty_arg],
            "none": [empty_arg, empty_arg], 
        }
        feature_transformation_map = util.merge_dicts(
            util.to_dict(feature_labels, var.default_feature_transformation, True), 
            var.feature_transformation_map
        )
        if "temporal" in transformation_resolution and "spatial" in transformation_resolution:
            for i in range(n_windows):
                temporal_indices = periodic_indices[i,:]
                for j in range(n_features):
                    f = feature_indices[j]
                    transformations = feature_transformation_map[feature_labels[j]]
                    if revert:
                        transformations.reverse()
                    for transformation in transformations:
                        features[i,:,:,j] = transformation_function_map[transformation](
                            features[i,:,:,j],
                            transformation_metric_map[transformation][0][temporal_indices,:,:][:,spatial_indices,f],
                            transformation_metric_map[transformation][1][temporal_indices,:,:][:,spatial_indices,f],
                            revert
                        )
        elif "temporal" in transformation_resolution:
            for i in range(n_windows):
                temporal_indices = temporal_indices[i,:]
                for j in range(n_features):
                    f = feature_indices[j]
                    transformations = feature_transformation_map[feature_labels[j]]
                    if revert:
                        transformations.reverse()
                    for transformation in transformations:
                        features[i,:,:,j] = transformation_function_map[transformation](
                            features[i,:,:,j],
                            transformation_metric_map[transformation][0][temporal_indices,f],
                            transformation_metric_map[transformation][1][temporal_indices,f],
                            revert
                        )
        elif "spatial" in transformation_resolution:
            for j in range(n_features):
                f = feature_indices[j]
                transformations = feature_transformation_map[feature_labels[j]]
                if revert:
                    transformations.reverse()
                for transformation in transformations:
                    features[:,:,:,j] = transformation_function_map[transformation](
                        features[:,:,:,j],
                        transformation_metric_map[transformation][0][spatial_indices,f],
                        transformation_metric_map[transformation][1][spatial_indices,f],
                        revert
                    )
        else:
            for j in range(n_features):
                f = feature_indices[j]
                transformations = feature_transformation_map[feature_labels[j]]
                if revert:
                    transformations.reverse()
                for transformation in transformations:
                    features[:,:,:,j] = transformation_function_map[transformation](
                        features[:,:,:,j],
                        transformation_metric_map[transformation][0][f],
                        transformation_metric_map[transformation][1][f],
                        revert
                    )
        return features


class Transformed(Container, DataSelection):

    def __init__(self, var):
        misc_var = var.get("misc")
        orig_var = var.get("original")
        met_var = var.get("metrics")
        red_var = var.get("reduced")
        redmet_var = var.get("reduced_metrics")
        load_var = var.get("loading")
        cache_var = var.get("caching")
        part_var = var.get("partitioning")
        proc_var = var.get("processing")
        struct_var = var.get("structure")
        dist_var = var.get("distribution")
        proc_var = var.get("processing")
        map_var = var.get("mapping")
        self.original, self.reduced, self.windowed = Container(), Container(), Container()
        tmp_var = Container().copy([misc_var, proc_var])
        for partition in part_var.partitions:
            tmp_var.metrics = met_var
            # Transform original data
            self.original.set(
                "features", 
                orig_var.transform(
                    orig_var.get("features", partition), 
                    orig_var.get("periodic_indices", partition), 
                    orig_var.get("spatial_indices", partition), 
                    misc_var.get("feature_indices"), 
                    tmp_var
                ), 
                partition
            )
            self.original.set(
                "predictor_features", 
                self.filter_axis(self.original.get("features", partition), -1, misc_var.predictor_indices), 
                partition
            )
            self.original.set(
                "response_features", 
                self.filter_axis(self.original.get("features", partition), -1, misc_var.response_indices), 
                partition
            )
            # Transform reduced data
            tmp_var.metrics = redmet_var
            self.reduced.set(
                "features", 
                red_var.transform(
                    red_var.get("features", partition), 
                    red_var.get("periodic_indices", partition), 
                    red_var.get("spatial_indices", partition), 
                    misc_var.get("feature_indices"), 
                    tmp_var
                ), 
                partition
            )
            self.reduced.set(
                "predictor_features", 
                self.filter_axis(self.reduced.get("features", partition), -1, misc_var.predictor_indices), 
                partition
            )
            self.reduced.set(
                "response_features", 
                self.filter_axis(self.reduced.get("features", partition), -1, misc_var.response_indices), 
                partition
            )
