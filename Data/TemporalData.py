import numpy as np
import datetime as dt
import sys
import os
import time
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import Utility as util
from Container import Container
from Data.DataSelection import DataSelection
from Data import Imputation


class TemporalData(Container, DataSelection):

    debug = 0

    def __init__(self, var):
        tmp_var = Container().copy(var)
        print(util.make_msg_block(" Temporal Data Initialization : Started "))
        start = time.time()
        self.misc = Miscellaneous(tmp_var)
        tmp_var.misc = self.misc
        print("    Initialized Miscellaneous: %.3fs" % ((time.time() - start)))
        start = time.time()
        self.original = Original(tmp_var)
        tmp_var.original = self.original
        print("    Initialized Original: %.3fs" % ((time.time() - start)))
        start = time.time()
        self.statistics = Statistics(tmp_var)
        tmp_var.statistics = self.statistics
        print("    Initialized Statistics: %.3fs" % ((time.time() - start)))
        start = time.time()
        self.windowed = Windowed(tmp_var)
        tmp_var.windowed = self.windowed
        print("    Initialized Windowed: %.3fs" % ((time.time() - start)))
        start = time.time()
        self.transformed = Transformed(tmp_var)
        tmp_var.transformed = self.transformed
        print("    Initialized Transformed: %.3fs" % ((time.time() - start)))
        print(util.make_msg_block("Temporal Data Initialization : Completed"))
        self.partitioning = Container().copy(tmp_var.partitioning)
#        self.loading = Container().copy(tmp_var.loading)


class Miscellaneous(Container, DataSelection):

    def __init__(self, var):
        map_var = var.mapping
        load_var = var.loading
        proc_var = var.processing
        self.feature_index_map = util.to_key_index_dict(load_var.feature_fields)
        self.index_feature_map = util.invert_dict(self.feature_index_map)
        # Init global feature labels
        self.features = list(self.feature_index_map.keys())
        self.feature_indices = np.array(util.get_dict_values(self.feature_index_map, self.features))
        self.n_feature = len(self.features)
        # Init predictor feature labels
        if map_var.temporal.predictor_features is None:
            map_var.temporal.predictor_features = self.features
        for feature in map_var.temporal.predictor_features:
            if not feature in self.feature_index_map:
                raise ValueError("Predictor feature \"%s\" does not exist in this dataset" % (feature))
        self.predictor_features = map_var.temporal.predictor_features
        self.predictor_indices = np.array(util.get_dict_values(self.feature_index_map, self.predictor_features))
        self.n_predictor = len(self.predictor_features)
        # Init response feature labels
        if map_var.temporal.response_features is None:
            map_var.temporal.response_features = self.features
        for feature in map_var.temporal.response_features:
            if not feature in self.feature_index_map:
                raise ValueError("Response feature \"%s\" does not exist in this dataset" % (feature))
        self.response_features = map_var.temporal.response_features
        self.response_indices = np.array(util.get_dict_values(self.feature_index_map, self.response_features))
        self.n_response = len(self.response_features)
        # Init other feature labels
        self.categorical_features = load_var.categorical_fields
        self.numerical_features = util.list_subtract(self.features, self.categorical_features)
        # Set precision
        self.int_dtype = {16: np.int16, 32: np.int32, 64: np.int64}[load_var.precision]
        self.float_dtype = {16: np.float16, 32: np.float32, 64: np.float64}[load_var.precision]
        # Set loading vars
        self.temporal_label_field = load_var.temporal_label_field
        self.temporal_resolution = load_var.temporal_resolution
        self.temporal_seasonality_period = load_var.temporal_seasonality_period
        self.temporal_label_format = load_var.temporal_label_format


class Original(Container, DataSelection):

    def __init__(self, var):
        # Unpack vars
        misc_var = var.misc
        load_var = var.loading
        cache_var = var.caching
        part_var = var.partitioning
        struct_var = var.structure
        dist_var = var.distribution
        # Load/init data
        tmp_var = Container().copy([load_var, cache_var, struct_var])
        #   Temporal label data
        self.temporal_interval = self.interval_from_selection(part_var.temporal_selection)
        self.temporal_labels = self.load_temporal_labels(tmp_var)
        self.periodic_indices = util.temporal_labels_to_periodic_indices(
            self.temporal_labels,
            load_var.temporal_seasonality_period,
            load_var.temporal_resolution,
            load_var.temporal_label_format
        )
        self.period_size = np.unique(self.periodic_indices).shape[0]
        #   Feature data
        self.features = self.load_features(tmp_var).astype(misc_var.float_dtype)
        self.n_temporal, self.n_feature = self.features.shape
        # Filter/init data for each partition
        for partition in part_var.partitions:
            #   Temporal label data
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
                    self.temporal_labels, 
                    part_var.get("temporal_selection", partition)
                ),
                partition
            )
            self.set(
                "temporal_labels",
                self.filter_axis(
                    self.temporal_labels,
                    0,
                    self.get("temporal_indices", partition)
                ),
                partition
            )
            self.set(
                "periodic_indices", 
                util.temporal_labels_to_periodic_indices(
                    self.get("temporal_labels", partition),
                    load_var.temporal_seasonality_period,
                    load_var.temporal_resolution,
                    load_var.temporal_label_format
                ), 
                partition
            )
            #   Feature data
            self.set(
                "features",
                self.filter_axis(
                    self.features,
                    0,
                    self.get("temporal_indices", partition),
                ),
                partition
            )
            self.n_temporal = self.get("features", partition).shape[0]

    def load_features(self, var):
        text_path = os.sep.join([var.data_dir, var.original_text_filename])
        cache_path = os.sep.join([var.cache_dir, var.original_cache_filename])
        if var.from_cache or var.to_cache:
            os.makedirs(var.cache_dir, exist_ok=True)
        if os.path.exists(cache_path) and var.from_cache:
            features = util.from_cache(cache_path)
        elif os.path.exists(text_path):
            # Read data and determine its dimensions
            df = pd.read_csv(
                text_path, usecols=[temporal_label_field] + feature_fields, 
                dtype=var.dtypes, 
                na_values=var.missing_value_code
            )
            n_temporal = len(df[var.temporal_label_field])
            n_feature = len(var.feature_fields)
            # Handle missing values
            missing_df = Imputation.missing_values(df, "temporal", var)
            path = os.sep.join([var.cache_dir, "MissingValues_Data[Temporal].csv"])
            missing_values_df.to_csv(path, index=False)
            df = Imputation.impute(df, "temporal", "median", var)
            # Convert to NumPy.ndarray with shape=(n_temporal, n_feature)
            features = df[var.feature_fields].to_numpy()
            # Handle special features: derive numerical representation for temporal and/or spatial labels
            if temporal_label_field in feature_fields:
                idx = feature_fields.index(var.temporal_label_field)
                features[:,idx] = util.temporal_labels_to_periodic_indices(
                    features[:,idx], 
                    var.temporal_seasonality_period, 
                    var.temporal_resolution, 
                    var.temporal_label_format
                )
            if var.to_cache:
                util.to_cache(features, cache_path)
        else:
            raise FileNotFoundError("Text file \"%s\" nor cache file \"%s\" exist" % (text_path, cache_path))
        return features

    def load_temporal_labels(self, var):
        text_path = os.sep.join([var.data_dir, var.original_temporal_labels_text_filename])
        cache_path = os.sep.join([var.cache_dir, var.original_temporal_labels_cache_filename])
        if var.from_cache or var.to_cache:
            os.makedirs(var.cache_dir, exist_ok=True)
        if os.path.exists(cache_path) and var.from_cache:
            temporal_labels = util.from_cache(cache_path)
        elif os.path.exists(text_path):
            df = pd.read_csv(text_path, dtype=var.dtypes, na_values=var.missing_value_code)
            temporal_labels = df[var.temporal_label_field].astype(str).unique()
            if var.to_cache:
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

    def transform(self, features, periodic_indices, feature_indices, var, revert=False):
        # Unpack vars
        n_temporal, n_feature = features.shape
        feature_labels = util.get_dict_values(var.index_feature_map, feature_indices)
        transform_resolution = var.temporal.transform_resolution
        feature_transform_map = var.temporal.feature_transform_map
        default_feature_transform = var.temporal.default_feature_transform
        # Prepare statistics
        mins = var.statistics.minimums
        maxes = var.statistics.maximums
        meds = var.statistics.medians
        means = var.statistics.means
        stds = var.statistics.standard_deviations
        mins = var.statistics.reduce_statistic_to_resolution(mins, transform_resolution, np.min)
        maxes = var.statistics.reduce_statistic_to_resolution(maxes, transform_resolution, np.max)
        meds = var.statistics.reduce_statistic_to_resolution(meds, transform_resolution, np.median)
        means = var.statistics.reduce_statistic_to_resolution(means, transform_resolution, np.mean)
        stds = var.statistics.reduce_statistic_to_resolution(stds, transform_resolution, np.std)
        if var.temporal_label_field in feature_labels:
            idx = feature_labels.index(var.temporal_label_field)
            if len(stds.shape) == 1:
                stds[idx] = var.statistics.minimums.shape[0] / 2 / 3
        # Setup for transform function calls
        empty_arg = np.zeros(mins.shape)
        transform_args_map = {
            "minmax": [mins, maxes],
            "zscore": [means, stds],
            "log": [empty_arg, empty_arg],
            "root": [empty_arg, empty_arg],
            "identity": [empty_arg, empty_arg],
        }
        if default_feature_transform is None:
            default_feature_transform = var.default_feature_transform
        feature_transform_map = util.merge_dicts(
            util.to_dict(feature_labels, default_feature_transform, True), 
            feature_transform_map
        )
        # Start transforming features
        if "temporal" in transform_resolution:
            for j, (feature_label, f) in enumerate(zip(feature_labels, feature_indices)):
                transform = feature_transform_map[feature_label]
                if isinstance(transform, dict): transform = transform["name"]
                features[:,j] = util.transform(
                    [
                        features[:,j],
                        transform_args_map[transform][0][periodic_indices,f], 
                        transform_args_map[transform][1][periodic_indices,f]
                    ], 
                    feature_transform_map[feature_label], 
                    revert
                )
        else:
            for j, (feature_label, f) in enumerate(zip(feature_labels, feature_indices)):
                transform = feature_transform_map[feature_label]
                if isinstance(transform, dict): transform = transform["name"]
                features[:,j] = util.transform(
                    [
                        features[:,j], 
                        transform_args_map[transform][0][f], 
                        transform_args_map[transform][1][f]
                    ], 
                    feature_transform_map[feature_label], 
                    revert
                )
        return features


class Statistics(Container, DataSelection):

    def __init__(self, var):
        # Unpack vars
        misc_var = var.misc
        orig_var = var.original
        load_var = var.loading
        cache_var = var.caching
        part_var = var.partitioning
        struct_var = var.structure
        dist_var = var.distribution
        proc_var = var.processing
        # Load/init statistics
        tmp_var = Container().copy([proc_var, load_var, cache_var, struct_var, misc_var])
        stat_fn_map = {
            "minimums": np.min, 
            "maximums": np.max, 
            "medians": np.median, 
            "means": np.mean, 
            "standard_deviations": np.std, 
        }
        for stat_name, stat_fn in stat_fn_map.items():
            self.set(
                stat_name, 
                self.load_statistic(
                    orig_var.filter_axis(
                        orig_var.features, 
                        0, 
                        orig_var.get("temporal_indices", "train")
                    ), 
                    orig_var.get("periodic_indices", "train"), 
                    orig_var.period_size, 
                    orig_var.get("temporal_interval", "train"),
                    stat_name, 
                    stat_fn, 
                    tmp_var
                )
            )

    def load_statistic(self, features, periodic_indices, period_size, temporal_interval, stat_name, stat_fn, var):
        cache_filename = "TemporalStatistic_Type[%s]_TemporalInterval[%s].pkl" % (
            util.convert_name_convention(stat_name, "snake", "Pascal"), 
            ",".join(temporal_interval)
        )
        cache_path = os.sep.join([var.cache_dir, cache_filename])
        if os.path.exists(cache_path) and var.from_cache:
            stat = util.from_cache(cache_path)
        else:
            stat = self.compute_statistic(features, periodic_indices, period_size, stat_fn, var)
            if var.to_cache:
                util.to_cache(stat, cache_path)
        return stat

    def compute_statistic(self, features, periodic_indices, period_size, stat_fn, var):
        n_temporal, n_feature = features.shape
        stat = np.ones((period_size, n_feature))
        for t in range(period_size):
            periodic_mask = periodic_indices == t
            stat[t,:] = stat_fn(features[periodic_mask], 0)
        # Handle special features
        if var.temporal_label_field in var.feature_index_map:
            idx = var.feature_index_map[var.temporal_label_field]
            if stat_fn == np.min:
                stat[:,idx] = 0
            elif stat_fn == np.max:
                stat[:,idx] = period_size - 1
            elif stat_fn == np.median:
                stat[:,idx] = period_size // 2
            elif stat_fn == np.mean:
                stat[:,idx] = period_size / 2
            elif stat_fn == np.std:
                stat[:,idx] = period_size / 2 / 3
        return stat

    def reduce_statistic_to_resolution(self, stat, transform_resolution, reduction):
        if "temporal" in transform_resolution: # reduce out no dimensions and proceed
            pass
        else: # reduce out temporal dimension
            stat = reduction(stat, 0)
        return stat


class Windowed(Container, DataSelection):

    def __init__(self, var):
        misc_var = var.misc
        orig_var = var.original
        load_var = var.loading
        cache_var = var.caching
        part_var = var.partitioning
        proc_var = var.processing
        struct_var = var.structure
        proc_var = var.processing
        map_var = var.mapping
        tmp_var = Container().copy([misc_var, proc_var])

    def transform(self, features, periodic_indices, feature_indices, var, revert=False):
        # Unpack vars
        n_window, n_temporal, n_spatial, n_feature = features.shape
        feature_labels = util.get_dict_values(var.index_feature_map, feature_indices)
        transform_resolution = var.temporal.transform_resolution
        feature_transform_map = var.temporal.feature_transform_map
        default_feature_transform = var.temporal.default_feature_transform
        # Prepare statistics
        mins = var.statistics.minimums
        maxes = var.statistics.maximums
        meds = var.statistics.medians
        means = var.statistics.means
        stds = var.statistics.standard_deviations
        mins = var.statistics.reduce_statistic_to_resolution(mins, transform_resolution, np.min)
        maxes = var.statistics.reduce_statistic_to_resolution(maxes, transform_resolution, np.max)
        meds = var.statistics.reduce_statistic_to_resolution(meds, transform_resolution, np.median)
        means = var.statistics.reduce_statistic_to_resolution(means, transform_resolution, np.mean)
        stds = var.statistics.reduce_statistic_to_resolution(stds, transform_resolution, np.std)
        if var.temporal_label_field in feature_labels:
            idx = feature_labels.index(var.temporal_label_field)
            if len(stds.shape) == 1:
                stds[idx] = var.statistics.minimums.shape[0] / 2 / 3
        # Setup for transform function calls
        empty_arg = np.zeros(mins.shape)
        transform_args_map = {
            "minmax": [mins, maxes],
            "zscore": [means, stds],
            "log": [empty_arg, empty_arg],
            "root": [empty_arg, empty_arg],
            "identity": [empty_arg, empty_arg],
        }
        if default_feature_transform is None:
            default_feature_transform = var.default_feature_transform
        feature_transform_map = util.merge_dicts(
            util.to_dict(feature_labels, default_feature_transform, True), 
            feature_transform_map
        )
        # Start transforming features
        if "temporal" in transform_resolution:
            for i in range(n_window):
                temporal_indices = temporal_indices[i,:]
                for j, (feature_label, f) in enumerate(zip(feature_labels, feature_indices)):
                    transform = feature_transform_map[feature_label]
                    if isinstance(transform, dict): transform = transform["name"]
                    features[i,:,j] = util.transform(
                        [
                            features[i,:,j],
                            transform_args_map[transform][0][periodic_indices,f], 
                            transform_args_map[transform][1][periodic_indices,f]
                        ], 
                        feature_transform_map[feature_label], 
                        revert
                    )
        else:
            for j, (feature_label, f) in enumerate(zip(feature_labels, feature_indices)):
                transform = feature_transform_map[feature_label]
                if isinstance(transform, dict): transform = transform["name"]
                features[:,:,j] = transform_fn_map[transform](
                    [
                        features[:,:,j], 
                        transform_args_map[transform][0][f], 
                        transform_args_map[transform][1][f]
                    ], 
                    feature_transform_map[feature_label], 
                    revert
                )
        return features


class Transformed(Container, DataSelection):

    def __init__(self, var):
        # Unpack vars
        misc_var = var.misc
        orig_var = var.original
        load_var = var.loading
        cache_var = var.caching
        part_var = var.partitioning
        proc_var = var.processing
        struct_var = var.structure
        dist_var = var.distribution
        proc_var = var.processing
        map_var = var.mapping
        self.original, self.reduced, self.windowed = Container(), Container(), Container()
        # Start transforming data
        tmp_var = Container().copy([misc_var, proc_var])
        tmp_var.statistics = var.statistics
        self.original.periodic_onehots = OneHotEncoder(sparse=False).fit_transform(orig_var.periodic_indices[:,None])
        for partition in part_var.partitions:
            # Transform original data
            self.original.set(
                "features", 
                orig_var.transform(
                    np.copy(orig_var.get("features", partition)), 
                    orig_var.get("periodic_indices", partition), 
                    misc_var.feature_indices, 
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
            self.original.set(
                "periodic_onehots", 
                self.filter_axis(
                    self.original.periodic_onehots, 
                    0, 
                    orig_var.get("periodic_indices", partition)
                ), 
                partition
            )
