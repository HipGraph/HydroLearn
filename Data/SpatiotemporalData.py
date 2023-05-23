import numpy as np
import datetime as dt
import sys
import os
import time
import pandas as pd
import matplotlib

import Utility as util
from Plotting import Plotting
from Container import Container
from Data.DataSelection import DataSelection
from Data import Imputation
from Data import Clustering, Probability


class SpatiotemporalData(Container, DataSelection):

    debug = 0

    def __init__(self, var):
        tmp_var = Container().copy(var)
        print(util.make_msg_block(" Spatiotemporal Data Initialization : Started "))
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
        self.reduced = Reduced(tmp_var)
        tmp_var.reduced = self.reduced
        print("    Initialized Reduced: %.3fs" % ((time.time() - start)))
        start = time.time()
        self.reduced_statistics = ReducedStatistics(tmp_var)
        tmp_var.reduced_statistics = self.reduced_statistics
        print("    Initialized Reduced Statistics: %.3fs" % ((time.time() - start)))
        start = time.time()
        self.windowed = Windowed(tmp_var)
        tmp_var.windowed = self.windowed
        print("    Initialized Windowed: %.3fs" % ((time.time() - start)))
        start = time.time()
        self.transformed = Transformed(tmp_var)
        tmp_var.transformed = self.transformed
        print("    Initialized Transformed: %.3fs" % ((time.time() - start)))
        print(util.make_msg_block("Spatiotemporal Data Initialization : Completed"))
        self.partitioning = Container().copy(tmp_var.partitioning)
#        self.loading = Container().copy(tmp_var.loading)


class Miscellaneous(Container, DataSelection):

    def __init__(self, var):
        map_var = var.mapping
        load_var = var.loading
        proc_var = var.processing
        self.feature_index_map = util.to_key_index_dict(load_var.feature_fields)
        self.index_feature_map = util.invert_dict(self.feature_index_map)
        # Init global features labels
        self.features = list(self.feature_index_map.keys())
        self.feature_indices = np.array(util.get_dict_values(self.feature_index_map, self.features))
        self.n_feature = len(self.features)
        # Init predictor feature labels
        if map_var.spatiotemporal.predictor_features is None:
            map_var.spatiotemporal.predictor_features = self.features
        for feature in map_var.spatiotemporal.predictor_features:
            if not feature in self.feature_index_map:
                raise ValueError("Predictor feature \"%s\" does not exist in this dataset" % (feature))
        self.predictor_features = map_var.spatiotemporal.predictor_features
        self.predictor_indices = np.array(util.get_dict_values(self.feature_index_map, self.predictor_features))
        self.n_predictor = len(self.predictor_features)
        # Init response feature labels
        if map_var.spatiotemporal.response_features is None:
            map_var.spatiotemporal.response_features = self.features
        for feature in map_var.spatiotemporal.response_features:
            if not feature in self.feature_index_map:
                raise ValueError("Response feature \"%s\" does not exist in this dataset" % (feature))
        self.response_features = map_var.spatiotemporal.response_features
        self.response_indices = np.array(util.get_dict_values(self.feature_index_map, self.response_features))
        self.n_response = len(self.response_features)
        # Init other feature labels
        self.categorical_features = load_var.categorical_fields
        self.numerical_features = util.list_subtract(self.features, self.categorical_features)
        # Set precision
        self.int_dtype = {16: np.int16, 32: np.int32, 64: np.int64}[load_var.precision]
        self.float_dtype = {16: np.float16, 32: np.float32, 64: np.float64}[load_var.precision]
        # Set loading vars
        self.spatial_label_field = load_var.spatial_label_field
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
        #   Spatial label data
        self.spatial_labels = self.load_spatial_labels(tmp_var)
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
        self.features, self.gtmask = self.load_features(tmp_var)
        self.features = self.features.astype(misc_var.float_dtype)
        self.n_temporal, self.n_spatial, self.n_feature = self.features.shape
        # Filter/init data for each partition
        for partition in part_var.partitions:
            #   Spatial label data
            self.set(
                "spatial_indices",
                self.indices_from_selection(
                    self.spatial_labels, 
                    part_var.get("spatial_selection", partition)
                ),
                partition
            )
            self.set(
                "spatial_labels",
                self.filter_axis(
                    self.spatial_labels,
                    0,
                    self.get("spatial_indices", partition)
                ),
                partition
            )
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
                self.filter_axis(
                    self.periodic_indices, 
                    0, 
                    self.get("temporal_indices", partition)
                ), 
                partition
            )
            #   Feature data
            self.set(
                "features",
                self.filter_axis(
                    self.features,
                    [0, 1],
                    [
                        self.get("temporal_indices", partition),
                        self.get("spatial_indices", partition)
                    ]
                ),
                partition
            )
            self.set(
                "gtmask",
                self.filter_axis(
                    self.gtmask,
                    [0, 1],
                    [
                        self.get("temporal_indices", partition),
                        self.get("spatial_indices", partition)
                    ]
                ),
                partition
            )
            self.set(
                ["n_temporal", "n_spatial"], 
                list(self.get("features", partition).shape[:-1]), 
                partition, 
                multi_value=True
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
                "predictor_gtmask",
                self.filter_axis(
                    self.get("gtmask", partition), 
                    -1, 
                    misc_var.predictor_indices
                ),
                partition
            )
            self.set(
                "response_gtmask",
                self.filter_axis(
                    self.get("gtmask", partition), 
                    -1, 
                    misc_var.response_indices
                ),
                partition
            )

    def load_features(self, var):
        text_path = os.sep.join([var.data_dir, var.original_text_filename])
        cache_path = os.sep.join([var.cache_dir, var.original_cache_filename])
        gtmask_cache_path = os.sep.join([var.cache_dir, "Spatiotemporal_Form[Original]_Data[GtMask].pkl"])
        if var.from_cache or var.to_cache:
            os.makedirs(var.cache_dir, exist_ok=True)
        if var.from_cache and os.path.exists(cache_path) and os.path.exists(gtmask_cache_path):
            features = util.from_cache(cache_path)
            gtmask = util.from_cache(gtmask_cache_path)
        elif os.path.exists(text_path):
            # Read data and determine its dimensions
            df = pd.read_csv(
                text_path, 
                usecols=[var.spatial_label_field, var.temporal_label_field] + var.feature_fields, 
                dtype=var.dtypes, 
                na_values=var.missing_value_code
            )
            spatial_labels = df[var.spatial_label_field].unique()
            n_spatial = len(spatial_labels)
            temporal_labels = df[var.temporal_label_field].unique()
            n_temporal = len(temporal_labels)
            n_feature = len(var.feature_fields)
            # Handle missing values
            if 0:
                plt = Plotting()
                cmap = matplotlib.colors.ListedColormap(["green", "red"])
                M = Imputation.missing_value_matrix(df, "spatiotemporal", var)
                for key, value in M.items():
                    path = os.sep.join(
                        [var.cache_dir, "MissingValuesHeatmap_Data[Spatiotemporal]_Feature[%s].png" % (key)]
                    )
                    plt.plot_heatmap(value, cmap=cmap, plot_cbar=False, size=(50, 25), path=path)
            missing_df = Imputation.missing_values(df, "spatiotemporal", var)
            path = os.sep.join([var.cache_dir, "MissingValues_Data[Spatiotemporal].csv"])
            missing_df.to_csv(path, index=False)
            df, imputed_mask = Imputation.impute(df, "spatiotemporal", var.imputation_method, var)
            # Convert to NumPy.ndarray with shape=(n_temporal, n_spatial, n_feature)
            features = df[var.feature_fields].to_numpy()
            gtmask = np.logical_not(imputed_mask[var.feature_fields].to_numpy())
            if var.shape == ["spatial", "temporal", "feature"]:
                features = np.reshape(features, [n_spatial, n_temporal, n_feature])
                features = np.swapaxes(features, 0, 1)
                gtmask = np.reshape(gtmask, [n_spatial, n_temporal, n_feature])
                gtmask = np.swapaxes(gtmask, 0, 1)
            elif var.shape == ["temporal", "spatial", "feature"]:
                features = np.reshape(features, [n_temporal, n_spatial, n_feature])
                gtmask = np.reshape(gtmask, [n_temporal, n_spatial, n_feature])
            else:
                raise ValueError("Unknown shape %s given for layout of original spatiotemporal data" % (shape))
            # Handle special features: derive numerical representation for temporal and/or spatial labels
            if var.temporal_label_field in var.feature_fields:
                periodic_indices = util.temporal_labels_to_periodic_indices(
                    temporal_labels, 
                    var.temporal_seasonality_period, 
                    var.temporal_resolution, 
                    var.temporal_label_format
                )
                periodic_indices = np.reshape(np.repeat(periodic_indices, n_spatial), (n_temporal, n_spatial))
                idx = var.feature_fields.index(var.temporal_label_field)
                features[:,:,idx] = periodic_indices
            if var.spatial_label_field in var.feature_fields:
                idx = var.feature_fields.index(var.spatial_label_field)
                features[:,:,idx] = np.tile(util.labels_to_ids(spatial_labels), (n_temporal, 1))
            if var.to_cache:
                util.to_cache(features, cache_path)
                util.to_cache(gtmask, gtmask_cache_path)
        else:
            raise FileNotFoundError("Neither text file \"%s\" nor cache file \"%s\" exist" % (text_path, cache_path))
        return features, gtmask

    def load_spatial_labels(self, var):
        text_path = os.sep.join([var.data_dir, var.original_spatial_labels_text_filename])
        cache_path = os.sep.join([var.cache_dir, var.original_spatial_labels_cache_filename])
        if var.from_cache or var.to_cache:
            os.makedirs(var.cache_dir, exist_ok=True)
        if os.path.exists(cache_path) and var.from_cache:
            spatial_labels = util.from_cache(cache_path)
        elif os.path.exists(text_path):
            df = pd.read_csv(text_path, dtype=var.dtypes, na_values=var.missing_value_code)
            spatial_labels = df[var.spatial_label_field].astype(str).unique()
            if var.to_cache:
                util.to_cache(spatial_labels, cache_path)
        else:
            raise FileNotFoundError("text file \"%s\" nor cache file \"%s\" exist" % (text_path, cache_path))
        return spatial_labels

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
            raise FileNotFoundError("text file \"%s\" nor cache file \"%s\" exist" % (text_path, cache_path))
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
        # Unpack vars
        n_temporal, n_spatial, n_feature = features.shape
        feature_labels = util.get_dict_values(var.index_feature_map, feature_indices)
        transform_resolution = var.spatiotemporal.transform_resolution
        feature_transform_map = var.spatiotemporal.feature_transform_map
        default_feature_transform = var.spatiotemporal.default_feature_transform
        period_size = var.statistics.minimums.shape[0]
        # Prepare statistics
        if transform_resolution == ["temporal", "spatial", "feature"]:
            raise NotImplementedError(transform_resolution)
        elif transform_resolution == ["temporal", "feature"]:
            raise NotImplementedError(transform_resolution)
        elif transform_resolution == ["spatial", "feature"]:
            mins, maxes, meds, means, stds = var.statistics.get(
                ["minimums", "maximums", "medians", "means", "standard_deviations"]
            )
        elif transform_resolution == ["feature"]:
            raise NotImplementedError(transform_resolution)
        else:
            raise ValueError(transform_resolution)
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
        if transform_resolution == ["temporal", "spatial", "feature"]:
            for j, (feature_label, f) in enumerate(zip(feature_labels, feature_indices)):
                transform = feature_transform_map[feature_label]
                if isinstance(transform, dict): transform = transform["name"]
                features[:,:,j] = util.transform(
                    [
                        features[:,:,j],
                        transform_args_map[transform][0][periodic_indices,:,:][:,spatial_indices,f],
                        transform_args_map[transform][1][periodic_indices,:,:][:,spatial_indices,f]
                    ], 
                    feature_transform_map[feature_label], 
                    revert
                )
        elif transform_resolution == ["temporal", "feature"]:
            for j, (feature_label, f) in enumerate(zip(feature_labels, feature_indices)):
                transform = feature_transform_map[feature_label]
                if isinstance(transform, dict): transform = transform["name"]
                features[:,:,j] = util.transform(
                    [
                        features[:,:,j],
                        transform_args_map[transform][0][periodic_indices,f], 
                        transform_args_map[transform][1][periodic_indices,f]
                    ], 
                    feature_transform_map[feature_label], 
                    revert
                )
        elif transform_resolution == ["spatial", "feature"]:
            for j, (feature_label, f) in enumerate(zip(feature_labels, feature_indices)):
                transform = feature_transform_map[feature_label]
                if isinstance(transform, dict): transform = transform["name"]
                features[:,:,j] = util.transform(
                    [
                        features[:,:,j], 
                        transform_args_map[transform][0][spatial_indices,f], 
                        transform_args_map[transform][1][spatial_indices,f]
                    ],
                    feature_transform_map[feature_label], 
                    revert
                )
        elif transform_resolution == ["feature"]:
            for j, (feature_label, f) in enumerate(zip(feature_labels, feature_indices)):
                transform = feature_transform_map[feature_label]
                if isinstance(transform, dict): transform = transform["name"]
                features[:,:,j] = util.transform(
                    [
                        features[:,:,j], 
                        transform_args_map[transform][0][f], 
                        transform_args_map[transform][1][f]
                    ], 
                    feature_transform_map[feature_label], 
                    revert
                )
        else:
            raise ValueError(transform_resolution)
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
        def compute_statistic(features, stat_fn):
            return stat_fn(features, 0)
        features = orig_var.filter_axis(orig_var.features, 0, orig_var.get("temporal_indices", "train"))
        self.minimums = compute_statistic(features, np.min)
        self.maximums = compute_statistic(features, np.max)
        self.medians = compute_statistic(features, np.median)
        self.means = compute_statistic(features, np.mean)
        self.standard_deviations = compute_statistic(features, np.std)
        return
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
        cache_filename = "SpatiotemporalStatistic_Type[%s]_TemporalInterval[%s].pkl" % (
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
        n_temporal, n_spatial, n_feature = features.shape
        stat = np.ones((period_size, n_spatial, n_feature))
        for t in range(period_size):
            periodic_mask = periodic_indices == t
            stat[t,:,:] = stat_fn(features[periodic_mask], 0)
        # Handle special features
        if var.spatial_label_field in var.feature_index_map:
            idx = var.feature_index_map[var.spatial_label_field]
            if stat_fn == np.min:
                stat[:,:,idx] = 0
            elif stat_fn == np.max:
                stat[:,:,idx] = n_spatial - 1
            elif stat_fn == np.median:
                stat[:,:,idx] = n_spatial // 2
            elif stat_fn == np.mean:
                stat[:,:,idx] = n_spatial / 2
            elif stat_fn == np.std:
                stat[:,:,idx] = n_spatial / 2 / 3
        if var.temporal_label_field in var.feature_index_map:
            idx = var.feature_index_map[var.temporal_label_field]
            if stat_fn == np.min:
                stat[:,:,idx] = 0
            elif stat_fn == np.max:
                stat[:,:,idx] = period_size - 1
            elif stat_fn == np.median:
                stat[:,:,idx] = period_size // 2
            elif stat_fn == np.mean:
                stat[:,:,idx] = period_size / 2
            elif stat_fn == np.std:
                stat[:,:,idx] = period_size / 2 / 3
        if stat_fn == np.std:
            stat[stat == 0] = 1
        return stat


class Reduced(Container, DataSelection):

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
        # Load/init data
        if proc_var.temporal_reduction[1] == 1:
            return
        tmp_var = Container().copy([load_var, cache_var, struct_var, misc_var])
        #   Spatial label data
        self.spatial_labels = orig_var.spatial_labels
        #   Temporal label data
        self.temporal_interval = orig_var.temporal_interval
        self.temporal_labels = self.load_temporal_labels(
            orig_var.features,
            orig_var.temporal_labels,
            orig_var.temporal_interval,
            proc_var.temporal_reduction,
            tmp_var
        )
        self.periodic_indices = util.temporal_labels_to_periodic_indices(
            self.temporal_labels,
            load_var.temporal_seasonality_period,
            load_var.temporal_resolution,
            load_var.temporal_label_format
        )
        self.period_size = np.unique(self.periodic_indices).shape[0]
        #   Feature data
        self.features = self.load_features(
            orig_var.features,
            orig_var.temporal_labels,
            orig_var.temporal_interval,
            proc_var.temporal_reduction,
            tmp_var
        ).astype(misc_var.float_dtype)
        self.set(
            ["n_channel", "n_temporal", "n_spatial", "n_feature"], 
            list(self.features.shape), 
            multi_value=True
        )
        # Filter/init data for each partition
        for partition in part_var.partitions:
            #   Spatial label data
            self.set("spatial_indices", orig_var.get("spatial_indices", partition), partition)
            self.set("spatial_labels", orig_var.get("spatial_labels", partition), partition)
            #   Temporal label data
            self.set("temporal_interval", orig_var.get("temporal_interval", partition), partition)
            self.set(
                "temporal_indices", 
                self.get_reduced_temporal_indices(
                    part_var.get("temporal_selection", partition),
                    self.temporal_labels,
                    load_var.temporal_resolution, 
                    load_var.temporal_label_format
                ),
                partition
            )
            self.set(
                "temporal_labels",
                self.filter_axes(
                    self.temporal_labels, 
                    [0, 1], 
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
                self.filter_axes(
                    self.filter_axis(
                        self.features,
                        2,
                        self.get("spatial_indices", partition)
                    ),
                    [0, 1], 
                    self.get("temporal_indices", partition),
                ),
                partition
            )
            self.set(
                ["n_temporal", "n_spatial"], 
                list(self.get("features", partition).shape[1:-1]), 
                partition, 
                multi_value=True
            )

    def load_features(self, original, original_temporal_labels, temporal_interval, temporal_reduction, var):
        cache_filename = var.reduced_cache_filename % (
            temporal_interval[0],
            temporal_interval[1],
            temporal_reduction[0],
            temporal_reduction[1],
            temporal_reduction[2]
        )
        cache_path = os.sep.join([var.cache_dir, cache_filename])
        if os.path.exists(cache_path) and var.from_cache:
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
                if var.to_cache:
                    util.to_cache(reduced, cache_path)
        return reduced

    def load_temporal_labels(self, original, original_temporal_labels, temporal_interval, temporal_reduction, var):
        cache_filename = var.reduced_temporal_labels_cache_filename % (
            temporal_interval[0],
            temporal_interval[1],
            temporal_reduction[0],
            temporal_reduction[1],
            temporal_reduction[2]
        )
        cache_path = os.sep.join([var.cache_dir, cache_filename])
        if os.path.exists(cache_path) and var.from_cache:
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
                if var.to_cache:
                    util.to_cache(reduced_temporal_labels, cache_path)
        return reduced_temporal_labels

    def reduce_original(self, original, original_temporal_labels, temporal_reduction, var):
        n_temporal, n_spatial, n_feature = original.shape
        method, factor, stride = temporal_reduction
        temporal_label_field = var.temporal_label_field
        feature_index_map = var.feature_index_map
        method_fn_map = {
            "min": np.min, 
            "max": np.max, 
            "avg": np.mean, 
            "sum": np.sum, 
        }
        indices = self.get_temporal_reduction_indices(n_temporal, temporal_reduction)
        reduced = method_fn_map[method](original[indices,:,:], 2)
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
        if a == "" or b == "":
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
        # Unpack vars
        n_channel, n_temporal, n_spatial, n_feature = features.shape
        feature_labels = util.get_dict_values(var.index_feature_map, feature_indices)
        transform_resolution = var.spatiotemporal.transform_resolution
        feature_transform_map = var.spatiotemporal.feature_transform_map
        default_feature_transform = var.spatiotemporal.default_feature_transform
        period_size = var.statistics.minimums.shape[0]
        # Prepare statistics
        if transform_resolution == ["temporal", "spatial", "feature"]:
            raise NotImplementedError(transform_resolution)
        elif transform_resolution == ["temporal", "feature"]:
            raise NotImplementedError(transform_resolution)
        elif transform_resolution == ["spatial", "feature"]:
            mins, maxes, meds, means, stds = var.statistics.get(
                ["minimums", "maximums", "medians", "means", "standard_deviations"]
            )
        elif transform_resolution == ["feature"]:
            raise NotImplementedError(transform_resolution)
        else:
            raise ValueError(transform_resolution)
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
        if "temporal" in transform_resolution and "spatial" in transform_resolution:
            for j, (feature_label, f) in enumerate(zip(feature_labels, feature_indices)):
                transform = feature_transform_map[feature_label]
                if isinstance(transform, dict): transform = transform["name"]
                features[:,:,:,j] = util.transform(
                    [
                        features[:,:,:,j],
                        transform_args_map[transform][0][periodic_indices,:,:][:,spatial_indices,f],
                        transform_args_map[transform][1][periodic_indices,:,:][:,spatial_indices,f]
                    ], 
                    feature_transform_map[feature_label], 
                    revert
                )
        elif "temporal" in transform_resolution:
            for j, (feature_label, f) in enumerate(zip(feature_labels, feature_indices)):
                transform = feature_transform_map[feature_label]
                if isinstance(transform, dict): transform = transform["name"]
                features[:,:,:,j] = util.transform(
                    [
                        features[:,:,:,j],
                        transform_args_map[transform][0][periodic_indices,f], 
                        transform_args_map[transform][1][periodic_indices,f]
                    ], 
                    feature_transform_map[feature_label], 
                    revert
                )
        elif "spatial" in transform_resolution:
            for j, (feature_label, f) in enumerate(zip(feature_labels, feature_indices)):
                transform = feature_transform_map[feature_label]
                if isinstance(transform, dict): transform = transform["name"]
                features[:,:,:,j] = util.transform(
                    [
                        features[:,:,:,j], 
                        transform_args_map[transform][0][spatial_indices,f], 
                        transform_args_map[transform][1][spatial_indices,f]
                    ],
                    feature_transform_map[feature_label], 
                    revert
                )
        else:
            for j, (feature_label, f) in enumerate(zip(feature_labels, feature_indices)):
                transform = feature_transform_map[feature_label]
                if isinstance(transform, dict): transform = transform["name"]
                features[:,:,:,j] = util.transform(
                    [
                        features[:,:,:,j], 
                        transform_args_map[transform][0][f], 
                        transform_args_map[transform][1][f]
                    ], 
                    feature_transform_map[feature_label], 
                    revert
                )
        return features


class ReducedStatistics(Container, DataSelection):

    def __init__(self, var):
        # Unpack vars
        misc_var = var.misc
        orig_var = var.original
        red_var = var.reduced
        load_var = var.loading
        cache_var = var.caching
        part_var = var.partitioning
        struct_var = var.structure
        dist_var = var.distribution
        proc_var = var.processing
        # Load/init statistics
        if proc_var.temporal_reduction[1] == 1:
            return
        tmp_var = Container().copy([proc_var, load_var, cache_var, struct_var, misc_var])
        statistic_fn_map = {
            "minimums": np.min, 
            "maximums": np.max, 
            "medians": np.median, 
            "means": np.mean, 
            "standard_deviations": np.std, 
        }
        for stat_name, stat_fn in statistic_fn_map.items():
            self.set(
                stat_name, 
                self.load_statistic(
                    red_var.filter_axes(
                        red_var.features, 
                        [0, 1], 
                        red_var.get("temporal_indices", "train")
                    ), 
                    red_var.get("periodic_indices", "train"), 
                    red_var.period_size, 
                    red_var.get("temporal_interval", "train"),
                    proc_var.temporal_reduction,
                    stat_name, 
                    stat_fn, 
                    tmp_var
                )
            )

    def load_statistic(self, features, periodic_indices, period_size, temporal_interval, temporal_reduction, stat_name, stat_fn, var):
        cache_filename = "SpatiotemporalStatistic_Type[%s]_TemporalInterval[%s]_TemporalReduction[%s].pkl" % (
            util.convert_name_convention(stat_name, "snake", "Pascal"), 
            ",".join(temporal_interval),
            ",".join(map(str, temporal_reduction))
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
        n_channel, n_temporal, n_spatial, n_feature = features.shape
        stat = np.ones((period_size, n_spatial, n_feature))
        for t in range(period_size):
            periodic_mask = periodic_indices == t
            stat[t,:,:] = stat_fn(features[periodic_mask,:,:], 0)
        # Handle special features
        if var.spatial_label_field in var.feature_index_map:
            idx = var.feature_index_map[var.spatial_label_field]
            if stat_fn == np.min:
                stat[:,:,idx] = 0
            elif stat_fn == np.max:
                stat[:,:,idx] = n_spatial - 1
            elif stat_fn == np.median:
                stat[:,:,idx] = n_spatial // 2
            elif stat_fn == np.mean:
                stat[:,:,idx] = n_spatial / 2
            elif stat_fn == np.std:
                stat[:,:,idx] = n_spatial / 2 / 3
        if var.temporal_label_field in var.feature_index_map:
            idx = var.feature_index_map[var.temporal_label_field]
            if stat_fn == np.min:
                stat[:,:,idx] = 0
            elif stat_fn == np.max:
                stat[:,:,idx] = period_size - 1
            elif stat_fn == np.median:
                stat[:,:,idx] = period_size // 2
            elif stat_fn == np.mean:
                stat[:,:,idx] = period_size / 2
            elif stat_fn == np.std:
                stat[:,:,idx] = period_size / 2 / 3
        if stat_fn == np.std:
            stat[stat == 0] = 1
        return stat


class Windowed(Container, DataSelection):

    def __init__(self, var):
        misc_var = var.misc
        orig_var = var.original
        red_var = var.reduced
        stat_var = var.statistics
        load_var = var.loading
        cache_var = var.caching
        part_var = var.partitioning
        proc_var = var.processing
        struct_var = var.structure
        dist_var = var.distribution
        proc_var = var.processing
        map_var = var.mapping
        tmp_var = Container().copy([misc_var, proc_var])
        tmp_var.statistics = stat_var

    def transform(self, features, periodic_indices, spatial_indices, feature_indices, var, revert=False):
        # Unpack vars
        n_window, n_temporal, n_spatial, n_feature = features.shape
        feature_labels = util.get_dict_values(var.index_feature_map, feature_indices)
        transform_resolution = var.spatiotemporal.transform_resolution
        feature_transform_map = var.spatiotemporal.feature_transform_map
        default_feature_transform = var.spatiotemporal.default_feature_transform
        period_size = var.statistics.minimums.shape[0]
        # Prepare statistics
        if transform_resolution == ["temporal", "spatial", "feature"]:
            raise NotImplementedError(transform_resolution)
        elif transform_resolution == ["temporal", "feature"]:
            raise NotImplementedError(transform_resolution)
        elif transform_resolution == ["spatial", "feature"]:
            mins, maxes, meds, means, stds = var.statistics.get(
                ["minimums", "maximums", "medians", "means", "standard_deviations"]
            )
        elif transform_resolution == ["feature"]:
            raise NotImplementedError(transform_resolution)
        else:
            raise ValueError(transform_resolution)
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
        if "temporal" in transform_resolution and "spatial" in transform_resolution:
            for i in range(n_window):
                temporal_indices = periodic_indices[i,:]
                for j, (feature_label, f) in enumerate(zip(feature_labels, feature_indices)):
                    transform = feature_transform_map[feature_label]
                    if isinstance(transform, dict): transform = transform["name"]
                    features[i,:,:,j] = util.transform(
                        [
                            features[i,:,:,j],
                            transform_args_map[transform][0][periodic_indices,:,:][:,spatial_indices,f],
                            transform_args_map[transform][1][periodic_indices,:,:][:,spatial_indices,f]
                        ], 
                        feature_transform_map[feature_label], 
                        revert
                    )
        elif "temporal" in transform_resolution:
            for i in range(n_window):
                temporal_indices = temporal_indices[i,:]
                for j, (feature_label, f) in enumerate(zip(feature_labels, feature_indices)):
                    transform = feature_transform_map[feature_label]
                    if isinstance(transform, dict): transform = transform["name"]
                    features[i,:,:,j] = util.transform(
                        [
                            features[i,:,:,j],
                            transform_args_map[transform][0][periodic_indices,f], 
                            transform_args_map[transform][1][periodic_indices,f]
                        ], 
                        feature_transform_map[feature_label], 
                        revert
                    )
        elif "spatial" in transform_resolution:
            for j, (feature_label, f) in enumerate(zip(feature_labels, feature_indices)):
                transform = feature_transform_map[feature_label]
                if isinstance(transform, dict): transform = transform["name"]
                features[:,:,:,j] = util.transform(
                    [
                        features[:,:,:,j], 
                        transform_args_map[transform][0][spatial_indices,f], 
                        transform_args_map[transform][1][spatial_indices,f]
                    ],
                    feature_transform_map[feature_label], 
                    revert
                )
        else:
            for j, (feature_label, f) in enumerate(zip(feature_labels, feature_indices)):
                transform = feature_transform_map[feature_label]
                if isinstance(transform, dict): transform = transform["name"]
                features[:,:,:,j] = util.transform(
                    [
                        features[:,:,:,j], 
                        transform_args_map[transform][0][f], 
                        transform_args_map[transform][1][f]
                    ], 
                    feature_transform_map[feature_label], 
                    revert
                )
        return features


class Transformed(Container, DataSelection):

    def __init__(self, var):
        misc_var = var.misc
        orig_var = var.original
        stat_var = var.statistics
        red_var = var.reduced
        redstat_var = var.reduced_statistics
        load_var = var.loading
        cache_var = var.caching
        part_var = var.partitioning
        proc_var = var.processing
        struct_var = var.structure
        dist_var = var.distribution
        proc_var = var.processing
        map_var = var.mapping
        self.original, self.reduced, self.windowed = Container(), Container(), Container()
        tmp_var = Container().copy([misc_var, proc_var])
        for partition in part_var.partitions:
            tmp_var.statistics = stat_var
            # Transform original data
            self.original.set(
                "features", 
                orig_var.transform(
                    np.copy(orig_var.get("features", partition)), 
                    orig_var.get("periodic_indices", partition), 
                    orig_var.get("spatial_indices", partition), 
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
            # Transform reduced data
            if not red_var.is_empty():
                tmp_var.statistics = redstat_var
                self.reduced.set(
                    "features", 
                    red_var.transform(
                        np.copy(red_var.get("features", partition)), 
                        red_var.get("periodic_indices", partition), 
                        red_var.get("spatial_indices", partition), 
                        misc_var.feature_indices, 
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


def compute_histograms(features, bins=12, lims=np.array([-3, 3])):
    n_temporal, n_spatial, n_feature = features.shape
    histograms = Probability.compute_histograms(np.reshape(features, (n_temporal, -1)), bins=bins, lims=lims)
    return np.reshape(histograms, (n_spatial, n_feature, histograms.shape[-1]))


def cluster(spatmp, alg="Agglomerative", n_clusters=3, rep="histogram", partition="train", **kwargs):
    # Handle arguments
    bins, lims = kwargs.get("bins", 12), kwargs.get("lims", np.array([-3, 3]))
    debug = kwargs.get("debug", 0)
    # Start
    #   Derive representation of data x
    if rep == "histogram":
        x = compute_histograms(spatmp.transformed.original.get("predictor_features", partition), bins, lims)
        if debug:
            print("x =", x.shape)
            if debug > 1:
                print(x)
        x = np.reshape(x, (x.shape[0], -1))
        if debug:
            print("x =", x.shape)
            if debug > 1:
                print(x)
    elif rep == "wasserstein":
        raise NotImplementedError(rep)
        path = os.sep.join(
            [result_dir, "WassersteinMatrix_Features[%s].pkl" % (",".join(args.features))]
        )
        if os.path.exists(path):
            x = util.from_cache(path)
        else:
            histograms = compute_histograms(
                spatmp.transformed.original.get("predictor_features", partition), bins, lims
            )[0,:,:]
            x = np.zeros((n_spatial, n_spatial))
            for i in range(n_spatial):
                for j in range(i, n_spatial):
                    u, v = histograms[i,:], histograms[j,:]
                    d = stats.wasserstein_distance(u, v)
                    x[i,j] = d
                    x[j,i] = d
                    if debug:
                        print(i, j, d)
            if debug:
                print(x)
                input()
            util.to_cache(x, path)
    elif rep == "mean-std":
        indices = spatmp.misc.predictor_indices
        means = spatmp.filter_axis(spatmp.statistics.means, -1, indices)
        stds = spatmp.filter_axis(spatmp.statistics.standard_deviations, -1, indices)
        x = np.concatenate((means, stds), -1)
    elif rep == "std":
        indices = spatmp.misc.predictor_indices
        x = spatmp.filter_axis(spatmp.statistics.standard_deviations, -1, indices)
    else:
        raise NotImplementedError(rep)
    #   Compute clusters
    if alg in ["Agglomerative"] and rep in ["Wasserstein"]:
        cluster_index = Clustering.cluster(x, alg, n_clusters, affinity="precomputed")
    else:
        cluster_index = Clustering.cluster(x, alg, n_clusters)
    return x, cluster_index
