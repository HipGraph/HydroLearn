import numpy as np
import os
import time
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from Container import Container
from Data.DataSelection import DataSelection
from Data import Imputation
import Utility as util


class SpatialData(Container, DataSelection):

    def __init__(self, var):
        tmp_var = Container().copy(var)
        print(util.make_msg_block(" Spatial Data Initialization : Started ", "#"))
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
        self.transformed = Transformed(tmp_var)
        tmp_var.transformed = self.transformed
        print("    Initialized Transformed: %.3fs" % ((time.time() - start)))
        print(util.make_msg_block("Spatial Data Initialization : Completed", "#"))
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
        if map_var.spatial.predictor_features is None:
            map_var.spatial.predictor_features = self.features
        for feature in map_var.spatial.predictor_features:
            if not feature in self.feature_index_map:
                raise ValueError("Predictor feature \"%s\" does not exist in this dataset" % (feature))
        self.predictor_features = map_var.spatial.predictor_features
        self.predictor_indices = np.array(util.get_dict_values(self.feature_index_map, self.predictor_features))
        self.n_predictor = len(self.predictor_features)
        # Init response feature labels
        if map_var.spatial.response_features is None:
            map_var.spatial.response_features = self.features
        for feature in map_var.spatial.response_features:
            if not feature in self.feature_index_map:
                raise ValueError("Response feature \"%s\" does not exist in this dataset" % (feature))
        self.response_features = map_var.spatial.response_features
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


class Original(Container, DataSelection):

    def __init__(self, var):
        # Unpack vars
        load_var = var.loading
        cache_var = var.caching
        part_var = var.partitioning
        struct_var = var.structure
        dist_var = var.distribution
        misc_var = var.misc
        # Load/init global data
        tmp_var = Container().copy([load_var, cache_var, struct_var])
        self.features, self.gtmask = self.load_features(tmp_var)
        self.n_feature = self.features.shape[-1]
        self.spatial_labels = self.load_spatial_labels(tmp_var)
        self.spatial_indices = self.indices_from_selection(self.spatial_labels, part_var.spatial_selection)
        self.n_spatial = len(self.spatial_labels)
        # Separate numerical and categorical features
        self.categorical_indices, self.numerical_indices = np.arange(0), np.arange(self.n_feature)
        if not load_var.categorical_fields is None:
            self.categorical_indices = np.array(
                util.get_dict_values(misc_var.feature_index_map, load_var.categorical_fields), dtype=int
            )
        self.numerical_indices = np.delete(self.numerical_indices, self.categorical_indices)
        self.numerical_features = self.filter_axis(
            self.features, 
            -1, 
            self.numerical_indices
        ).astype(misc_var.float_dtype)
        self.categorical_features = self.filter_axis(self.features, -1, self.categorical_indices)
        self.n_numerical_features = self.numerical_features.shape[-1]
        self.n_categorical_features = self.categorical_features.shape[-1]
        # Filter/init data for each partition
        for partition in part_var.partitions:
            #   Spatial label data
            self.set(
                "spatial_indices", 
                self.indices_from_selection(self.spatial_labels, part_var.get("spatial_selection", partition)), 
                partition
            )
            self.set(
                "spatial_labels",
                self.filter_axis(self.spatial_labels, 0, self.get("spatial_indices", partition)), 
                partition
            )
            #   Feature data
            self.set(
                "features",
                self.filter_axis(self.features, 0, self.get("spatial_indices", partition)), 
                partition
            )
            self.set(
                "gtmask",
                self.filter_axis(self.gtmask, 0, self.get("spatial_indices", partition)), 
                partition
            )
            self.set(
                "numerical_features",
                self.filter_axis(self.numerical_features, 0, self.get("spatial_indices", partition)), 
                partition
            )
            self.set(
                "categorical_features",
                self.filter_axis(self.categorical_features, 0, self.get("spatial_indices", partition)), 
                partition
            )
            self.set(
                "n_spatial",
                self.get("features", partition).shape[0], 
                partition
            )
            try:
                self.set(
                    "predictor_features", 
                    self.filter_axis(self.get("numerical_features", partition), -1, misc_var.predictor_indices), 
                    partition
                )
                self.set(
                    "predictor_gtmask", 
                    self.filter_axis(self.get("gtmask", partition), -1, misc_var.predictor_indices), 
                    partition
                )
            except: pass
            try:
                self.set(
                    "response_features", 
                    self.filter_axis(self.get("numerical_features", partition), -1, misc_var.response_indices), 
                    partition
                )
                self.set(
                    "response_gtmask", 
                    self.filter_axis(self.get("gtmask", partition), -1, misc_var.response_indices), 
                    partition
                )
            except: pass

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
            df = pd.read_csv(
                text_path, 
                usecols=[var.spatial_label_field] + var.feature_fields, 
                dtype=var.dtypes, 
                na_values=var.missing_value_code
            )
            n_spatial = len(df[var.spatial_label_field])
            n_feature = len(var.feature_fields)
            # Handle missing values
            M = Imputation.missing_value_matrix(df, "spatial", var)
            missing_df = Imputation.missing_values(df, "spatial", var)
            missing_df.to_csv(os.sep.join([var.cache_dir, "MissingValues_Data[Spatial].csv"]), index=False)
            df, imputed_mask = Imputation.impute(df, "spatial", "median", var)
            # Convert to NumPy.ndarray with shape=(n_spatial, n_feature)
            features = np.reshape(df[var.feature_fields].to_numpy(), [n_spatial, n_feature])
            gtmask = np.reshape(np.logical_not(imputed_mask[var.feature_fields].to_numpy()), [n_spatial, n_feature])
            if var.spatial_label_field in var.feature_fields:
                idx = var.feature_fields.index(var.spatial_label_field)
                features[:,:,idx] = util.labels_to_ids(features[:,:,idx])
            feature_index_map = util.to_key_index_dict(var.feature_fields)
            index_feature_map = util.invert_dict(feature_index_map)
            for feature, j in feature_index_map.items(): # Handle types
                dtype = str if feature in var.categorical_fields else float
                # Check if this is a multi-value feature
                if any([isinstance(features[i,j], str) and ";" in features[i,j] for i in range(n_spatial)]):
                    for i in range(n_spatial):
                        features[i,j] = np.array(features[i,j].split(";"), dtype)
                else:
                    features[:,j] = features[:,j].astype(dtype)
            if var.to_cache:
                util.to_cache(features, cache_path)
                util.to_cache(gtmask, gtmask_cache_path)
        else:
            raise FileNotFoundError("Text file \"%s\" nor cache file \"%s\" exist" % (text_path, cache_path))
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
            spatial_labels = df[var.spatial_label_field].unique().astype(str)
            if var.to_cache:
                util.to_cache(spatial_labels, cache_path)
        else:
            raise FileNotFoundError("Text file \"%s\" nor cache file \"%s\" exist" % (text_path, cache_path))
        return spatial_labels

    def transform(self, features, feature_indices, var, revert=False):
        # Unpack vars
        n_spatial, n_feature = features.shape
        feature_labels = util.get_dict_values(var.index_feature_map, feature_indices)
        transform_resolution = var.spatial.transform_resolution
        feature_transform_map = var.spatial.feature_transform_map
        default_feature_transform = var.spatial.default_feature_transform
        # Prepare statistics
        mins = var.statistics.minimums
        maxes = var.statistics.maximums
        meds = var.statistics.medians
        means = var.statistics.means
        stds = var.statistics.standard_deviations
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
                    orig_var.numerical_features, 
                    stat_name, 
                    stat_fn, 
                    tmp_var
                )
            )

    def load_statistic(self, features, stat_name, stat_fn, var):
        cache_dir = var.cache_dir
        cache_filename = "SpatialStatistic_Type[%s].pkl" % (
            util.convert_name_convention(stat_name, "snake", "Pascal") 
        )
        cache_path = os.sep.join([cache_dir, cache_filename])
        if os.path.exists(cache_path) and var.from_cache:
            stat = util.from_cache(cache_path)
        else:
            stat = self.compute_statistic(features, stat_fn, var)
            if var.to_cache:
                util.to_cache(stat, cache_path)
        return stat

    def compute_statistic(self, features, stat_fn, var):
        n_spatial, n_feature = features.shape
        stat = stat_fn(features, 0)
        # Handle special features
        if var.spatial_label_field in var.feature_index_map:
            idx = var.feature_index_map[var.spatial_label_field]
            if stat_fn == np.min:
                stat[idx] = 0
            elif stat_fn == np.max:
                stat[idx] = n_spatial - 1
            elif stat_fn == np.median:
                stat[idx] = n_spatial // 2
            elif stat_fn == np.mean:
                stat[idx] = n_spatial / 2
            elif stat_fn == np.std:
                stat[idx] = n_spatial / 2 / 3
        if stat_fn == np.std:
            stat[stat == 0] = 1
        return stat


class Transformed(Container, DataSelection):

    def __init__(self, var):
        # Unpack vars
        load_var = var.loading
        cache_var = var.caching
        part_var = var.partitioning
        struct_var = var.structure
        dist_var = var.distribution
        proc_var = var.processing
        misc_var = var.misc
        orig_var = var.original
        # Start transforming data
        self.original = Container()
        tmp_var = Container().copy([misc_var, proc_var])
        tmp_var.statistics = var.statistics
        self.original.numerical_features = orig_var.transform(
            orig_var.numerical_features, 
            np.arange(orig_var.n_numerical_features), 
            tmp_var
        )
        self.original.categorical_features = orig_var.categorical_features
        self.original.categorical_features, self.original.categorical_encoders = self.transform_categorical(
            orig_var.categorical_features, 
            util.get_dict_values(misc_var.index_feature_map, orig_var.categorical_indices)
        )
        for partition in part_var.partitions:
            self.original.set(
                "numerical_features",
                self.filter_axis(
                    self.original.numerical_features, 
                    0,
                    orig_var.get("spatial_indices", partition)
                ), 
                partition
            )
            self.original.set(
                "categorical_features", 
                self.filter_axis_foreach(
                    self.original.categorical_features, 
                    0, 
                    orig_var.get("spatial_indices", partition)
                ), 
                partition
            )
            try:
                self.original.set(
                    "predictor_features", 
                    self.filter_axis(
                        self.original.numerical_features,
                        [0, 1], 
                        [orig_var.get("spatial_indices", partition), misc_var.predictor_indices]
                    ), 
                    partition
                )
            except: pass
            try:
                self.original.set(
                    "response_features", 
                    self.filter_axis(
                        self.original.numerical_features,
                        [0, 1], 
                        [orig_var.get("spatial_indices", partition), misc_var.response_indices]
                    ), 
                    partition
                )
            except: pass

    def transform_categorical(self, features, feature_labels, feature_catlabels_map={}):
        feature_onehots_map = {}
        feature_encoder_map = {}
        for i, feature_label in enumerate(feature_labels):
            if feature_label in feature_catlabels_map:
                encoder = OneHotEncoder(feature_catlabels_map[feature_label], sparse=False)
            else:
                encoder = OneHotEncoder(sparse=False).fit(features[:,i][:,None])
            feature_onehots_map[feature_label] = encoder.transform(features[:,i][:,None])
            feature_encoder_map[feature_label] = encoder
        return feature_onehots_map, feature_encoder_map


# Created this to replace SkLearn's preprocessing.LabelEncoder because it is slow. Specifically, SkLearn's LabelEncoder required over 5 second to preprocess "wabashriver_swat" spatial data.
class LabelEncoder:

    def __init__(self):
        pass

    def fit(self, labels):
        self.classes_ = np.unique(labels)
        self.label_encoding_map = util.to_key_index_dict(self.classes_)
        return self

    def transform(self, labels):
        if isinstance(labels, str): # Single label
            labels = [labels]
        return util.get_dict_values(self.label_encoding_map, labels)
