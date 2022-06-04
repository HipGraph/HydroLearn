import numpy as np
import os
import time
from progressbar import ProgressBar
import pandas as pd
from sklearn import preprocessing
from Container import Container
from Data.DataSelection import DataSelection
import Utility as util


class SpatialData(Container, DataSelection):

    def __init__(self, var):
        print(util.make_msg_block(" Spatial Data Initialization : Started ", "#"))
        start = time.time()
        tmp_var = Container().copy(var)
        self.set("misc", Miscellaneous(tmp_var))
        print("    Initialized Miscellaneous: %.3fs" % ((time.time() - start)))
        start = time.time()
        tmp_var = tmp_var.set("misc", self.get("misc"))
        self.set("original", Original(tmp_var))
        print("    Initialized Original: %.3fs" % ((time.time() - start)))
        start = time.time()
        tmp_var = tmp_var.set("original", self.get("original"))
        self.set("transformed", Transformed(tmp_var))
        print("    Initialized Transformed: %.3fs" % ((time.time() - start)))
        self.set("partitioning", var.get("partitioning"))
        print(util.make_msg_block("Spatial Data Initialization : Completed", "#"))


class Miscellaneous(Container, DataSelection):

    def __init__(self, var):
        map_var = var.get("mapping")
        load_var = var.get("loading")
        self.set("feature_index_map", util.to_key_index_dict(load_var.get("feature_fields")))
        self.set("index_feature_map", util.invert_dict(self.get("feature_index_map")))
        self.set("features", list(self.get("feature_index_map").keys()))
        self.set(
            "feature_indices",
            np.array(util.get_dict_values(self.get("feature_index_map"), self.get("features")))
        )
        self.set("n_features", len(self.get("features")))


class Original(Container, DataSelection):

    def __init__(self, var):
        load_var = var.get("loading")
        cache_var = var.get("caching")
        part_var = var.get("partitioning")
        struct_var = var.get("structure")
        dist_var = var.get("distribution")
        tmp_var = Container().copy([load_var, cache_var, struct_var])
        if dist_var.get("process_rank") == dist_var.get("root_process_rank"):
            self.set("features", self.load_features(tmp_var))
            self.set("n_features", self.get("features").shape[-1])
            self.set("spatial_labels", self.load_spatial_labels(tmp_var))
            self.set(
                "spatial_indices", 
                self.indices_from_selection(self.get("spatial_labels"), part_var.get("spatial_selection"))
            )
            self.set("n_spatial", len(self.get("spatial_labels")))
            for partition in part_var.get("partitions"):
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
                        0,
                        self.get("spatial_indices", partition)
                    ), 
                    partition
                )
                self.set("n_features", self.get("features", partition).shape[-1], partition)

    def load_features(self, var):
        from_cache, to_cache = var.get("from_cache"), var.get("to_cache")
        feature_fields = var.get("feature_fields")
        spatial_label_field = var.get("spatial_label_field")
        categorical_fields = [] if var.get("categorical_fields") is None else var.get("categorical_fields")
        missing_value_code = var.get("missing_value_code")
        text_path = os.sep.join([var.get("data_dir"), var.get("original_text_filename")])
        cache_path = os.sep.join([var.get("cache_dir"), var.get("original_cache_filename")])
        if os.path.exists(cache_path) and from_cache:
            features = util.from_cache(cache_path)
        elif os.path.exists(text_path):
            if not missing_value_code is None:
                df = pd.read_csv(text_path, na_values=missing_value_code)
            else:
                df = pd.read_csv(text_path)
            n_spatial = len(df[spatial_label_field].unique())
            n_features = len(feature_fields)
            features = df[feature_fields].to_numpy()
            features = np.reshape(features, [n_spatial, n_features])
            if spatial_label_field in feature_fields:
                idx = feature_fields.index(spatial_label_field)
                features[:,:,idx] = util.labels_to_ids(features[:,:,idx])
            feature_index_map = util.to_key_index_dict(feature_fields)
            index_feature_map = util.invert_dict(feature_index_map)
            for feature, j in feature_index_map.items(): # Handle types
                dtype = str if feature in categorical_fields else float
                # Check if this is a multi-value feature
                if any([isinstance(features[i,j], str) and ";" in features[i,j] for i in range(n_spatial)]):
                    for i in range(n_spatial):
                        features[i,j] = np.array(features[i,j].split(";"), dtype)
                else:
                    features[:,j] = features[:,j].astype(dtype)
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


class Transformed(Container, DataSelection):

    def __init__(self, var):
        load_var = var.get("loading")
        cache_var = var.get("caching")
        part_var = var.get("partitioning")
        struct_var = var.get("structure")
        dist_var = var.get("distribution")
        misc_var = var.get("misc")
        orig_var = var.get("original")
        self.copy(orig_var)
        features = np.copy(self.get("features"))
        feature_index_map = misc_var.get("feature_index_map")
        index_feature_map = misc_var.get("index_feature_map")
        categorical_indices, numerical_indices = np.array((), dtype=int), np.arange(features.shape[-1])
        if not load_var.get("categorical_fields") is None: # Apply categorical transforms
            categorical_encoder_map = {}
            categorical_indices = np.array(
                util.get_dict_values(misc_var.get("feature_index_map"), load_var.get("categorical_fields"))
            )
            feature_encoder_map = {}
            for feature in load_var.get("categorical_fields"):
                j = feature_index_map[feature]
                if isinstance(features[0,j], np.ndarray): # Spatial element has multiple classes for this feature
                    labels = np.concatenate([features[i,j] for i in range(features.shape[0])])
                else:
                    labels = np.array([features[i,j] for i in range(features.shape[0])])
                categorical_encoder_map[feature] = LabelEncoder().fit(labels)
            self.set("categorical_encoder_map", categorical_encoder_map)
            for feature in load_var.get("categorical_fields"):
                j = feature_index_map[feature]
                encoder = categorical_encoder_map[feature]
                for i in range(features.shape[0]):
                    features[i,j] = categorical_encoder_map[feature].transform(features[i,j])
            numerical_indices = np.delete(numerical_indices, categorical_indices)
        # Apply numerical transforms
        numerical_features = features[:,numerical_indices]
        scaler = preprocessing.StandardScaler().fit(numerical_features)
        features[:,numerical_indices] = scaler.transform(numerical_features)
        self.set("features", features)
        self.set("scaler", scaler)
        self.set("numerical_feature_indices", numerical_indices)
        self.set(
            "numerical_features", 
            self.filter_axis(
                self.get("features"), 
                -1,
                self.get("numerical_feature_indices")
            ).astype(np.float32)
        )
        self.set("n_numerical_features", self.get("numerical_features").shape[-1])
        self.set("categorical_feature_indices", categorical_indices)
        self.set(
            "categorical_features", 
            self.filter_axis(
                self.get("features"), 
                -1,
                self.get("categorical_feature_indices")
            )
        )
        self.set("n_categorical_features", self.get("categorical_features").shape[-1])
        if dist_var.get("process_rank") == dist_var.get("root_process_rank"):
            for partition in part_var.get("partitions"):
                self.set(
                    "features",
                    self.filter_axis(
                        self.get("features"), 
                        0,
                        self.get("spatial_indices", partition)
                    ), 
                    partition
                )
                self.set(
                    "numerical_features",
                    self.filter_axis(
                        self.get("numerical_features"), 
                        0,
                        self.get("spatial_indices", partition)
                    ), 
                    partition
                )
                self.set(
                    "categorical_features",
                    self.filter_axis(
                        self.get("categorical_features"), 
                        0,
                        self.get("spatial_indices", partition)
                    ), 
                    partition
                )


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
