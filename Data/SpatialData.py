import numpy as np
import os
import time
from progressbar import ProgressBar
from Container import Container
import Utility as util


class SpatialData(Container):

    def __init__(self, var):
        print(util.make_msg_block(" Spatial Data Initialization : Started ", "#"))
        start = time.time()
        con = self.init_original(Container(), var)
        self.set("original", con)
        print("    Initialized Original: %.3fs" % ((time.time() - start)))
        self.set(["partitioning"], [var.get("partitioning")])
        print(util.make_msg_block("Spatial Data Initialization : Completed", "#"))

    def init_original(self, con, var):
        load_var = var.get("loading")
        cache_var = var.get("caching")
        part_var = var.get("partitioning")
        struct_var = var.get("structure")
        dist_var = var.get("distribution")
        tmp_var = Container().copy([load_var, cache_var, struct_var])
        if dist_var.get("process_rank") == dist_var.get("root_process_rank"):
            con.set("original", self.load_original(tmp_var))
            con.set("original_spatial_labels", self.load_original_spatial_labels(tmp_var))
            con.set(
                "original_spatial_indices", 
                self.get_original_spatial_indices(
                    part_var.get("spatial_selection"), 
                    con.get("original_spatial_labels")
                )
            )
            for partition in part_var.get("partitions"):
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
                    "original",
                    self.filter_axis(
                        con.get("original"), 
                        0,
                        con.get("original_spatial_indices", partition)
                    ), 
                    partition
                )
            con.set("original_n_exogenous", con.get("original").shape[-1])
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

    def load_original(self, var, from_cache=True, to_cache=True):
        data_dir = var.get("data_dir")
        cache_dir = var.get("cache_dir")
        n_spatial = var.get("original_n_spatial")
        spatial_feature = var.get("spatial_feature")
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
                original = np.zeros((n_spatial, n_features), dtype=np.float)
                spatial_idx = header_field_index_map[spatial_feature]
                spatial_label = None
                last_values = ["0.0" for i in range(len(lines[0].split(",")))]
                i = 0
                pb = ProgressBar()
                for line in pb(lines):
                    values = line.split(",")
                    for feature in header_feature_fields:
                        idx = header_field_index_map[feature]
                        j = feature_index_map[feature]
                        if values[idx] == missing_value_code:
                            values[idx] = last_values[idx]
                        original[i,j] = float(values[idx])
                    last_values = values
                    i += 1
            if to_cache:
                util.to_cache(original, cache_path)
        else:
            raise FileNotFoundError("Text file \"%s\" nor cache file \"%s\" exist" % (text_path, cache_path))
        return original

    def load_original_spatial_labels(self, var, from_cache=True, to_cache=True):
        data_dir = var.get("data_dir")
        cache_dir = var.get("cache_dir")
        n_spatial = var.get("original_n_spatial")
        spatial_feature = var.get("spatial_feature")
        header_field_index_map = var.get("header_field_index_map")
        missing_value_code = var.get("missing_value_code")
        text_filename = var.get("original_spatial_labels_text_filename")
        cache_filename = var.get("original_spatial_labels_cache_filename")
        text_path = os.sep.join([data_dir, text_filename])
        cache_path = os.sep.join([cache_dir, cache_filename])
        if os.path.exists(cache_path) and from_cache:
            original_spatial_labels = util.from_cache(cache_path)
        elif os.path.exists(text_path):
            with open(text_path) as f:
                lines = f.read().split("\n")[1:-1]
                original_spatial_labels = np.empty((n_spatial), dtype=object)
                spatial_idx = header_field_index_map[spatial_feature]
                i = 0
                pb = ProgressBar()
                for line in pb(lines):
                    values = line.split(",")
                    if values[spatial_idx] == missing_value_code:
                        raise ValueError("Labels cannot be missing")
                    original_spatial_labels[i] = values[spatial_idx]
                    i += 1
            if to_cache:
                util.to_cache(original_spatial_labels, cache_path)
        else:
            raise FileNotFoundError("Text file \"%s\" nor cache file \"%s\" exist" % (text_path, cache_path))
        return original_spatial_labels

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