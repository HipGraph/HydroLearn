import itertools
import gzip
import pandas as pd
import os
import sys
import time
import shutil
import numpy as np
import datetime as dt
from inspect import currentframe

import Utility as util
from Data.Integration import *
from Data.DataSelection import DataSelection
from Container import Container


def copy_missing_data(integrator, generator):
    if os.path.exists(integrator.spatial_labels_outpath()) and not os.path.exists(generator.spatial_labels_outpath()):
        shutil.copy(integrator.spatial_labels_outpath(), generator.spatial_labels_outpath())
    if os.path.exists(integrator.temporal_labels_outpath()) and not os.path.exists(generator.temporal_labels_outpath()):
        shutil.copy(integrator.temporal_labels_outpath(), generator.temporal_labels_outpath())
    if os.path.exists(integrator.spatiotemporal_features_outpath()) and not os.path.exists(generator.spatiotemporal_features_outpath()):
        shutil.copy(integrator.spatiotemporal_features_outpath(), generator.spatiotemporal_features_outpath())
    if os.path.exists(integrator.spatial_features_outpath()) and not os.path.exists(generator.spatial_features_outpath()):
        shutil.copy(integrator.spatial_features_outpath(), generator.spatial_features_outpath())
    if os.path.exists(integrator.temporal_features_outpath()) and not os.path.exists(generator.temporal_features_outpath()):
        shutil.copy(integrator.temporal_features_outpath(), generator.temporal_features_outpath())
    if os.path.exists(integrator.graph_features_outpath()) and not os.path.exists(generator.graph_features_outpath()):
            shutil.copy(integrator.graph_features_outpath(), generator.graph_features_outpath())


def generate_graph(dataset, inputs, var, args):
    # Handle imports
    import torch
    from Models.Model import GraphConstructor, GraphConstructor_HyperparameterVariables
    # Handle arguments
    hyp_var = Container().set(
        [
            "graph_constructor_kwargs", 
        ], 
        [
            GraphConstructor_HyperparameterVariables(), 
        ], 
        multi_value=True
    )
    hyp_var = hyp_var.merge(args)
    # Prepare model and data
    model = GraphConstructor(**hyp_var.graph_constructor_kwargs.to_dict())
    model.debug = args.debug
    model, inputs = model.prepare(inputs, var.execution.use_gpu)
    # Generate graph data
    model.debug = 1
    outputs = model(**inputs)
    edge_index = util.to_ndarray(outputs["edge_index"])
    edge_weight = util.to_ndarray(outputs["edge_weight"])
    if args.debug:
        print("edge_index =", edge_index.shape)
        print(edge_index)
        print("edge_weight =", edge_weight.shape)
        print(edge_weight)
    if args.prune_weightless:
        edge_index = edge_index[:,edge_weight > 0]
        edge_weight = edge_weight[edge_weight > 0]
        if args.debug:
            print("edge_index =", edge_index.shape)
            print(edge_index)
            print("edge_weight =", edge_weight.shape)
            print(edge_weight)
    spatial_labels = dataset.get(
        "spatial_labels", 
        "train", 
        path=[var.execution.principle_data_type, var.execution.principle_data_form]
    )
    df = pd.DataFrame(
        {
            "source": spatial_labels[edge_index[0,:]], 
            "destination": spatial_labels[edge_index[1,:]], 
            "weight": edge_weight, 
        }, 
        dtype=str
    )
    return df


class Generator:

    debug = 2

    def __init__(self):
        os.makedirs(self.root_dir(), exist_ok=True)
        os.makedirs(self.cache_dir(), exist_ok=True)
        os.makedirs(self.generate_dir(), exist_ok=True)

    def name(self):
        return self.__class__.__name__

    def root_dir(self):
        return os.sep.join([data_dir(), self.name()]) 

    def cache_dir(self):
        return os.sep.join([self.root_dir(), "Generation", "Cache"])

    def generate_dir(self):
        return self.root_dir()

    def spatial_labels_fname(self):
        return "SpatialLabels.csv"

    def temporal_labels_fname(self):
        return "TemporalLabels.csv"

    def spatial_features_fname(self):
        return "Spatial.csv"

    def temporal_features_fname(self):
        return "Temporal.csv"

    def spatiotemporal_features_fname(self):
        return "Spatiotemporal.csv.gz"

    def graph_features_fname(self):
        return "Graph.csv"

    def spatial_labels_outpath(self):
        return os.sep.join([self.convert_dir(), self.spatial_labels_fname()])

    def temporal_labels_outpath(self):
        return os.sep.join([self.convert_dir(), self.temporal_labels_fname()])

    def spatial_features_outpath(self):
        return os.sep.join([self.convert_dir(), self.spatial_features_fname()])

    def temporal_features_outpath(self):
        return os.sep.join([self.convert_dir(), self.temporal_features_fname()])

    def spatiotemporal_features_outpath(self):
        return os.sep.join([self.convert_dir(), self.spatiotemporal_features_fname()])

    def graph_features_outpath(self):
        return os.sep.join([self.convert_dir(), self.graph_features_fname()])

    def generate(self, args):
        func_name = "%s.%s" % (self.__class__.__name__, currentframe().f_code.co_name)
        raise NotImplementedError("Implement %s() please!" % (func_name))


class GraphGenerator(Generator):

    def name(self):
        raise NotImplementedError()

    def dataset(self):
        raise NotImplementedError()

    def default_args(self):
        return Container().set(
            [
                "debug", 
                "graph_construction_method", 
                "prune_weightless", 
            ], 
            [
                2, 
                ["k-nn", "Minkowski-2", 2], 
                True, 
            ], 
            multi_value=True
        )

    def pull_data(self, dataset, partition, var):
        raise NotImplementedError()

    def generate(self, args):
        # Handle imports
        from Variables import Variables
        from Data.Data import Data
        # Handle arguments + variables
        args = self.default_args().merge(args)
        var = Variables().merge(args)
        var.meta.load_graph = False
        # Generate graph
        var.execution.set("dataset", self.dataset(), ["train", "valid", "test"])
        dataset = Data(var).get("dataset", "train")
        inputs = self.pull_data(dataset, "train", None)
        df = generate_graph(dataset, inputs, var, args)
        df.to_csv(self.graph_features_outpath(), index=False)


class _METR_LA(GraphGenerator):

    def name(self):
        return "_METR-LA"

    def dataset(self):
        return "metr-la"

    def pull_data(self, dataset, partition, var):
        data = {}
        data["__sampled__"] = []
        P = dataset.spatial.filter_axis(
            dataset.spatial.original.features, 
            -1, 
            np.array(util.get_dict_values(dataset.spatial.misc.feature_index_map, ["longitude", "latitude"]))
        ).astype(float)
        P = util.minmax_transform(P, np.min(P, 0), np.max(P, 0))
        data["x"] = P
        data["b"] = -np.eye(P.shape[0])
        data["ignore_self_loops"] = False
        for key, value in data.items():
            if key == "x" and isinstance(value, list):
                for i, x in enumerate(value):
                    print("x[%d] =" % (i), x.shape)
            elif isinstance(value, np.ndarray):
                print(key, "=", value.shape)
            else:
                print(key, "=", value)
        return data


class _PEMS_BAY(_METR_LA):

    def name(self):
        return "_PEMS-BAY"

    def dataset(self):
        return "_pems-bay"


class NEW_METR_LA(GraphGenerator):

    def name(self):
        return "NEW-METR-LA"

    def dataset(self):
        return "new-metr-la"

    def pull_data(self, dataset, partition, var):
        data = {}
        data["__sampled__"] = []
        P = dataset.spatial.filter_axis(
            dataset.spatial.original.features, 
            -1, 
            np.array(util.get_dict_values(dataset.spatial.misc.feature_index_map, ["Longitude", "Latitude"]))
        ).astype(float)
        print(P.shape)
        print(np.min(P, 0), np.max(P, 0))
        P = util.minmax_transform(P, np.min(P, 0), np.max(P, 0))
        print(np.min(P, 0), np.max(P, 0))
        input()
        F = dataset.spatial.filter_axis(
            dataset.spatial.original.features, 
            -1, 
            dataset.spatial.misc.feature_index_map["Freeway"]
        )[:,None]
        B = dataset.spatial.filter_axis(
            dataset.spatial.original.features, 
            -1, 
            dataset.spatial.misc.feature_index_map["Direction"]
        )[:,None]
        from sklearn.preprocessing import OneHotEncoder
        _F = OneHotEncoder(sparse=False).fit_transform(F)
        _B = OneHotEncoder(sparse=False).fit_transform(B)
        _R = np.dot(_F, _F.T) * np.dot(_B, _B.T)
        _D = np.zeros(_R.shape)
        for i, j in zip(*np.where(_R == 1)):
            if B[i] == "N":
                _D[i,j] = P[j][1] > P[i][1]
            if B[i] == "S":
                _D[i,j] = P[j][1] < P[i][1]
            if B[i] == "W":
                _D[i,j] = P[j][0] < P[i][0]
            if B[i] == "E":
                _D[i,j] = P[j][0] > P[i][0]
        data["x"] = P
        data["m"] = _R
        data["b"] = -np.eye(P.shape[0])# * sys.float_info.max
        data["ignore_self_loops"] = False
        for key, value in data.items():
            if key == "x" and isinstance(value, list):
                for i, X in enumerate(value):
                    print("x[%d] =" % (i), X.shape)
            elif isinstance(value, np.ndarray):
                print(key, "=", value.shape)
            else:
                print(key, "=", value)
        return data


class NEW_PEMS_BAY(NEW_METR_LA):

    def name(self):
        return "NEW-PEMS-BAY"

    def dataset(self):
        return "new-pems-bay"


class _Traffic(Generator):

    def name(self):
        return "_Traffic"

    def dataset(self):
        return "traffic"


class _Solar_Energy(Generator):

    def name(self):
        return "_Solar-Energy"

    def dataset(self):
        return "solar-energy"


class _Electricity(Generator):

    def name(self):
        return "_Electricity"

    def dataset(self):
        return "electricity"


class _Exchange_Rate(Generator):

    def name(self):
        return "_Exchange-Rate"

    def dataset(self):
        return "exchange-rate"


class US_Streams(GraphGenerator):

    def name(self):
        return "US-Streams"

    def default_args(self):
        return Container().set(
            [
                "debug", 
                "graph_construction_method", 
                "prune_weightless", 
            ], 
            [
                2, 
#                ["top-k", "Minkowski-2", 1.0], 
                ["k-nn", "Minkowski-2", 1], 
#                ["threshold", "Minkowski-2", "gt", 0.9], 
                True, 
            ], 
            multi_value=True
        )

    def dataset(self):
        return "us-streams-wa"

    def pull_data(self, dataset, partition, var):
        from sklearn.preprocessing import OneHotEncoder
        data = {}
        data["__sampled__"] = []
        pos = dataset.spatial.filter_axis(
            dataset.spatial.original.features, 
            -1, 
            np.array(util.get_dict_values(dataset.spatial.misc.feature_index_map, ["dec_long_va", "dec_lat_va"]))
        ).astype(float)
        elev = dataset.spatial.filter_axis(
            dataset.spatial.original.features, 
            -1, 
            util.get_dict_values(dataset.spatial.misc.feature_index_map, ["elev_m"])
            )[:,None]
        huc12s = dataset.spatial.filter_axis(
            dataset.spatial.original.features, 
            -1, 
            util.get_dict_values(dataset.spatial.misc.feature_index_map, ["huc12"])
            )[:,None]
        to_huc12s = dataset.spatial.filter_axis(
            dataset.spatial.original.features, 
            -1, 
            dataset.spatial.misc.feature_index_map["tohuc12"]
        )[:,None]
        names = dataset.spatial.filter_axis(
            dataset.spatial.original.features, 
            -1, 
            dataset.spatial.misc.feature_index_map["nm"]
        )[:,None]
        def uptok_hops(G, k):
            if k < 2:
                return G
            G_ks = [np.linalg.matrix_power(G, _k+1) for _k in range(k)]
            for i in range(k):
                G = np.logical_or(G, G_ks[i]).astype(G.dtype)
            return G
        idx = slice(0, 10)
        print(pos[idx])
        print(elev[idx])
        print(huc12s[idx])
        print(to_huc12s[idx])
        print(names[idx])
        P = util.minmax_transform(pos, np.min(pos, 0), np.max(pos, 0))
        n_node = P.shape[0]
        G = np.zeros((n_node, n_node), dtype=int)
        for i in range(n_node):
            huc12 = huc12s[i,0]
            sinks = np.where(huc12s[:,0] == to_huc12s[i])
            if len(sinks) > 0:
                G[i,sinks[0]] = 1
        G = uptok_hops(G, 3)
        print(G)
        print(np.sum(G, 0))
        print(np.sum(G, 1))
        print(np.unique(np.sum(G, 0), return_counts=True))
        print(np.unique(np.sum(G, 1), return_counts=True))
        N = OneHotEncoder(sparse=False).fit_transform(names)
        H = OneHotEncoder(sparse=False).fit_transform(huc12s)
        R = np.dot(N, N.T)
        R = np.logical_or(G, np.dot(H, H.T)).astype(int)
        R = np.logical_or(G, np.dot(H, H.T)).astype(int) * np.dot(N, N.T)
        D = np.zeros((n_node, n_node), dtype=int)
        for i, j in itertools.permutations(range(n_node), 2):
            D[i,j] = elev[i,0] > elev[j,0]
        data["x"] = P
        data["m"] = R * D
        data["b"] = -np.eye(n_node)
        data["prune"] = ["weight", "le", 0.0]
        for key, value in data.items():
            if key == "x" and isinstance(value, list):
                for i, X in enumerate(value):
                    print("x[%d] =" % (i), X.shape)
            elif isinstance(value, np.ndarray):
                print(key, "=", value.shape)
            else:
                print(key, "=", value)
        return data

    def generate(self, args):
        # Handle imports
        from Variables import Variables
        from Data.Data import Data
        import re
        # Handle arguments
        args = self.default_args().merge(args)
        var = Variables().merge(args)
        var.execution.principle_data_type = "spatial"
        var.meta.load_spatiotemporal = False
        var.meta.load_graph = False
        # Generate graph
        state_abbrevs = ["WA"]
        for state_abbrev in state_abbrevs:
            dataset = "us-streams-%s" % (state_abbrev.lower())
            var.execution.set("dataset", dataset, ["train", "valid", "test"])
            dataset = Data(var).get("dataset", "train")
            inputs = self.pull_data(dataset, "train", None)
            df = generate_graph(dataset, inputs, var, args)
            print(df)
            path = os.sep.join([self.generate_dir(), state_abbrev, self.graph_features_fname()])
            df.to_csv(path, index=False)


class Arsenic(GraphGenerator):

    def name(self):
        return "Arsenic"

    def pull_data(self, dataset, partition, var):
        data = {}
        data["__sampled__"] = []
        data["x"] = dataset.spatial.transformed.filter_axis(
            dataset.spatial.transformed.original.numerical_features, 
            -1, 
            np.array(
                util.get_dict_values(dataset.spatial.misc.feature_index_map, ["X_Albers", "Y_Albers", "WellDepth"])
            )
        )
        data["ignore_self_loops"] = False
        for key, value in data.items():
            if key == "x" and isinstance(value, list):
                for i, X in enumerate(value):
                    print("x[%d] =" % (i), X.shape)
            elif isinstance(value, np.ndarray):
                print(key, "=", value.shape)
            else:
                print(key, "=", value)
        return data
