import numpy as np
import os
import sys
import time
import Utility as util
from Container import Container
from Variables import Variables
from Data.SpatiotemporalData import SpatiotemporalData
import torch
import torch_geometric as torch_g
import NetworkProperties as netprop
import networkx as nx
from Plotting import Plotting


class GraphData(Container):

    def __init__(self, spatmp, var):
        print("Initialize Graph")
        start = time.time()
        self.copy(var)
        # Construct adjacency matrix
        if self.get("construction_method") == "prodige":
            self.set(
                "adjacency",
                self.prodige_to_adjacency(
                    self.load_prodige(self.get("features")[0], self.get("similarity_measure"), var),
                    spatmp.get("original_n_spatial")
                )
            )
        else:
            # Load metric for the given feature
            if self.get("similarity_measure") == "correlation":
                f = spatmp.get("feature_index_map")[self.get("features")[0]]
                correlations = self.load_correlations(spatmp)
                self.set("metric", self.load_correlations(spatmp)[f,:,:])
            elif self.get("similarity_measure") == "dynamic_time_warping":
                self.set("metric",  self.load_dynamic_time_warping(self.get("features")[0], "complete", self))
            else:
                raise NotImplementedError(self.get("similarity_measure"))
            # Convert metric to similarities in [0, 1]
            self.set("similarity", self.metric_to_similarity(self.get("metric"), self.get("similarity_measure")))
            self.set(
                "adjacency", 
                self.construct_adjacency(
                    self.get("similarity"),
                    self.get("construction_method")
                )
            )
        # Apply diffusion
        if self.get("diffuse"):
            print("Diffusing...")
            self.set(
                "adjacency", 
                self.diffuse_adjacency(
                    self.get("adjacency"), 
                    self.get("diffusion_method"), 
                    self.get("diffusion_sparsification")
                )
            )
        # Remove or add self loops
        for i in range(self.get("adjacency").shape[0]):
            self.get("adjacency")[i,i] = int(self.get("self_loops"))
        # Partition adjacency
        for partition in spatmp.get("partitions"):
            spatial_indices = spatmp.get("original_spatial_indices", partition)
            self.set(
                "adjacency",
                self.get("adjacency")[spatial_indices,:][:,spatial_indices],
                partition
            )
        # Convert adjacency to COO format edges for PyTorch Geometric
        self.set("edges", self.adjacency_to_edges(self.get("adjacency")))
        for partition in self.get("partitions"):
            self.set(
                "edges", 
                self.adjacency_to_edges(self.get("adjacency", partition)), 
                partition
            )
        # Draw spatial indices for each partition
        for partition in self.get("partitions"):
            self.set("spatial_indices", np.arange(self.get("adjacency", partition).shape[0]), partition)
        # Plot graph characteristics
        graph_name = ",".join(map(
            str,
            [
                self.get("similarity_measure"), 
                self.get("construction_method"), 
                self.get("diffusion_method"),
                self.get("diffusion_sparsification")
            ]
        ))
        if len(self.get("plot_graph_distributions")) > 0:
            plt = Plotting()
            G = nx.convert_matrix.from_numpy_matrix(self.get("adjacency"))
            plt.plot_graph_distributions(G, self.get("plot_graph_distributions"), graph_name)
        if len(self.get("compute_graph_metrics")) > 0:
            path = spatmp.get("data_dir") + os.sep + "GraphMetrics_Graph[%s].txt" % (graph_name)
            if not os.path.exists(path):
                G = nx.convert_matrix.from_numpy_matrix(self.get("adjacency"))
                self.set(
                    "graph_metrics", 
                    netprop.compute_graph_metrics(G, self.get("compute_graph_metrics"), graph_name)
                )
                with open(path, "w") as f:
                    f.write("\n".join(map(str, self.get("graph_metrics"))))
        print(time.time() - start)
        # Print adjacencies
        print(self.get("adjacency").shape)
        print("No. edges =", np.sum(self.get("adjacency")))
        print(self.get("adjacency"))
        for partition in self.get("partitions"):
            print(self.get("adjacency", partition).shape)
            print("No. edges =", np.sum(self.get("adjacency", partition)))
            print(self.get("adjacency", partition))
        return
        # Print edges
        print(self.get("edges").shape)
        print(self.get("edges"))
        for partition in self.get("partitions"):
            print(self.get("edges", partition).shape)
            print(self.get("edges", partition))

    def prodige_to_adjacency(self, prodige, n_nodes):
        adj = np.zeros([n_nodes, n_nodes])
        edge_prbs = prodige["edge_probabilities"]
        edge_srcs, edge_dsts = prodige["edge_source_indices"], prodige["edge_destination_indices"]
        edge_srcs = edge_srcs[edge_prbs > 0.5]
        edge_dsts = edge_dsts[edge_prbs > 0.5]
        adj[(edge_srcs, edge_dsts)] = 1
        adj[(edge_dsts, edge_srcs)] = 1
        return adj

    def load_prodige(self, feature, measure, var):
        path = var.get("data_dir") + os.sep + "ProdigeGraph_Feature[%s]_Metric[%s]_Source[%s].pkl" % (
            feature,
            measure,
            var.get("data_source")
        )
        return util.from_cache(path)

    def metric_to_similarity(self, metric, measure):
        if measure == "correlation": # adjust range [-1, 1] -> [0, 1]
            similarity = np.abs(metric)
#            similarity = (similarity + 1) / 2
        elif measure == "dynamic_time_warping": # adjust range [0, inf] -> [0, 1]
            similarity = 1 / (1 + metric)
        else:
            raise NotImplementedError(measure)
        return similarity

    def adjacency_to_edges(self, adj):
        indices = np.where(adj > 0)
        srcs = np.reshape(indices[0], (1, -1))
        dsts = np.reshape(indices[1], (1, -1))
        return np.concatenate([srcs, dsts], axis=0)

    def diffuse_adjacency(self, adj, method, sparsification):
        data = torch_g.data.Data(
            edge_index=torch.tensor(self.adjacency_to_edges(adj), dtype=torch.long),
            num_nodes=adj.shape[0]
        )
        if method[0] == "ppr":
            diffusion_kwargs = {"method": method[0], "alpha": method[1]}
        elif method[0] == "heat":
            diffusion_kwargs = {"method": method[0], "t": method[1]}
        elif method[0] == "coeff":
            diffusion_kwargs = {"method": method[0], "coeffs": method[1]}
        else:
            raise NotImplementedError()
        if sparsification[0] == "threshold":
            if sparsification[1] < 1:
                sparsification_kwargs = {"method": sparsification[0], "eps": sparsification[1]}
            else:
                sparsification_kwargs = {"method": sparsification[0], "avg_degree": sparsification[1]}
        elif sparsification[0] == "topk":
            sparsification_kwargs = {"method": sparsification[0], "k": sparsification[1], "dim": 0}
        else:
            raise NotImplementedError()
        gdc = torch_g.transforms.GDC(
            self_loop_weight=1,
            normalization_in="sym",
            normalization_out="col",
            diffusion_kwargs=diffusion_kwargs,
            sparsification_kwargs=sparsification_kwargs,
            exact=True
        )
        data = gdc(data)
        diffused_edges = util.to_ndarray([data.edge_index])[0]
        diffused_adj = np.zeros(adj.shape, dtype=np.int)
        diffused_adj[(diffused_edges[0], diffused_edges[1])] = 1
        return diffused_adj

    def construct_adjacency(self, similarity, method):
        np.random.seed(1)
        sims = similarity
        n_spa = sims.shape[0]
        adj = np.zeros([n_spa, n_spa], dtype=np.int)
        if method[0] == "threshold":
            threshold = method[1]
            adj[sims > threshold] = 1
        elif method[0] == "topk":
            k = int(method[1])
            for i in range(n_spa):
                indices = np.argsort(sims[i,:])
                indices = np.delete(indices, np.where(indices == i))
                top_k = indices[-k:]
                adj[i,top_k] = 1
                adj[top_k,i] = 1
        elif method[0] == "bottomk":
            k = int(method[1])
            for i in range(n_spa):
                bot_k = np.argsort(sims[i,:])[:k]
                adj[i,bot_k], adj[bot_k,i] = 1, 1
        elif method[0] == "randr":
            r = int(method[1])
            for i in range(n_spa):
                indices = np.delete(np.arange(n_spa), i)
                rand_r = np.random.choice(indices, size=r, replace=False)
                adj[i,rand_r], adj[rand_r,i] = 1, 1
        elif method[0] == "topk_randr":
            k = int(method[1])
            r = int(method[2])
            for i in range(n_spa):
                indices = np.argsort(sims[i,:])
                indices = np.delete(indices, np.where(indices == i))
                top_k = indices[-k:]
                indices = np.delete(np.arange(n_spa), np.concatenate([top_k, np.array([i])]))
                rand_r = np.random.choice(indices, size=r, replace=False)
                adj[i,top_k], adj[i,rand_r] = 1, 1
                adj[top_k,i], adj[rand_r,i] = 1, 1
        return adj

    def load_dynamic_time_warping(self, feature, quality, var):
        data_dir = var.get("data_dir")
        cache_path = data_dir + os.sep + "DynamicTimeWarping_Feature[%s]_Quality[%s].pkl" % (
            feature,
            quality
        )
        if os.path.exists(cache_path):
            dtw = util.from_cache(cache_path)
        else:
            raise FileNotFoundError("File %s does not exist" % (cache_path))
        return dtw

    def load_correlations(self, spatmp, from_cache=True, to_cache=True):
        data_dir = spatmp.get("data_dir")
        cache_path = data_dir + os.sep + "Correlation_Source[%s].pkl" % (spatmp.get("data_source"))
        if os.path.exists(cache_path) and from_cache:
            correlations = util.from_cache(cache_path)
        else:
            feature_transformation_map = spatmp.get("feature_transformation_map")
            new_feature_transformation_map = {}
            for feature, transformation in feature_transformation_map.items():
                new_feature_transformation_map[feature] = "z_score".split(",")
                if feature == "date":
                    new_feature_transformation_map[feature] = "min_max".split(",")
            spatmp.set("feature_transformation_map", new_feature_transformation_map)
            transformation_resolution = spatmp.get("transformation_resolution")
            new_transformation_resolution = "temporal,spatial,feature".split(",")
            spatmp.set("transformation_resolution", new_transformation_resolution)
            partition = "train"
            partition = None
            n = spatmp.get("reduced_n_temporal", partition)[0]
            reduced = spatmp.get("reduced", partition)[:1]
            features = spatmp.get("features")
            feature_indices = spatmp.get("feature_indices")
            reduced = spatmp.transform_reduced(
                reduced,
                spatmp.get("reduced_n_temporal", partition),
                [0, 1, 2, 3],
                util.convert_dates_to_daysofyear(spatmp.get("reduced_temporal_labels", partition)) - 1,
                spatmp.get("original_spatial_indices", partition),
                feature_indices,
                spatmp
            )
            spatmp.set("feature_transformation_map", feature_transformation_map)
            spatmp.set("transformation_resolution", transformation_resolution)
            for i in range(len(feature_indices)):
                print(features[i])
                f = feature_indices[i]
                print(np.mean(reduced[0,:n,0,f]))
                print(np.std(reduced[0,:n,0,f]))
            original = reduced[0]
            n_tmp, n_spa, n_ftr = original.shape[0], original.shape[1], original.shape[2]
            original = util.move_axes([original], [0, 1, 2], [1, 0, 2])[0]
            correlations = np.zeros([n_ftr, n_spa, n_spa])
            for i in range(n_ftr):
                try:
                    correlations[i,:,:] = np.corrcoef(original[:,:,i])
                except FloatingPointError as e:
                    pass
            if to_cache:
                util.to_cache(correlations, cache_path)
        return correlations


if __name__ == "__main__":
    pass
