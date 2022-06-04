import os
import time
import numpy as np
import networkx as nx
import pandas as pd
import Utility as util
from Container import Container
from Data.DataSelection import DataSelection


class GraphData(Container, DataSelection):

    def __init__(self, var):
        print(util.make_msg_block(" Graph Data Initialization : Started ", "#"))
        start = time.time()
        self.set("original", Original(var))
        print("    Initialized Original: %.3fs" % ((time.time() - start)))
        print(util.make_msg_block("Graph Data Initialization : Completed", "#"))


class Original(GraphData, DataSelection):

    def __init__(self, var):
        part_var = var.get("partitioning")
        self.set(["edges", "weights"], self.load_edges_weights(var), multi_value=True)
        self.set("node_labels", self.load_node_labels(var))
        self.set("node_indices", self.indices_from_selection(self.get("node_labels"), ["all"]))
        self.set("n_nodes", self.get("node_labels").shape[0])
        self.set("edge_indices", self.edges_to_edge_indices(self.get("node_labels"), self.get("edges")))
        self.set("coo_edge_indices", np.transpose(self.get("edge_indices")))
        self.set("adjacency", edge_indices_to_adjacency(self.get("n_nodes"), self.get("edge_indices")))
        self.set("nx_graph", nx.convert_matrix.from_numpy_matrix(self.get("adjacency")))
        self.set(
            "nx_graph", 
            nx.from_pandas_adjacency(
                pd.DataFrame(
                    self.get("adjacency"), 
                    index=self.get("node_labels"), 
                    columns=self.get("node_labels")
                ),
                create_using=nx.DiGraph()
            )
        )
        node_indegree_map, node_outdegree_map = self.adjacency_to_node_degrees(
            self.get("adjacency"), 
            self.get("node_labels"), 
            self.get("node_indices")
        )
        self.set("node_indegree_map", node_indegree_map)
        self.set("node_outdegree_map", node_outdegree_map)
        for partition in part_var.get("partitions"):
            self.set(
                "node_indices", 
                self.indices_from_selection(self.get("node_labels"), part_var.get("node_selection", partition)),
                partition
            )
            self.set(
                "node_labels", 
                self.filter_axis(self.get("node_labels"), 0, self.get("node_indices", partition)), 
                partition
            )
            self.set(
                "adjacency", 
                self.filter_axis(self.get("adjacency"), [0, 1], self.get("node_indices", partition)), 
                partition
            )
            self.set("edge_indices", adjacency_to_edge_indices(self.get("adjacency", partition)), partition)
            self.set(
                "edges", 
                np.reshape(
                    self.filter_axis(
                        self.get("node_labels", partition), 0, np.reshape(self.get("edge_indices", partition), -1)
                    ), 
                    self.get("edge_indices", partition).shape
                ), 
                partition
            )
            self.set("coo_edge_indices", np.transpose(self.get("edge_indices", partition)), partition)
            self.set("coo_edge_labels", np.transpose(self.get("edges", partition)), partition)
            self.set(
                "nx_graph", 
                nx.from_pandas_adjacency(
                    pd.DataFrame(
                        self.get("adjacency", partition), 
                        index=self.get("node_labels", partition), 
                        columns=self.get("node_labels", partition)
                    ),
                    create_using=nx.DiGraph()
                ), 
                partition
            )
            self.set(
                "node_mask", 
                self.indices_from_selection(self.get("node_labels", partition), ["all"]), 
                partition
            )
            node_indegree_map, node_outdegree_map = self.adjacency_to_node_degrees(
                self.get("adjacency"), 
                self.get("node_labels", partition), 
                self.get("node_indices", partition)
            )
            self.set("node_indegree_map", node_indegree_map, partition)
            self.set("node_outdegree_map", node_outdegree_map, partition)

    def load_edges_weights(self, var):
        load_var = var.get("loading")
        struct_var = var.get("structure")
        data_dir = struct_var.get("data_dir")
        fname = load_var.get("original_text_filename")
        src_field = load_var.get("source_field")
        dst_field = load_var.get("destination_field")
        weight_field = load_var.get("weight_field")
        path = os.sep.join([data_dir, fname])
        df = pd.read_csv(path, dtype={src_field: str, dst_field: str})
        edges = df[[src_field, dst_field]].to_numpy()
        if weight_field is None:
            weights = np.ones(edges.shape[0])
        else:
            weights = df[weight_field].to_numpy()
        return [edges, weights]

    def load_node_labels(self, var):
        load_var = var.get("loading")
        struct_var = var.get("structure")
        data_dir = struct_var.get("data_dir")
        fname = load_var.get("original_node_labels_text_filename")
        node_label_field = load_var.get("node_label_field")
        path = os.sep.join([data_dir, fname])
        if "missing_value_code" in load_var:
            df = pd.read_csv(path, na_values=load_var.get("missing_value_code"))
        else:
            df = pd.read_csv(path)
        node_labels = df[node_label_field].to_numpy().astype(str)
        return node_labels

    def edges_to_edge_indices(self, node_labels, edges):
        edges_shape = edges.shape
        selection = ["literal"] + np.reshape(edges, -1).tolist()
        edge_indices = np.reshape(self.indices_from_selection(node_labels, selection), edges_shape)
        return edge_indices

    def adjacency_to_node_degrees(self, adj, node_labels, node_indices):
        node_indegree_map, node_outdegree_map = {}, {}
        for label, idx in zip(node_labels, node_indices):
            node_indegree_map[label] = np.sum(adj[node_indices,idx])
            node_outdegree_map[label] = np.sum(adj[idx,node_indices])
        return node_indegree_map, node_outdegree_map


def adjacency_to_edge_indices(adj):
    edge_indices = np.column_stack(np.where(adj != 0))
    return edge_indices


def edge_indices_to_adjacency(n, edge_indices):
    adj = np.zeros((n, n), dtype=int)
    for i in range(edge_indices.shape[0]):
        src_idx, dst_idx = edge_indices[i][0], edge_indices[i][1]
        adj[src_idx, dst_idx] = 1
    return adj


if __name__ == "__main__":
    pass
