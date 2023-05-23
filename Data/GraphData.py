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
        part_var = var.partitioning
        self.edges, self.weights = self.load_edges_weights(var)
        self.node_labels = self.load_node_labels(var)
        self.node_indices = self.indices_from_selection(self.node_labels, ["all"])
        self.edge_indices = edge_indices_from_selection(self.edges, self.weights, self.node_labels)
        self.n_nodes = self.node_labels.shape[0]
        self.n_edges = self.edges.shape[0]
        self.edge_index = self.edges_to_edge_index(self.edges, self.node_labels)
        self.coo_edge_index = np.transpose(self.edge_index)
        self.A = edgelist_to_A(self.n_nodes, self.edge_index)
        self.W = edgelist_to_W(self.n_nodes, self.edge_index, self.weights)
        self.nx_graph = nx.from_pandas_adjacency(
            pd.DataFrame(self.A, index=self.node_labels, columns=self.node_labels),
            create_using=nx.DiGraph()
        )
        self.node_degree_map, self.node_indegree_map, self.node_outdegree_map = self.A_to_node_degree_maps(
            self.A, self.node_labels
        )
        for partition in part_var.partitions:
            self.set(
                "node_indices",
                self.indices_from_selection(self.node_labels, part_var.get("node_selection", partition)),
                partition
            )
            self.set(
                "node_labels",
                self.filter_axis(self.node_labels, 0, self.get("node_indices", partition)),
                partition
            )
            self.set("n_nodes", self.get("node_labels", partition).shape[0], partition)
            self.set(
                "A",
                self.filter_axis(self.get("A"), [0, 1], self.get("node_indices", partition)),
                partition
            )
            self.set(
                "W",
                self.filter_axis(self.get("W"), [0, 1], self.get("node_indices", partition)),
                partition
            )
            self.set(
                "edge_indices", 
                edge_indices_from_selection(
                    self.edges, 
                    self.weights, 
                    self.get("node_labels", partition)
                ), 
                partition
            )
            self.set(
                "edges", 
                self.filter_axis(self.edges, 0, self.get("edge_indices", partition)), 
                partition
            )
            self.set("weights", self.filter_axis(self.weights, 0, self.get("edge_indices", partition)), partition)
            self.set(
                "edge_index", 
                self.edges_to_edge_index(self.get("edges", partition), self.get("node_labels", partition)), 
                partition
            )
            self.set("n_edges", self.get("edges", partition).shape[0], partition)
            self.set("coo_edge_index", np.transpose(self.get("edge_index", partition)), partition)
            self.set("coo_edges", np.transpose(self.get("edges", partition)), partition)
            self.set(
                "nx_graph",
                nx.from_pandas_adjacency(
                    pd.DataFrame(
                        self.get("A", partition),
                        index=self.get("node_labels", partition),
                        columns=self.get("node_labels", partition)
                    ),
                    create_using=nx.DiGraph()
                ),
                partition
            )
#            self.set(
#                "node_mask",
#                self.indices_from_selection(self.get("node_labels", partition), ["all"]),
#                partition
#            )
            node_degree_map, node_indegree_map, node_outdegree_map = self.A_to_node_degree_maps(
                self.get("A", partition),
                self.get("node_labels", partition)
            )
            self.set("node_degree_map", node_degree_map, partition)
            self.set("node_indegree_map", node_indegree_map, partition)
            self.set("node_outdegree_map", node_outdegree_map, partition)

    def load_edges_weights(self, var):
        load_var = var.loading
        struct_var = var.structure
        data_dir = struct_var.data_dir
        fname = load_var.original_text_filename
        src_field = load_var.source_field
        dst_field = load_var.destination_field
        weight_field = load_var.weight_field
        path = os.sep.join([data_dir, fname])
        df = pd.read_csv(path, dtype=load_var.dtypes, na_values=load_var.missing_value_code)
        edges = df[[src_field, dst_field]].astype(str).to_numpy()
        if weight_field is None:
            weights = np.ones(edges.shape[0])
        else:
            weights = df[weight_field].to_numpy()
        return [edges, weights]

    def load_node_labels(self, var):
        load_var = var.loading
        struct_var = var.structure
        data_dir = struct_var.data_dir
        fname = load_var.original_node_labels_text_filename
        node_label_field = load_var.node_label_field
        path = os.sep.join([data_dir, fname])
        df = pd.read_csv(path, dtype=load_var.dtypes, na_values=load_var.missing_value_code)
        node_labels = df[node_label_field].to_numpy().astype(str)
        return node_labels

    def edges_to_edge_index(self, edges, node_labels):
        edges_shape = edges.shape
        selection = ["literal"] + np.reshape(edges, -1).tolist()
        edge_index = np.reshape(self.indices_from_selection(node_labels, selection), edges_shape)
        return edge_index

    def _A_to_node_degrees(self, A, node_labels, node_indices):
        node_indegree_map, node_outdegree_map = {}, {}
        for label, idx in zip(node_labels, node_indices):
            node_indegree_map[label] = np.sum(A[node_indices,idx])
            node_outdegree_map[label] = np.sum(A[idx,node_indices])
        return node_indegree_map, node_outdegree_map

    def A_to_node_degree_maps(self, A, node_labels):
        node_degree_map, node_indegree_map, node_outdegree_map = {}, {}, {}
        for i, node_label in enumerate(node_labels):
            node_indegree_map[node_label] = np.sum(A[:,i] > 0)
            node_outdegree_map[node_label] = np.sum(A[i,:] > 0)
            node_degree_map[node_label] = node_indegree_map[node_label] + node_outdegree_map[node_label]
        return node_degree_map, node_indegree_map, node_outdegree_map


def edge_indices_from_selection(edges, weights, selection):
    indices = []
    if edges.shape[0] == 2: # edges.shape=(2, |E|)
        for i in range(edges.shape[1]):
            if edges[0,i] in selection and edges[1,i] in selection:
                indices.append(i)
        indices = np.array(indices, dtype=int)
        selected_edges = edges[:,indices]
    elif edges.shape[1] == 2: # edges.shape=(|E|, 2)
        for i in range(edges.shape[0]):
            if edges[i,0] in selection and edges[i,1] in selection:
                indices.append(i)
        indices = np.array(indices, dtype=int)
    else:
        raise NotImplementedError(edge.shape)
    return indices


def edgelist_to_A(n, edge_index, weights=None):
    A = np.zeros((n, n))
    A[edge_index[:,0],edge_index[:,1]] = 1
    return A


def edgelist_to_W(n, edge_index, weights):
    W = np.zeros((n, n))
    W[edge_index[:,0],edge_index[:,1]] = weights
    return W


def A_to_edgelist(A, labels=None):
    src, dst = np.where(A == 1)
    edgelist = []
    for _src, _dst in zip(src, dst):
        if labels is None:
            edge = [_src, _dst]
        else:
            edge = [labels[_src], labels[_dst]]
        edgelist.append(edge)
    return edgelist


def W_to_edgelist(W, labels=None):
    src, dst = np.where(W > 0)
    edgelist = []
    for _src, _dst in zip(src, dst):
        if labels is None:
            edge = [_src, _dst, W[_src,_dst]]
        else:
            edge = [labels[_src], labels[_dst], W[_src,_dst]]
        edgelist.append(edge)
    return edgelist


if __name__ == "__main__":
    pass
