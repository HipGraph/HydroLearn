import sys
from Container import Container
from Data.SpatiotemporalData import SpatiotemporalData
from Data.SpatialData import SpatialData
from Data.TemporalData import TemporalData
from Data.GraphData import GraphData
from Data.DataSelection import DataSelection
import Utility as util
import networkx as nx


class Data(Container, DataSelection):

    def __init__(self, var):
        exec_var = var.get("execution")
        partitions = list(exec_var.get("partitions"))
        datasets = exec_var.get("dataset", partitions)
        uniq_datasets = list(set(datasets))
        # Initialize each unique dataset being considered. This way we only init each dataset once.
        tmp = Container()
        for dataset in uniq_datasets:
            tmp.set(dataset, self.init_dataset(dataset, var))
        # Add additional reference(s) to each dataset as "dataset" under the partition given in exec_var.
        for dataset, partition in zip(datasets, partitions):
            self.set("dataset", tmp.get(dataset), partition)
            self.get("dataset", partition).set("name", dataset)

    def init_dataset(self, dataset, var, con=None):
        if con is None: con = Container()
        spatiotemporal, spatial, temporal, graph = Container(), Container(), Container(), Container()
        if var.meta.load_spatial and not var.datasets.get(dataset).spatial.is_empty():
            init_var = pull_init_var(var, dataset, "spatial")
            spatial = SpatialData(init_var)
        if var.meta.load_temporal and not var.datasets.get(dataset).temporal.is_empty():
            init_var = pull_init_var(var, dataset, "temporal")
            temporal = TemporalData(init_var)
        if var.meta.load_spatiotemporal and not var.datasets.get(dataset).spatiotemporal.is_empty():
            init_var = pull_init_var(var, dataset, "spatiotemporal")
            spatiotemporal = SpatiotemporalData(init_var)
        if var.meta.load_graph and not var.datasets.get(dataset).graph.is_empty():
            init_var = pull_init_var(var, dataset, "graph")
            graph = GraphData(init_var)
        con.set("spatial", spatial)
        con.set("temporal", temporal)
        con.set("spatiotemporal", spatiotemporal)
        con.set("graph", graph)
        return con


def pull_init_var(var, dataset, data_type):
    init_var = Container().copy(
        [
            var.datasets.get(dataset).get(data_type), 
            var.checkout(["mapping", "distribution", "processing"])
        ]
    )
    return init_var
