from Container import Container
from SpatiotemporalData import SpatiotemporalData
from SpatialData import SpatialData
from TemporalData import TemporalData
from Graph import Graph


class Data(Container):

    def __init__(self, var):
        exec_var = var.get("execution")
        partitions = list(exec_var.get("partitions"))
        datasets = exec_var.get("dataset", partitions)
        uniq_datasets = list(set(datasets))
        # Initialize each unique dataset being considered. This way we do not init the same dataset twice.
        for dataset in uniq_datasets:
            self.set(dataset, self.init_dataset(Container(), var.get("datasets").get(dataset), var))
        # Add additional reference(s) to each dataset as "dataset" under the partition given in exec_var. This adds some abstraction but is primarily to allow the user to simply call get("dataset", partition) rather than look-up the dataset name for each partition.
        for dataset, partition in zip(datasets, partitions):
            self.set("dataset", self.get(dataset), partition)
        graph = None

    def init_dataset(self, con, dataset_var, var):
        spatiotemporal, spatial, temporal, graph = None, None, None, None
        if not dataset_var.get("spatiotemporal") is None:
            init_var = Container().copy(
                [
                    dataset_var.get("spatiotemporal"), 
                    var.checkout(["mapping", "distribution", "processing"]), 
                ]
            )
            spatiotemporal = SpatiotemporalData(init_var)
        if not dataset_var.get("spatial") is None:
            init_var = Container().copy(
                [
                    dataset_var.get("spatial"), 
                    var.checkout(["mapping", "distribution", "processing"]), 
                ]
            )
            spatial = SpatialData(init_var)
        if not dataset_var.get("temporal") is None:
            init_var = Container().copy(
                [
                    dataset_var.get("temporal"), 
                    var.checkout(["mapping", "distribution", "processing"]), 
                ]
            )
            temporal = TemporalData(init_var)
        init_var = var.checkout(["execution", "graph", "plotting"])
        if var.get("execution").get("model") == "GNN":
            graph = Graph(spatiotemporal, init_var)
        con.set("spatiotemporal", spatiotemporal)
        con.set("spatial", spatial)
        con.set("temporal", temporal)
        con.set("graph", graph)
        return con
