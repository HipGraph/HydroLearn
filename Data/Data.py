from Container import Container
from SpatiotemporalData import SpatiotemporalData
from SpatialData import SpatialData
from TemporalData import TemporalData
from Graph import Graph


class Data(Container):

    def __init__(self, var):
        exec_var = var.get("execution")
        dist_var = var.get("distribution")
        partitions = list(exec_var.get("partitions"))
        datasets = exec_var.get("dataset", partitions)
        uniq_datasets = list(set(datasets))
        for dataset in uniq_datasets:
            self.set(dataset, self.init_dataset(Container(), dataset, var))
            print(self.get(dataset))
        quit()
        for dataset, partition in zip(datasets, partitions):
            self.set("spatiotemporal", self.get(dataset), partition)
        graph = None
        if exec_var.get("model") == "GNN":
            grph_var = Container().copy([exec_var, var.get("graph"), var.get("plotting")])
            graph = Graph(self.get("spatiotemporal", "train"), grph_var)
        spatial = None
        if exec_var.get("model") == "GEOMAN":
            spa_var = Container().copy([exec_var, var.get("basindata")])
            for partition in partitions:
                spa_var.set("spatial_selection", spatmp_var.get("spatial_selection", partition), partition)
            spatial = SpatialData(spa_var)
        for partition in partitions:
            self.set("graph", graph, partition)
            self.set("spatial", spatial, partition)

    def init_dataset(self, con, dataset, var):
        dataset_var = var.get("datasets").get(dataset)
        if not dataset_var.get("spatiotemporal") is None:
            spatmp_var = Container().copy(
                [
                    dataset_var.get("spatiotemporal"), 
                    var.checkout(["mapping", "distribution", "processing"]), 
                ]
            )
            con.set("spatiotemporal", SpatiotemporalData(spatmp_var))
        if not dataset_var.get("spatial") is None:
            spatmp_var = Container().copy(
                [
                    dataset_var.get("spatial"), 
                    var.checkout(["mapping", "distribution", "processing"]), 
                ]
            )
            con.set("spatial", SpatialData(init_var))
        if not dataset_var.get("temporal") is None:
            spatmp_var = Container().copy(
                [
                    dataset_var.get("temporal"), 
                    var.checkout(["mapping", "distribution", "processing"]), 
                ]
            )
            con.set("temporal", TemporalData(init_var))
        return con
