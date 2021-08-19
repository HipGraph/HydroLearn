from Container import Container
from SpatialData import SpatialData
from SpatiotemporalData import SpatiotemporalData
from Graph import Graph


class Data(Container):

    def __init__(self, var):
        comm_var = var.get("common")
        dist_var = var.get("distributed")
        partitions = list(comm_var.get("partitions"))
        data_srcs = comm_var.get("data_source", partitions)
        uniq_data_srcs = sorted(list(set(data_srcs)))
        main_data_src = comm_var.get("data_source")
        for data_src in uniq_data_srcs:
            src_var = var.get(data_src)
            spatmp_var = Container().copy([src_var, comm_var, dist_var])
            self.set(data_src, SpatiotemporalData(spatmp_var))
        for data_src, partition in zip(data_srcs, partitions):
            self.set("spatiotemporal", self.get(data_src), partition)
        graph = None
        if comm_var.get("model") == "GNN":
            grph_var = Container().copy([comm_var, var.get("graph"), var.get("plotting")])
            graph = Graph(self.get("spatiotemporal", "train"), grph_var)
        spatial = None
        if comm_var.get("model") == "GEOMAN":
            spa_var = Container().copy([comm_var, var.get("basindata")])
            for partition in partitions:
                spa_var.set("spatial_selection", spatmp_var.get("spatial_selection", partition), partition)
            spatial = SpatialData(spa_var)
        for partition in partitions:
            self.set("graph", graph, partition)
            self.set("spatial", spatial, partition)
