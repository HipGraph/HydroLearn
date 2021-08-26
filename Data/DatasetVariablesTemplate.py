import os
import Utility as util
from Container import Container


def dataset_name():
    return "_".join(dataset_dir().split(os.sep)[-2:]).lower()


def dataset_dir():
    return os.path.dirname(os.path.realpath(__file__))


class DataVariables(Container):

    def __init__(self):
        self.set("loading", self.loading_var(Container()))
        self.set("caching", self.caching_var(Container()))
        self.set("partitioning", self.partitioning_var(Container()))
        self.set("structure", self.structure_var(Container()))

    def loading_var(self, con):
        con.set("missing_value_code", "NA")
        return con

    def caching_var(self, con):
        con.set("from_cache", True)
        con.set("to_cache", True)
        return con

    def partitioning_var(self, con):
        con.set("spatial_selection", "literal,43,169,348,529,757".split(","))
        con.set("spatial_selection", "literal,43".split(","), "train")
        con.set("spatial_selection", "literal,43".split(","), "valid")
        con.set("spatial_selection", "literal,43".split(","), "test")
        con.set("temporal_selection", "interval,1985-01-01,2019-01-06".split(","))
        con.set("temporal_selection", "interval,1985-01-01,1997-12-31".split(","), "train")
        con.set("temporal_selection", "interval,1998-01-01,2005-12-31".split(","), "valid")
        con.set("temporal_selection", "interval,2006-01-01,2013-12-31".split(","), "test")
        return con

    def structure_var(self, con):
        con.set("data_dir", dataset_dir())
        con.set("cache_dir", os.sep.join([con.get("data_dir"), "Cache"]))
        return con


class SpatiotemporalDataVariables(DataVariables):

    def __init__(self):
        self.set("loading", self.loading_var(Container()))
        self.set("caching", self.caching_var(Container()))
        self.set("partitioning", self.partitioning_var(Container()))
        self.set("structure", self.structure_var(Container()))

    def loading_var(self, con):
        con.copy(DataVariables().get("loading"))
        con.set("header_fields", ["subbasin", "date", "flow", "FLOW_OUTcms", "wind", "PRECIPmm", "tmax", "tmin"])
        con.set("header_nonfeature_fields", ["subbasin"])
        fname = "Spatiotemporal.csv"
        con.set("original_text_filename", fname)
        con.set("original_spatial_labels_text_filename", fname)
        con.set("original_temporal_labels_text_filename", fname)
        con.set("original_n_spatial", 5)
        con.set("original_n_temporal", 12424)
        con.set("spatial_feature", "subbasin")
        con.set("temporal_feature", "date")
        return con

    def caching_var(self, con):
        con.copy(DataVariables().get("caching"))
        con.set("data_type", self.__class__.__name__.replace("DataVariables", ""))
        return con


class SpatialDataVariables(DataVariables):

    def __init__(self):
        self.set("loading", self.loading_var(Container()))
        self.set("caching", self.caching_var(Container()))
        self.set("partitioning", self.partitioning_var(Container()))
        self.set("structure", self.structure_var(Container()))

    def loading_var(self, con):
        con.copy(DataVariables().get("loading"))
        con.set("header_fields", ["FID", "GRIDCODE", "subbasin", "Area_ha", "Lat_1", "Long_1"])
        con.set("header_nonfeature_fields", ["FID", "GRIDCODE", "subbasin"])
        fname = "Spatial.csv"
        con.set("original_text_filename", fname)
        con.set("original_spatial_labels_text_filename", fname)
        con.set("original_n_spatial", 1276)
        con.set("spatial_feature", "subbasin")
        return con

    def caching_var(self, con):
        con.copy(DataVariables().get("caching"))
        con.set("data_type", self.__class__.__name__.replace("DataVariables", ""))
        return con

    def partitioning_var(self, con):
        con.copy(DataVariables().get("partitioning"))
        con.rem("temporal_selection", "*")
        return con


if __name__ == "__main__":
    pass
