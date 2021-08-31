import os
from Container import Container


# The name to be used for this dataset (important: will be used as a reference internally)
def dataset_name():
    raise NotImplementedError()


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
        con.set("data_type", self.__class__.__name__.replace("DataVariables", ""))
        return con

    def partitioning_var(self, con):
        return con

    def structure_var(self, con):
        con.set("data_dir", os.path.dirname(os.path.realpath(__file__)))
        con.set("cache_dir", os.sep.join([con.get("data_dir"), "Cache"]))
        return con


class SpatiotemporalDataVariables(DataVariables):

    def __init__(self):
        super(SpatiotemporalDataVariables, self).__init__()

    def loading_var(self, con):
        raise NotImplementedError()
        con.copy(DataVariables().get("loading"))
        con.set("header_fields", ["add", "all", "file", "header", "fields", "here"]) # All header fields
        con.set("header_nonfeature_fields", ["add", "excluded", "header", "fields", "here"]) # Header fields not included as features
        fname = "Spatiotemporal.csv"
        con.set("original_text_filename", fname) # Name of file containing all spatiotemporal data
        con.set("original_spatial_labels_text_filename", fname)
        con.set("original_temporal_labels_text_filename", fname)
        con.set("original_n_spatial", -1) # Number of spatial elements S
        con.set("original_n_temporal", -1) # Number of time-steps T
        con.set("spatial_feature", "") # Header field whose column contains spatial element labels
        con.set("temporal_feature", "") # Header field whose column contains time-step labels
        return con


class SpatialDataVariables(DataVariables):

    def __init__(self):
        super(SpatialDataVariables, self).__init__()

    def loading_var(self, con):
        con.copy(DataVariables().get("loading"))
        con.set("header_fields", ["add", "all", "file", "header", "fields", "here"]) # All header fields
        con.set("header_nonfeature_fields", ["add", "excluded", "header", "fields", "here"]) # Header fields not included as features
        fname = "Spatial.csv"
        con.set("original_text_filename", fname) # Name of file containing all spatial data
        con.set("original_spatial_labels_text_filename", fname)
        con.set("original_n_spatial", -1) # Number of spatial elements S
        con.set("spatial_feature", "") # Header field whose column contains spatial element labels
        return con


if __name__ == "__main__":
    pass
