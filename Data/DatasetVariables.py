import os
from pandas import read_csv
from Data.DataSelection import DataSelection
from Container import Container


def dataset_name():
    raise NotImplementedError()


class DataVariables(Container):

    def __init__(self):
        self.set("loading", self.loading_var(Container()))
        self.set("caching", self.caching_var(Container()))
        self.set("partitioning", self.partitioning_var(Container()))
        self.set("structure", self.structure_var(Container()))

    def loading_var(self, con):
        raise NotImplementedError()

    def caching_var(self, con):
        con.set("from_cache", self.from_cache())
        con.set("to_cache", self.to_cache())
        con.set("data_type", self.data_type())
        return con

    def partitioning_var(self, con):
        return con

    def structure_var(self, con):
        con.set("data_dir", self.data_dir())
        con.set("cache_dir", self.cache_dir())
        return con

    def precision(self): # Number of bits used for data (optimize this to reduced memory load)
        return 32

    def missing_value_code(self): # Alternative name for missing values (supplied to Pandas)
        raise NotImplementedError()

    def from_cache(self): # Whether or not to load reused data with pickle
        return True

    def to_cache(self): # Whether or not to save reused data with pickle
        return True

    def data_type(self): # One of: "Spatial", "Temporal", "Spatiotemporal", or "Graph" for naming cache data
        return self.__class__.__name__.replace("DataVariables", "")

    def feature_fields(self): # Column names that denote features (only these will be used & cached)
        raise NotImplementedError()

    def categorical_feature_fields(self): # Feature columns to be treated as categorical
        raise NotImplementedError()

    def features_filename(self): # Name of file containing feature data
        raise NotImplementedError()

    def partitions(self):
        return ["train", "valid", "test"]

    def data_dir(self): # Directory where original data exists
        raise NotImplementedError()

    def cache_dir(self): # Directory where cached data exists
        return os.sep.join([self.data_dir(), "Cache"])


class SpatialDataVariables(DataVariables):

    def __init__(self):
        super(SpatialDataVariables, self).__init__()

    def loading_var(self, con):
        con.set("precision", self.precision())
        con.set("missing_value_code", self.missing_value_code())
        con.set("feature_fields", self.feature_fields())
        con.set("categorical_fields", self.categorical_feature_fields())
        con.set("spatial_label_field", self.spatial_label_field())
        con.set("original_text_filename", self.features_filename())
        con.set("original_spatial_labels_text_filename", self.spatial_labels_filename())
        return con

    def partitioning_var(self, con):
        spatial_selections, partitions = self.spatial_partition()
        con.set("spatial_selection", ["all"])
        con.set("spatial_selection", spatial_selections, partitions, multi_value=True)
        return con

    def spatial_label_field(self): # Name of column containing labels for all spatial elements
        raise NotImplementedError()

    def features_filename(self):
        return "Spatial.csv"

    def spatial_labels_filename(self): # Name of file containing the labels for all spatial elements
        return "SpatialLabels.csv"

    def spatial_partition_from_split(self):
        return False

    def spatial_split(self): # Ratio of spatial elements split among partitions (e.g. [8, 1, 1] => 8:1:1)
        raise NotImplementedError()

    def spatial_partition(self): # Spatial elements assigned to each partition
        selections = [["all"] for _ in self.partitions()]
        if self.spatial_partition_from_split():
            df = read_csv(os.sep.join([self.data_dir(), self.spatial_labels_filename()]))
            spatial_labels = df[self.spatial_label_field()].to_numpy(str).reshape(-1)
            selections = DataSelection().selections_from_split(spatial_labels, self.spatial_split())
        return selections, self.partitions()


class TemporalDataVariables(DataVariables):

    def __init__(self):
        super(TemporalDataVariables, self).__init__()

    def loading_var(self, con):
        con.set("precision", self.precision())
        con.set("missing_value_code", self.missing_value_code())
        con.set("feature_fields", self.feature_fields())
        con.set("categorigcal_fields", self.categorical_feature_fields())
        con.set("temporal_label_field", self.temporal_label_field())
        con.set("original_text_filename", self.features_filename())
        con.set("original_temporal_labels_text_filename", self.temporal_labels_filename())
        return con

    def partitioning_var(self, con):
        temporal_selections, partitions = self.temporal_partition()
        con.set("temporal_selection", ["all"])
        con.set("temporal_selection", temporal_selections, partitions, multi_value=True)
        return con

    def temporal_label_field(self): # Name of column containing labels for all temporal elements
        raise NotImplementedError()

    def temporal_label_format(self): # Format (from package datetime) used temporal label representation
        raise NotImplementedError()

    def temporal_seasonality_period(self): # Duration of seasonality period (e.g. [1, "days"], [1, "years"], etc)
        raise NotImplementedError()

    def temporal_resolution(self): # Duration between temporal elements (e.g. [5, "minutes"], [1, "days"], etc)
        raise NotImplementedError()

    def features_filename(self): # Name of file containing feature data
        return "Temporal.csv"

    def temporal_labels_filename(self): # Name of file containing the labels for all temporal elements
        return "TemporalLabels.csv"

    def temporal_partition_from_split(self): # Whether or not to construct the partition from a given split
        return False

    def temporal_split(self): # Ratio of temporal elements split among partitions (e.g. [8, 1, 1] => 8:1:1)
        raise NotImplementedError()

    def temporal_partition(self): # Temporal elements assigned to each partition
        selections = [["all"] for _ in self.partitions()]
        if self.temporal_partition_from_split():
            df = read_csv(os.sep.join([self.data_dir(), self.temporal_labels_filename()]))
            temporal_labels = df[self.temporal_label_field()].to_numpy(str).reshape(-1)
            selections = DataSelection().selections_from_split(temporal_labels, self.temporal_split())
        return selections, self.partitions()


class SpatiotemporalDataVariables(SpatialDataVariables, TemporalDataVariables):

    def __init__(self):
        super(SpatiotemporalDataVariables, self).__init__()

    def loading_var(self, con):
        con.set("precision", self.precision())
        con.set("missing_value_code", self.missing_value_code())
        con.set("feature_fields", self.feature_fields())
        con.set("categorical_fields", self.categorical_feature_fields())
        con.set("spatial_label_field", self.spatial_label_field())
        con.set("temporal_label_field", self.temporal_label_field())
        con.set("temporal_label_format", self.temporal_label_format())
        con.set("temporal_seasonality_period", self.temporal_seasonality_period())
        con.set("temporal_resolution", self.temporal_resolution())
        con.set("shape", self.shape())
        con.set("original_text_filename", self.features_filename())
        con.set("original_spatial_labels_text_filename", self.spatial_labels_filename())
        con.set("original_temporal_labels_text_filename", self.temporal_labels_filename())
        return con

    def partitioning_var(self, con):
        spatial_selections, partitions = self.spatial_partition()
        con.set("spatial_selection", ["all"])
        con.set("spatial_selection", spatial_selections, partitions, multi_value=True)
        temporal_selections, partitions = self.temporal_partition()
        con.set("temporal_selection", ["all"])
        con.set("temporal_selection", temporal_selections, partitions, multi_value=True)
        return con

    # Ordering of features which include one of:
    #   spatial-major: ["spatial", "temporal", "feature"]
    #   temporal-major: ["temporal", "spatial", "feature"]
    def shape(self):
        raise NotImplementedError()

    def features_filename(self):
        return "Spatiotemporal.csv.gz"


class GraphDataVariables(DataVariables):

    def __init__(self):
        super(GraphDataVariables, self).__init__()

    def loading_var(self, con):
        con.set("precision", self.precision())
        con.set("missing_value_code", self.missing_value_code())
        con.set("source_field", self.edge_source_field())
        con.set("destination_field", self.edge_destination_field())
        con.set("weight_field", self.edge_weight_field())
        con.set("node_label_field", self.node_label_field())
        con.set("original_text_filename", self.features_filename())
        con.set("original_node_labels_text_filename", self.node_labels_filename())
        return con

    def partitioning_var(self, con):
        node_selections, partitions = self.node_partition()
        con.set("node_selection", ["all"])
        con.set("node_selection", node_selections, partitions, multi_value=True)
        return con

    def edge_source_field(self): # Name of column containing the source node of each edge
        raise NotImplementedError()

    def edge_destination_field(self): # Name of column containing the destination node of each edge
        raise NotImplementedError()

    def edge_weight_field(self): # Name of column containing the weight of each edge
        return None

    def node_label_field(self): # Name of column containing labels for all graph nodes
        raise NotImplementedError()

    def features_filename(self):
        return "Graph.csv"

    def node_labels_filename(self): # Name of file containing the labels for all graph nodes
        return "SpatialLabels.csv"

    def node_partition_from_split(self):
        return False

    def node_split(self): # Ratio of Graph nodes split among partitions (e.g. [8, 1, 1] => 8:1:1)
        raise NotImplementedError()

    def node_partition(self): # Spatial elements assigned to each partition
        selections = [["all"] for _ in self.partitions()]
        if self.node_partition_from_split():
            df = read_csv(os.sep.join([self.data_dir(), self.node_labels_filename()]))
            node_labels = df[self.node_label_field()].to_numpy(str).reshape(-1)
            selections = DataSelection().selections_from_split(node_labels, self.node_split())
        return selections, self.partitions()
