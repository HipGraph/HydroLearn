import os
import Data.DatasetVariables as DV


def dataset_name():
    return "wabashriver_swat"


class DataVariables(DV.DataVariables):

    def missing_value_code(self):   
        return None

    def data_dir(self):
        return os.path.dirname(os.path.realpath(__file__))


class SpatialDataVariables(DataVariables, DV.SpatialDataVariables):
    
    def feature_fields(self):
        feature_fields = [
            "landuse_types",
            "landuse_proportions",
            "soil_types",
            "soil_proportions",
            "area_ha",
            "lat",
            "lon",
            "elevation_min",
            "elevation_max",
            "elevation_mean",
            "elevation_std",
            "river_length",
            "river_slope",
            "river_width",
            "river_depth",
        ]
        return feature_fields

    def categorical_feature_fields(self):
        return ["landuse_types", "landuse_proportions", "soil_types", "soil_proportions"]

    def spatial_label_field(self):
        return "subbasin"


class SpatiotemporalDataVariables(DataVariables, DV.SpatiotemporalDataVariables):

    def feature_fields(self):
        return ["date", "wind", "FLOW_OUTcms", "PRECIPmm", "SWmm", "tmax", "tmin"]

    def categorical_feature_fields(self):
        return None

    def spatial_label_field(self):
        return "subbasin"

    def temporal_label_field(self):
        return "date"

    def temporal_label_format(self):
        return "%Y-%m-%d"

    def temporal_seasonality_period(self):
        return [1, "years"]

    def temporal_resolution(self):
        return [1, "days"]

    def shape(self):
        return ["spatial", "temporal", "feature"]

    def temporal_partition(self):
        selections = [
            ["interval", "1929-01-01", "1997-12-31"], 
            ["interval", "1998-01-01", "2005-12-31"], 
            ["interval", "2006-01-01", "2013-12-31"], 
        ]
        return selections, self.partitions()


class GraphDataVariables(DataVariables, DV.GraphDataVariables):

    def edge_source_field(self):
        return "FROM_NODE"

    def edge_destination_field(self):
        return "TO_NODE"

    def node_label_field(self):
        return "subbasin"