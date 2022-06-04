import os
import Data.DatasetVariables as DV


def dataset_name():
    return "littleriver_observed"


class DataVariables(DV.DataVariables):
    
    def missing_value_code(self):
        return "NA"

    def data_dir(self):
        return os.path.dirname(os.path.realpath(__file__))


class SpatiotemporalDataVariables(DataVariables, DV.SpatiotemporalDataVariables):

    def feature_fields(self):
        return ["date", "PRECIPmm", "tmax", "tmin", "FLOW_OUTcms"]

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
            ["interval", "1968-01-01", "1995-12-31"], 
            ["interval", "1996-01-01", "1999-12-31"], 
            ["interval", "2000-01-01", "2004-12-31"], 
        ]
        return selections, self.partitions()


class GraphDataVariables(DataVariables, DV.GraphDataVariables):

    def edge_source_field(self):
        return "FROM_NODE"

    def edge_destination_field(self):
        return "TO_NODE"

    def node_label_field(self):
        return "subbasin"
