import Utility as util
from Container import Container


class Variables(Container):

    def __init__(self):
        self.copy(self.get_all_vars())

    def get_all_vars(self):
        container = Container()
        container.set("common", self.get_common_vars())
        container.set("plotting", self.get_plotting_vars())
        container.set("hyperparameters", self.get_hyperparameter_vars())
        container.set("training", self.get_training_vars())
        container.set("evaluating", self.get_evaluating_vars())
        container.set("checkpointing", self.get_checkpointing_vars())
        container.set("distributed", self.get_distributed_vars())
        container.set("littleriver", self.get_littleriver_vars())
        container.set("historical", self.get_historical_vars())
        container.set("observed", self.get_observed_vars())
        container.set("basindata", self.get_basindata_vars())
        container.set("graph", self.get_graph_vars())
        container.set("debug", self.get_debug_vars())
        return container

    def get_debug_vars(self):
        container = Container()
        container.set("print_data", [True, False])
        container.set("data_memory", False)
        return container

    def get_plotting_vars(self):
        container = Container()
        container.set("plot_model_fit", False)
        container.set("plot_model_fit", True)
        container.set("n_spatial_per_plot", 1)
        container.set("options", "groundtruth,prediction".split(","))
        container.set("plot_graph_distributions", "degree,connected_component,local_clustering_coefficient,shortest_path".split(","))
        container.set("plot_graph_distributions", "".split(","))
        return container

    def get_hyperparameter_vars(self):
        container = Container()
        container.set("LSTM", self.get_LSTM_hyperparameter_vars())
        container.set("GNN", self.get_GNN_hyperparameter_vars())
        container.set("GEOMAN", self.get_GEOMAN_hyperparameter_vars())
        container.set("NaiveLastTimestep", self.get_NaiveLastTimestep_hyperparameter_vars())
        container.set("ARIMA", self.get_ARIMA_hyperparameter_vars())
        return container

    def get_LSTM_hyperparameter_vars(self):
        container = Container()
        n = 128
        ratios = [1.0, 1.0]
        container.set("encoding_size", int(ratios[0]*n))
        container.set("decoding_size", int(ratios[1]*n))
        container.set("n_layers", 1)
        container.set("use_bias", True)
        container.set("dropout", 0.0)
        container.set("bidirectional", False)
        container.set("output_activation", "identity")
        return container

    def get_GNN_hyperparameter_vars(self):
        container = Container()
        n = 128
        ratios = [1.0, 1.0, 1.0]
        container.set("temporal_encoding_size", int(ratios[0]*n))
        container.set("spatial_encoding_size", int(ratios[1]*n))
        container.set("temporal_decoding_size", int(ratios[2]*n))
        container.set("n_temporal_layers", 1)
        container.set("n_spatial_layers", 1)
        container.set("spatial_layer_type", "sage")
        container.set("use_bias", True)
        container.set("dropout", 0.0)
        container.set("bidirectional", False)
        return container

    def get_GEOMAN_hyperparameter_vars(self):
        container = Container()
        n = 128
        container.set("n_hidden_encoder", n)
        container.set("n_hidden_decoder", n)
        container.set("n_stacked_layers", 1)
        container.set("s_attn_flag", 2)
        container.set("dropout_rate", 0.0)
        return container

    def get_NaiveLastTimestep_hyperparameter_vars(self):
        container = Container()
        return container

    def get_ARIMA_hyperparameter_vars(self):
        container = Container()
        container.set("order", [0, 0, 0])
        container.set("seasonal_order", [0, 0, 0, 0])
        container.set("trend", None)
        container.set("enforce_stationarity", True)
        container.set("enforce_invertability", True)
        container.set("concentrate_scale", False)
        container.set("trend_offset", 1)
        return container

    def get_training_vars(self):
        container = Container()
        container.set("train", False)
        container.set("n_epochs", 100)
        container.set("early_stop_epochs", -1)
        container.set("mbatch_size", 128)
        container.set("lr", 0.1)
        container.set("lr_decay", 0.0)
        container.set("reg", 0.0)
        container.set("opt", "adam")
        container.set("opt", "adadelta")
        container.set("loss", "mse")
        container.set("init", "xavier")
        container.set("init_seed", 1)
        container.set("batch_shuf_seed", 1)
        return container

    def get_evaluating_vars(self):
        container = Container()
        container.set("evaluate", False)
        container.set("evaluation_range", [0.0, 1.0])
        container.set("evaluation_dir", "Evaluations")
        container.set("evaluated_checkpoint", "NULL")
        return container

    def get_checkpointing_vars(self):
        container = Container()
        container.set("checkpoint_dir", "Checkpoints")
        container.set("chkpt_epochs", -1)
        return container

    def get_distributed_vars(self):
        container = Container()
        container.set("root_process_rank", 0)
        container.set("process_rank", 0)
        container.set("n_processes", 1)
        container.set("n_processes", container.get("n_processes"), "train")
        container.set("n_processes", 1, "valid")
        container.set("n_processes", 1, "test")
        container.set(
            "nonroot_process_ranks",
            util.get_nonroot_process_ranks(
                container.get("root_process_rank"),
                container.get("n_processes")
            )
        )
        container.set(
            "nonroot_process_ranks",
            util.get_nonroot_process_ranks(
                container.get("root_process_rank"),
                container.get("n_processes", "train")
            ),
            "train"
        )
        container.set(
            "nonroot_process_ranks",
            util.get_nonroot_process_ranks(
                container.get("root_process_rank"),
                container.get("n_processes", "valid")
            ),
            "valid"
        )
        container.set(
            "nonroot_process_ranks",
            util.get_nonroot_process_ranks(
                container.get("root_process_rank"),
                container.get("n_processes", "test")
            ),
            "test"
        )
        container.set("backend", "gloo")
        container.set("nersc", False)
        return container

    def get_common_vars(self):
        container = Container()
        container.set("model", "LSTM")
        data_source = "observed"
        container.set("data_source", data_source)
        container.set("data_source", data_source, "train")
        container.set("data_source", data_source, "valid")
        container.set("data_source", data_source, "test")
        container.set("plot_dir", "Plots")
        container.set("data_dir", "Data")
        container.set("dtw_dir", util.path([container.get("data_dir"), "DTW"]))
        container.set("observed_dir", util.path([container.get("data_dir"), "ObservedQ"]))
        container.set("missing_value_code", "NA")
        container.set("n_daysofyear", 366)
        container.set("temporal_reduction", ["avg", 7, 1])
        container.set("n_temporal_in", 8)
        container.set("n_temporal_out", 1)
        container.set("transform_features", True)
        container.set("transformation_resolution", "spatial,feature".split(","))
        container.set(
            "feature_transformation_map",
            {
                "date": "min_max".split(","),
                "wind": "z_score".split(","),
                "ETmm": "z_score".split(","),
                "FLOW_OUTcms": "min_max".split(","),
                "GW_Qmm": "z_score".split(","),
                "PERCmm": "z_score".split(","),
                "PETmm": "z_score".split(","),
                "PRECIPmm": "z_score".split(","),
                "PRECIPmm": "none".split(","),
                "PRECIPmm": "log".split(","),
                "SNOMELTmm": "z_score".split(","),
                "SURQmm": "z_score".split(","),
                "SWmm": "z_score".split(","),
                "tmax": "z_score".split(","),
                "tmax": "none".split(","),
                "tmean": "z_score".split(","),
                "tmin": "z_score".split(","),
                "tmin": "none".split(","),
                "WYLDmm": "z_score".split(","),
                "flow": "z_score".split(","),
            }
        )
        if 1: # Convert all transformations
            for feature, transformation in container.get("feature_transformation_map").items():
                container.get("feature_transformation_map")[feature] = "min_max".split(",")
        container.set("adjust_predictions", True)
        prediction_adjustment_map = {
            "FLOW_OUTcms": "limit,0,+".split(","),
        }
        for feature in container.get("feature_transformation_map").keys():
            if feature not in prediction_adjustment_map:
                prediction_adjustment_map[feature] = "none".split(",")
        container.set("prediction_adjustment_map", prediction_adjustment_map)
        container.set("metric_source_partition", "train")
        container.set("dynamic_time_warping_cache_filename", "DynamicTimeWarping_Feature[%s]_Type[%s].pkl")
        return container

    def get_littleriver_vars(self):
        container = Container()
        container.set("source_name", "Little River")
        container.set(
            "header_fields",
            [
                "subbasin",
                "date",
                "DailyQin",
                "AvgDQcins",
                "MaxQcins",
                "MinQcins",
                "PRECIPmm",
                "tmax",
                "tmin",
                "DailyQmm",
                "FLOW_OUTcms",
                "MaxQcms",
                "MmmQcms",
            ]
        )
        container.set("header_nonfeature_fields", ["subbasin"])
        container.set(
            "header_feature_fields",
            util.list_subtract(container.get("header_fields"), container.get("header_nonfeature_fields"))
        )
        container.set(
            "header_field_index_map",
            util.to_dictionary(
                container.get("header_fields"), 
                [i for i in range(len(container.get("header_fields")))]
            )
        )
        container.set(
            "feature_index_map",
            util.to_dictionary(
                container.get("header_feature_fields"), 
                [i for i in range(len(container.get("header_feature_fields")))]
            )
        )
        container.set("index_feature_map", util.invert_dictionary(container.get("feature_index_map")))
        container.set("original_text_filename", "LittleRiver.csv")
        container.set(
            "original_cache_filename",
            "Spatiotemporal_Form[%s]_Source[%s].pkl" % ("Original", "LittleRiver")
        )
        container.set("original_temporal_labels_text_filename", "LittleRiver.csv")
        container.set(
            "original_temporal_labels_cache_filename",
            "TemporalLabels_Form[%s]_Source[%s].pkl" % ("Original", "LittleRiver")
        )
        container.set("original_spatial_labels_text_filename", "LittleRiver.csv")
        container.set(
            "original_spatial_labels_cache_filename",
            "SpatialLabels_Form[%s]_Source[%s].pkl" % ("Original", "LittleRiver")
        )
        container.set(
            "reduced_cache_filename",
            "Spatiotemporal_Form[%s]_%s_%s_Source[%s].pkl" % (
                "Reduced",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d]",
                "LittleRiver"
            )
        )
        container.set(
            "reduced_temporal_labels_cache_filename",
            "TemporalLabels_Form[%s]_%s_%s_Source[%s]" % (
                "Reduced",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d]",
                "LittleRiver"
            )
        )
        container.set(
            "minimums_cache_filename",
            "SpatiotemporalMetric_Type[%s]_%s_%s_Source[%s].pkl" % (
                "Minimums",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d].pkl",
                "LittleRiver"
            )
        )
        container.set(
            "maximums_cache_filename",
            "SpatiotemporalMetric_Type[%s]_%s_%s_Source[%s].pkl" % (
                "Maximums",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d].pkl",
                "LittleRiver"
            )
        )
        container.set(
            "medians_cache_filename",
            "SpatiotemporalMetric_Type[%s]_%s_%s_Source[%s].pkl" % (
                "Medians",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d].pkl",
                "LittleRiver"
            )
        )
        container.set(
            "means_cache_filename",
            "SpatiotemporalMetric_Type[%s]_%s_%s_Source[%s].pkl" % (
                "Means",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d].pkl",
                "LittleRiver"
            )
        )
        container.set(
            "standard_deviations_cache_filename",
            "SpatiotemporalMetric_Type[%s]_%s_%s_Source[%s].pkl" % (
                "StandardDeviations",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d].pkl",
                "LittleRiver"
            )
        )
        container.set("original_n_temporal", 13515)
        container.set("original_n_spatial", 8)
        container.set("original_n_features", len(container.get("feature_index_map")))
        container.set("spatial_feature", "subbasin")
        container.set("spatial_selection", "literal,B,F,I,J,K,M,N,O".split(","))
        container.set("spatial_selection", "literal,I".split(","), "train")
        container.set("spatial_selection", "literal,I".split(","), "valid")
        container.set("spatial_selection", "literal,I".split(","), "test")
        container.set("temporal_feature", "date")
        container.set("temporal_selection", "interval,1968-01-01,2004-12-31".split(","))
        container.set("temporal_selection", "interval,1968-01-01,1995-12-31".split(","), "train")
        container.set("temporal_selection", "interval,1996-01-01,1999-12-31".split(","), "valid")
        container.set("temporal_selection", "interval,2000-01-01,2004-12-31".split(","), "test")
        container.set("predictor_features", "date,PRECIPmm,FLOW_OUTcms".split(","))
        container.set("response_features", "FLOW_OUTcms".split(","))
        container.set("n_predictors", len(container.get("predictor_features")))
        container.set("n_responses", len(container.get("response_features")))
        return container

    def get_historical_vars(self):
        container = Container()
        container.set(
            "header_fields",
            [
                "GCM",
                "Period",
                "RCP",
                "date",
                "subbasin",
                "wind",
                "ETmm",
                "FLOW_OUTcms",
                "GW_Qmm",
                "PERCmm",
                "PETmm",
                "PRECIPmm",
                "SNOMELTmm",
                "SURQmm",
                "SWmm",
                "tmax",
                "tmean",
                "tmin",
                "WYLDmm"
            ]
        )
        container.set(
            "header_nonfeature_fields",
            [
                "GCM",
                "Period",
                "RCP",
                "subbasin"
            ]
        )
        container.set(
            "header_feature_fields",
            [
                feature for feature in container.get("header_fields") if feature not in container.get("header_nonfeature_fields")
            ]
        )
        container.set(
            "header_field_index_map",
            {
                container.get("header_fields")[i]:i for i in range(len(container.get("header_fields")))
            }
        )
        container.set(
            "feature_index_map",
            {
                container.get("header_feature_fields")[i]:i for i in range(len(container.get("header_feature_fields")))
            }
        )
        container.set(
            "index_feature_map",
            {
                container.get("feature_index_map")[feature]:feature for feature in container.get("header_feature_fields")
            }
        )
        container.set("original_text_filename", "historical.csv")
        container.set(
            "original_cache_filename",
            "Spatiotemporal_Form[%s]_Source[%s].pkl" % ("Original", "Historical")
        )
        container.set("original_temporal_labels_text_filename", "historical.csv")
        container.set(
            "original_temporal_labels_cache_filename",
            "TemporalLabels_Form[%s]_Source[%s].pkl" % ("Original", "Historical")
        )
        container.set("original_spatial_labels_text_filename", "historical.csv")
        container.set(
            "original_spatial_labels_cache_filename",
            "SpatialLabels_Form[%s]_Source[%s].pkl" % ("Original", "Historical")
        )
        container.set(
            "reduced_cache_filename",
            "Spatiotemporal_Form[%s]_%s_%s_Source[%s].pkl" % (
                "Reduced",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d]",
                "Historical"
            )
        )
        container.set(
            "reduced_temporal_labels_cache_filename",
            "TemporalLabels_Form[%s]_%s_%s_Source[%s]" % (
                "Reduced",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d]",
                "Historical"
            )
        )
        container.set(
            "minimums_cache_filename",
            "SpatiotemporalMetric_Type[%s]_%s_%s_Source[%s].pkl" % (
                "Minimums",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d].pkl",
                "Historical"
            )
        )
        container.set(
            "maximums_cache_filename",
            "SpatiotemporalMetric_Type[%s]_%s_%s_Source[%s].pkl" % (
                "Maximums",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d].pkl",
                "Historical"
            )
        )
        container.set(
            "medians_cache_filename",
            "SpatiotemporalMetric_Type[%s]_%s_%s_Source[%s].pkl" % (
                "Medians",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d].pkl",
                "Historical"
            )
        )
        container.set(
            "means_cache_filename",
            "SpatiotemporalMetric_Type[%s]_%s_%s_Source[%s].pkl" % (
                "Means",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d].pkl",
                "Historical"
            )
        )
        container.set(
            "standard_deviations_cache_filename",
            "SpatiotemporalMetric_Type[%s]_%s_%s_Source[%s].pkl" % (
                "StandardDeviations",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d].pkl",
                "Historical"
            )
        )
        container.set("original_n_temporal", 31046)
        container.set("original_n_spatial", 1276)
        container.set("original_n_features", len(container.get("feature_index_map")))
        container.set("spatial_feature", "subbasin")
        container.set("spatial_selection", "interval,1,1276".split(","))
        container.set("spatial_selection", "literal,1".split(","), "train")
        container.set("spatial_selection", "literal,1".split(","), "valid")
        container.set("spatial_selection", "literal,1".split(","), "test")
        container.set("temporal_feature", "date")
        container.set("temporal_selection", "interval,1929-01-01,2013-12-31".split(","))
        container.set("temporal_selection", "interval,1929-01-01,1997-12-31".split(","), "train")
        container.set("temporal_selection", "interval,1998-01-01,2005-12-31".split(","), "valid")
        container.set("temporal_selection", "interval,2006-01-01,2013-12-31".split(","), "test")
        container.set("predictor_features", "date,tmin,tmax,PRECIPmm,SWmm,FLOW_OUTcms".split(","))
        container.set("response_features", "FLOW_OUTcms".split(","))
        container.set("n_predictors", len(container.get("predictor_features")))
        container.set("n_responses", len(container.get("response_features")))
        return container

    def get_observed_vars(self):
        container = Container()
        container.set("header_fields", ["subbasin", "date", "flow", "FLOW_OUTcms"])
        container.set("header_fields", ["subbasin", "date", "flow", "FLOW_OUTcms", "wind", "PRECIPmm", "tmax", "tmin"])
        container.set("header_nonfeature_fields", ["subbasin"])
        container.set(
            "header_feature_fields",
            [
                feature for feature in container.get("header_fields") if feature not in container.get("header_nonfeature_fields")
            ]
        )
        container.set(
            "header_field_index_map",
            {
                container.get("header_fields")[i]:i for i in range(len(container.get("header_fields")))
            }
        )
        container.set(
            "feature_index_map",
            {
                container.get("header_feature_fields")[i]:i for i in range(len(container.get("header_feature_fields")))
            }
        )
        container.set(
            "index_feature_map",
            {
                container.get("feature_index_map")[feature]:feature for feature in container.get("header_feature_fields")
            }
        )
        container.set("original_text_filename", "observed.csv")
        container.set(
            "original_cache_filename",
            "Spatiotemporal_Form[%s]_Source[%s].pkl" % ("Original", "Observed")
        )
        container.set("original_temporal_labels_text_filename", "observed.csv")
        container.set(
            "original_temporal_labels_cache_filename",
            "TemporalLabels_Form[%s]_Source[%s].pkl" % ("Original", "Observed")
        )
        container.set("original_spatial_labels_text_filename", "observed.csv")
        container.set(
            "original_spatial_labels_cache_filename",
            "SpatialLabels_Form[%s]_Source[%s].pkl" % ("Original", "Observed")
        )
        container.set(
            "reduced_cache_filename",
            "Spatiotemporal_Form[%s]_%s_%s_Source[%s].pkl" % (
                "Reduced",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d]",
                "Observed"
            )
        )
        container.set(
            "reduced_temporal_labels_cache_filename",
            "TemporalLabels_Form[%s]_%s_%s_Source[%s]" % (
                "Reduced",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d]",
                "Observed"
            )
        )
        container.set(
            "minimums_cache_filename",
            "SpatiotemporalMetric_Type[%s]_%s_%s_Source[%s].pkl" % (
                "Minimums",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d].pkl",
                "Observed"
            )
        )
        container.set(
            "maximums_cache_filename",
            "SpatiotemporalMetric_Type[%s]_%s_%s_Source[%s].pkl" % (
                "Maximums",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d].pkl",
                "Observed"
            )
        )
        container.set(
            "medians_cache_filename",
            "SpatiotemporalMetric_Type[%s]_%s_%s_Source[%s].pkl" % (
                "Medians",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d].pkl",
                "Observed"
            )
        )
        container.set(
            "means_cache_filename",
            "SpatiotemporalMetric_Type[%s]_%s_%s_Source[%s].pkl" % (
                "Means",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d].pkl",
                "Observed"
            )
        )
        container.set(
            "standard_deviations_cache_filename",
            "SpatiotemporalMetric_Type[%s]_%s_%s_Source[%s].pkl" % (
                "StandardDeviations",
                "TemporalInterval[%s,%s]",
                "TemporalReduction[%s,%d,%d].pkl",
                "Observed"
            )
        )
        container.set("original_n_temporal", 12424)
        container.set("original_n_spatial", 5)
        container.set("original_n_features", len(container.get("feature_index_map")))
        container.set("spatial_feature", "subbasin")
        container.set("spatial_selection", "literal,43,169,348,529,757".split(","))
        container.set("spatial_selection", "literal,529".split(","), "train")
        container.set("spatial_selection", "literal,529".split(","), "valid")
        container.set("spatial_selection", "literal,529".split(","), "test")
        container.set("temporal_feature", "date")
        container.set("temporal_selection", "interval,1985-01-01,2013-12-31".split(","))
        container.set("temporal_selection", "interval,1985-01-01,1997-12-31".split(","), "train")
        container.set("temporal_selection", "interval,1998-01-01,2005-12-31".split(","), "valid")
        container.set("temporal_selection", "interval,2006-01-01,2013-12-31".split(","), "test")
        container.set("predictor_features", "date,tmin,tmax,PRECIPmm,FLOW_OUTcms".split(","))
        container.set("response_features", "FLOW_OUTcms".split(","))
        container.set("n_predictors", len(container.get("predictor_features")))
        container.set("n_responses", len(container.get("response_features")))
        return container

    def get_basindata_vars(self):
        container = Container()
        container.set(
            "header_fields",
            [
                "FID",
                "GRIDCODE",
                "subbasin",
                "Area_ha",
                "Lat_1",
                "Long_1"
            ]
        )
        container.set(
            "header_nonfeature_fields",
            [
                "FID",
                "GRIDCODE",
                "subbasin",
            ]
        )
        container.set(
            "header_feature_fields",
            [
                feature for feature in container.get("header_fields") if feature not in container.get("header_nonfeature_fields")
            ]
        )
        container.set(
            "header_field_index_map",
            {
                container.get("header_fields")[i]:i for i in range(len(container.get("header_fields")))
            }
        )
        container.set(
            "feature_index_map",
            {
                container.get("header_feature_fields")[i]:i for i in range(len(container.get("header_feature_fields")))
            }
        )
        container.set("original_text_filename", "basindata.csv")
        container.set(
            "original_cache_filename",
            "Spatial_Form[%s]_Source[%s].pkl" % ("Original", "BasinData")
        )
        container.set("original_spatial_labels_text_filename", "basindata.csv")
        container.set(
            "original_spatial_labels_cache_filename",
            "SpatialLabels_Form[%s]_Source[%s].pkl" % ("Original", "BasinData")
        )
        container.set("original_n_spatial", 1276)
        container.set("spatial_feature", "subbasin")
        container.set("original_n_features", len(container.get("feature_index_map")))
        container.set("spatial_selection", "interval,1,1276".split(","))
        container.set("spatial_selection", "literal,1".split(","), "train")
        container.set("spatial_selection", "literal,1".split(","), "valid")
        container.set("spatial_selection", "literal,1,2".split(","), "test")
        return container

    def get_graph_vars(self):
        container = Container()
        container.set("features", "FLOW_OUTcms".split(","))
        container.set("similarity_measure", "dynamic_time_warping")
        container.set("similarity_measure", "correlation")
        container.set("construction_method", ["threshold", 0.5])
        container.set("self_loops", True)
        container.set("diffuse", True)
        container.set("diffuse", False)
        container.set("diffusion_method", ["coeff", [2]])
        container.set("diffusion_method", ["heat", 2])
        container.set("diffusion_method", ["ppr", 0.15])
        container.set("diffusion_method", "none")
        container.set("diffusion_sparsification", ["threshold", 0.5])
        container.set("diffusion_sparsification", ["topk", 2])
        container.set("diffusion_sparsification", "none")
        container.set("compute_graph_metrics", "n_nodes,n_edges,average_degree,largest_connected_component,average_local_clustering_coefficient,average_shortest_path_length,diameter".split(","))
        container.set("compute_graph_metrics", "".split(","))
        return container


if __name__ == "__main__":
    var = Variables()
    print(var.get("historical").get("feature_index_map"))
