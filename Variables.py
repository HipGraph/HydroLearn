import os
import Utility as util
from Container import Container
from inspect import currentframe


class Variables(Container):

    def __init__(self):
        self.all_var(self)

    def all_var(self, con):
        con.set("execution", self.execution_var(Container()))
        con.set("plotting", self.plotting_var(Container()))
        con.set("training", self.training_var(Container()))
        con.set("evaluating", self.evaluating_var(Container()))
        con.set("checkpointing", self.checkpointing_var(Container()))
        con.set("distribution", self.distribution_var(Container()))
        con.set("datasets", self.datasets_var(Container()))
        con.set("models", self.models_var(Container()))
        con.set("processing", self.processing_var(Container()))
        con.set("graph", self.graph_var(Container()))
        con.set("debug", self.debug_var(Container()))
        return con

    def execution_var(self, con):
        con.set("model", "LSTM")
        dataset = "wabashriver_observed"
        con.set("dataset", dataset)
        con.set("dataset", dataset, "train")
        con.set("dataset", dataset, "valid")
        con.set("dataset", dataset, "test")
        return con

    def plotting_var(self, con):
        con.set("plot_model_fit", False)
        con.set("plot_model_fit", True)
        con.set("n_spatial_per_plot", 1)
        con.set("options", "groundtruth,prediction".split(","))
        con.set("plot_graph_distributions", "degree,connected_component,local_clustering_coefficient,shortest_path".split(","))
        con.set("plot_graph_distributions", "".split(","))
        return con

    def training_var(self, con):
        con.set("train", False)
        con.set("n_epochs", 100)
        con.set("early_stop_epochs", -1)
        con.set("mbatch_size", 128)
        con.set("lr", 0.1)
        con.set("lr_decay", 0.0)
        con.set("reg", 0.0)
        con.set("opt", "adam")
        con.set("opt", "adadelta")
        con.set("loss", "mse")
        con.set("init", "xavier")
        con.set("init_seed", 1)
        con.set("batch_shuf_seed", 1)
        return con

    def evaluating_var(self, con):
        con.set("evaluate", False)
        con.set("evaluation_range", [0.0, 1.0])
        con.set("evaluation_dir", "Evaluations")
        con.set("evaluated_checkpoint", "NULL")
        return con

    def checkpointing_var(self, con):
        con.set("checkpoint_dir", "Checkpoints")
        con.set("chkpt_epochs", -1)
        return con

    def distribution_var(self, con):
        con.set("root_process_rank", 0)
        con.set("process_rank", 0)
        con.set("n_processes", 1)
        con.set("n_processes", con.get("n_processes"), "train")
        con.set("n_processes", 1, "valid")
        con.set("n_processes", 1, "test")
        con.set(
            "nonroot_process_ranks",
            util.get_nonroot_process_ranks(
                con.get("root_process_rank"),
                con.get("n_processes")
            )
        )
        con.set(
            "nonroot_process_ranks",
            util.get_nonroot_process_ranks(
                con.get("root_process_rank"),
                con.get("n_processes", "train")
            ),
            "train"
        )
        con.set(
            "nonroot_process_ranks",
            util.get_nonroot_process_ranks(
                con.get("root_process_rank"),
                con.get("n_processes", "valid")
            ),
            "valid"
        )
        con.set(
            "nonroot_process_ranks",
            util.get_nonroot_process_ranks(
                con.get("root_process_rank"),
                con.get("n_processes", "test")
            ),
            "test"
        )
        con.set("backend", "gloo")
        con.set("nersc", False)
        return con

    def datasets_var(self, con):
        modules = get_modules("Data", "DatasetVariables.py", recurse=True)
        for module in modules:
            con.set(module.dataset_name(), DatasetVariables(module))
        return con

    def models_var(self, con):
        modules = get_modules("Models", "(?=[a-zA-Z_]+\.py)(?!.*Model\.py)(?!__init__\.py)")
        for module in modules:
            con.set(module.model_name(), ModelVariables(module))
        con.set("mapping", self.mapping_var(Container()))
        return con

    def processing_var(self, con):
        con.set("n_daysofyear", 366)
        con.set("temporal_reduction", ["avg", 7, 1])
        con.set("transform_features", True)
        con.set("transformation_resolution", "spatial,feature".split(","))
        con.set(
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
            for feature, transformation in con.get("feature_transformation_map").items():
                con.get("feature_transformation_map")[feature] = "min_max".split(",")
        con.set("adjust_predictions", True)
        prediction_adjustment_map = {
            "FLOW_OUTcms": "limit,0,+".split(","),
        }
        for feature in con.get("feature_transformation_map").keys():
            if feature not in prediction_adjustment_map:
                prediction_adjustment_map[feature] = "none".split(",")
        con.set("prediction_adjustment_map", prediction_adjustment_map)
        return con

    def graph_var(self, con):
        con.set("dynamic_time_warping_cache_filename", "DynamicTimeWarping_Feature[%s]_Type[%s].pkl")
        con.set("features", "FLOW_OUTcms".split(","))
        con.set("similarity_measure", "dynamic_time_warping")
        con.set("similarity_measure", "correlation")
        con.set("construction_method", ["threshold", 0.5])
        con.set("self_loops", True)
        con.set("diffuse", True)
        con.set("diffuse", False)
        con.set("diffusion_method", ["coeff", [2]])
        con.set("diffusion_method", ["heat", 2])
        con.set("diffusion_method", ["ppr", 0.15])
        con.set("diffusion_method", "none")
        con.set("diffusion_sparsification", ["threshold", 0.5])
        con.set("diffusion_sparsification", ["topk", 2])
        con.set("diffusion_sparsification", "none")
        con.set("compute_graph_metrics", "n_nodes,n_edges,average_degree,largest_connected_component,average_local_clustering_coefficient,average_shortest_path_length,diameter".split(","))
        con.set("compute_graph_metrics", "".split(","))
        return con

    def debug_var(self, con):
        con.set("print_data", [True, False])
        con.set("data_memory", False)
        return con

    def mapping_var(self, con):
        con.set("predictor_features", "date,tmin,tmax,PRECIPmm,FLOW_OUTcms".split(","))
        con.set("response_features", "FLOW_OUTcms".split(","))
        con.set("n_temporal_in", 8)
        con.set("n_temporal_out", 1)
        return con


class ModelVariables(Container):

    def __init__(self, model_module):
        self.set(
            "hyperparameters", 
            self.hyperparameter_var(Container(), model_module.HyperparameterVariables())
        )

    def hyperparameter_var(self, con, hyp_var):
        con.copy(hyp_var)
        return con


class DatasetVariables(Container):

    def __init__(self, dataset_module):
        from warnings import warn
        warn_msg = "Class %s was not found in %s. Execution will proceed assuming the data does not exist for this dataset."
        try:
            self.set(
                "spatiotemporal", 
                self.data_var(Container(), dataset_module.SpatiotemporalDataVariables())
            )
        except AttributeError as err:
            if "has no attribute \'SpatiotemporalDataVariables\'" in str(err):
                warn(warn_msg % ("SpatiotemporalDataVariables", dataset_module.__file__), UserWarning)
                self.set("spatiotemporal", None)
            else:
                raise AttributeError(err)
        try:
            self.set(
                "spatial", 
                self.data_var(Container(), dataset_module.SpatialDataVariables())
            )
        except AttributeError as err:
            if "has no attribute \'SpatialDataVariables\'" in str(err):
                warn(warn_msg % ("SpatialDataVariables", dataset_module.__file__), UserWarning)
                self.set("spatial", None)
            else:
                raise AttributeError(err)
        try:
            self.set(
                "temporal", 
                self.data_var(Container(), dataset_module.TemporalDataVariables())
            )
        except AttributeError as err:
            if "has no attribute \'TemporalDataVariables\'" in str(err):
                warn(warn_msg % ("TemporalDataVariables", dataset_module.__file__), UserWarning)
                self.set("temporal", None)
            else:
                raise AttributeError(err)

    def data_var(self, con, data_var):
        con.set("loading", self.loading_var(Container(), data_var))
        con.set("caching", self.caching_var(Container(), data_var))
        con.set("partitioning", self.partitioning_var(Container(), data_var))
        con.set("structure", self.structure_var(Container(), data_var))
        return con

    def loading_var(self, con, data_var):
        con.copy(data_var.get("loading"))
        con.set(
            "header_feature_fields",
            util.list_subtract(con.get("header_fields"), con.get("header_nonfeature_fields"))
        )
        con.set("header_field_index_map", util.to_key_index_dict(con.get("header_fields")))
        con.set("feature_index_map", util.to_key_index_dict(con.get("header_feature_fields")))
        con.set("index_feature_map", util.invert_dict(con.get("feature_index_map")))
        con.set("original_n_features", len(con.get("feature_index_map").keys()))
        return con

    def caching_var(self, con, data_var):
        cache_filename_format = data_var.get("caching").get("data_type") + "_Form[%s]_Data[%s].pkl"
        con.set(
            "original_cache_filename", 
             cache_filename_format % ("Original", "Features")
        )
        con.set(
            "original_spatial_labels_cache_filename", 
             cache_filename_format % ("Original", "SpatialLabels")
        )
        con.set(
            "original_temporal_labels_cache_filename", 
             cache_filename_format % ("Original", "TemporalLabels")
        )
        cache_filename_format = data_var.get("caching").get("data_type") + "_Form[%s]_Data[%s]_%s_%s.pkl"
        con.set(
            "reduced_cache_filename", 
            cache_filename_format % (
                "Reduced", 
                "Features", 
                "TemporalInterval[%s,%s]", 
                "TemporalReduction[%s,%d,%d]", 
            )
        )
        con.set(
            "reduced_temporal_labels_cache_filename", 
            cache_filename_format % (
                "Reduced", 
                "TemporalLabels", 
                "TemporalInterval[%s,%s]", 
                "TemporalReduction[%s,%d,%d]", 
            )
        )
        cache_filename_format = data_var.get("caching").get("data_type") + "Metric_%s_%s_%s.pkl"
        metric_types = ["Minimums", "Maximums", "Medians", "Means", "StandardDeviations"]
        name_prefixes = ["minimums", "maximums", "medians", "means", "standard_deviations"]
        for name_prefix, metric_type in zip(name_prefixes, metric_types):
            con.set(
                name_prefix + "_cache_filename",
                cache_filename_format % (
                    "Type[%s]" % (metric_type),
                    "TemporalInterval[%s,%s]",
                    "TemporalReduction[%s,%d,%d]",
                )
            )
        return con

    def partitioning_var(self, con, data_var):
        con.copy(data_var.get("partitioning"))
        return con

    def structure_var(self, con, data_var):
        con.copy(data_var.get("structure"))
        return con


def get_modules(search_dir, module_regex, recurse=False):
    from importlib import import_module
    from re import match
    module_paths = []
    for root, dirnames, filenames in os.walk(search_dir):
        for filename in filenames:
            if match(module_regex, filename):
                module_paths += [os.path.join(root, filename)]
        if not recurse:
            break
    modules = []
    for module_path in module_paths:
        modules += [import_module(module_path.replace(os.sep, ".").replace(".py", ""))]
    return modules


if __name__ == "__main__":
    var = Variables()
    print(var.to_string(True))
