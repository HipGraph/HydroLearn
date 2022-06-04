import os
import warnings
import importlib
import Utility as util
from Container import Container


class Variables(Container):

    warn_msgs = True

    def __init__(self):
        self.set("execution", self.execution_var(Container()))
        self.set("datasets", self.datasets_var(Container()))
        self.set("models", self.models_var(Container()))
        self.set("mapping", self.mapping_var(Container()))
        self.set("plotting", self.plotting_var(Container()))
        self.set("training", self.training_var(Container()))
        self.set("evaluating", self.evaluating_var(Container()))
        self.set("checkpointing", self.checkpointing_var(Container()))
        self.set("distribution", self.distribution_var(Container()))
        self.set("processing", self.processing_var(Container()))
        self.set("graph", self.graph_var(Container()))
        self.set("debug", self.debug_var(Container()))

    def debug_var(self, con):
        con.set("print_args", False)
        con.set("print_args", True)
        con.set("print_vars", False)
        con.set("print_data", False)
        con.set("data_memory", False)
        con.set("print_spatial_errors", False)
        con.set("print_errors", True)
        return con

    def execution_var(self, con):
        con.set("model", "LSTM")
        dataset = "littleriver_observed"
        con.set("dataset", dataset)
        con.set("dataset", dataset, "train")
        con.set("dataset", dataset, "valid")
        con.set("dataset", dataset, "test")
        return con

    def plotting_var(self, con):
        con.set("plot_dir", "Plots")
        con.set("plot_model_fit", False)
        con.set("plot_model_fit", True)
        con.set("plot_graph", True)
        con.set("plot_graph", False)
        con.set("plot_partitions", ["test"])
        con.set("n_spatial_per_plot", 1)
        con.set("line_options", ["groundtruth", "prediction"])
        con.set("figure_options", ["xlabel", "ylabel", "xticks", "yticks", "lims", "legend", "save"])
        con.set(
            "line_kwargs", 
            {
                "prediction": {"color": "Reds", "label": "Prediction"}, 
                "groundtruth": {"color": "Greys", "label": "Groundtruth"}, 
            }
        )
        con.set(
            "plot_graph_distributions", 
            ["degree", "connected_component", "local_clustering_coefficient", "shortest_path"]
        )
        con.set("plot_graph_distributions", [""])
        return con

    def training_var(self, con):
        con.set("train", True)
        con.set("n_epochs", 50)
        con.set("early_stop_epochs", -1)
        con.set("mbatch_size", 128)
        con.set("lr", 0.001)
        con.set("lr_decay", 0.0)
        con.set("gradient_clip", 5)
        con.set("regularization", 0.0)
        con.set("optimizer", "Adam")
        con.set("loss", "MSELoss")
        con.set("initializer", None)
        con.set("initialization_seed", 0)
        con.set("batch_shuffle_seed", 0)
        con.set("use_gpu", True)
        con.set("gpu_data_mapping", "all")
        return con

    def evaluating_var(self, con):
        con.set("evaluate", True)
        con.set("evaluated_partitions", ["train", "valid", "test"])
        con.set("evaluation_range", [0.0, 1.0])
        con.set("evaluation_dir", "Evaluations")
        con.set("evaluated_checkpoint", "Best")
        con.set("method", "direct")
        con.set("metrics", ["MAE", "MSE", "MAPE", "RMSE", "NRMSE"])
        con.set("cache", False)
        con.set("cache_partitions", ["train", "valid", "test"])
        return con

    def checkpointing_var(self, con):
        con.set("checkpoint_dir", "Checkpoints")
        con.set("checkpoint_epochs", -1)
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
            util.get_nonroot_process_ranks(con.get("root_process_rank"), con.get("n_processes"))
        )
        con.set(
            "nonroot_process_ranks",
            util.get_nonroot_process_ranks(con.get("root_process_rank"), con.get("n_processes", "train")),
            "train"
        )
        con.set(
            "nonroot_process_ranks",
            util.get_nonroot_process_ranks(con.get("root_process_rank"), con.get("n_processes", "valid")),
            "valid"
        )
        con.set(
            "nonroot_process_ranks",
            util.get_nonroot_process_ranks(con.get("root_process_rank"), con.get("n_processes", "test")),
            "test"
        )
        con.set("backend", "gloo")
        con.set("nersc", False)
        return con

    def datasets_var(self, con):
        module_paths = util.get_paths("Data", "DatasetVariables.py", recurse=True)
        for module_path in module_paths:
            if os.sep.join(["Data", "DatasetVariables.py"]) in module_path:
                to_remove = module_path
                break
        module_paths.remove(to_remove)
        import_statements = [path.replace(os.sep, ".").replace(".py", "") for path in module_paths]
        modules = [importlib.import_module(import_statement) for import_statement in import_statements]
        for module in modules:
            con.set(module.dataset_name(), DatasetVariables(module))
        return con

    def models_var(self, con):
        # Match modules named by one or more: digits, characters, underscores, and/or dashes
        path_regex = "(?=[0-9a-zA-Z_-]+\.py)"
        exclusions = ["Model", "ModelTemplate", "__init__"]
        for excl in exclusions:
            path_regex += "(?!%s\.py)" % (excl)
        module_paths = util.get_paths("Models", path_regex)
        import_statements = [path.replace(os.sep, ".").replace(".py", "") for path in module_paths]
        modules = []
        for module_path, import_statement in zip(module_paths, import_statements):
            try:
                modules.append(importlib.import_module(import_statement))
            except ImportError as err:
                if self.warn_msgs:
                    warnings.warn(
                        "Failed to import module \"%s\" due to ImportError: \"%s\". Execution will proceed without this module." % (import_statement, str(err)), 
                        UserWarning, 
                    )
        for module in modules:
            con.set(module.model_name(), ModelVariables(module))
        return con

    def mapping_var(self, con):
        con.set("predictor_features", ["date", "tmin", "tmax", "PRECIPmm", "SWmm", "FLOW_OUTcms"])
        con.set("predictor_features", ["date", "tmin", "tmax", "PRECIPmm", "FLOW_OUTcms"])
#        con.set("predictor_features", ["tmin", "tmax", "PRECIPmm", "FLOW_OUTcms"])
        con.set("response_features", ["FLOW_OUTcms"])
        con.set("temporal_mapping", [7, 1])
        return con

    def processing_var(self, con):
        con.set("metric_source_partition", "train")
        con.set("temporal_reduction", ["avg", 1, 1])
        con.set("transformation_resolution", ["spatial", "feature"])
        con.set("transform_features", True)
        con.set("feature_transformation_map", {})
        con.set("default_feature_transformation", ["z_score"])
        con.set("adjust_predictions", True)
        con.set("prediction_adjustment_map", {})
        con.set("default_prediction_adjustment", ["none"])
        return con

    def graph_var(self, con):
        return con


class ModelVariables(Container):

    def __init__(self, model_module):
        warn_msg = "Class %s was not found in %s. Execution will proceed using default settings."
        self.set(
            "hyperparameters", 
            self.hyperparameter_var(Container(), model_module.HyperparameterVariables())
        )
        try:
            self.set(
                "training", 
                self.training_var(Container(), model_module.TrainingVariables())
            )
        except AttributeError as err:
            if "has no attribute \'TrainingVariables\'" in str(err):
                if False and self.warn_msgs:
                    warnings.warn(warn_msg % ("TrainingVariables", model_module.__file__), UserWarning)
                self.set("training", Container())
            else:
                raise AttributeError(err)

    def hyperparameter_var(self, con, hyp_var):
        con.copy(hyp_var)
        return con

    def training_var(self, con, train_var):
        con.copy(train_var)
        return con


class DatasetVariables(Container):

    def __init__(self, dataset_module):
        warn_msg = "Class %s was not found in %s. Execution will proceed assuming the data does not exist for this dataset."
        try:
            self.set(
                "spatiotemporal", 
                self.data_var(Container(), dataset_module.SpatiotemporalDataVariables())
            )
        except AttributeError as err:
            if "has no attribute \'SpatiotemporalDataVariables\'" in str(err):
                if False and self.warn_msgs:
                    warnings.warn(warn_msg % ("SpatiotemporalDataVariables", dataset_module.__file__), UserWarning)
                self.set("spatiotemporal", Container())
            else:
                raise AttributeError(err)
        try:
            self.set(
                "spatial", 
                self.data_var(Container(), dataset_module.SpatialDataVariables())
            )
        except AttributeError as err:
            if "has no attribute \'SpatialDataVariables\'" in str(err):
                if False and self.warn_msgs:
                    warnings.warn(warn_msg % ("SpatialDataVariables", dataset_module.__file__), UserWarning)
                self.set("spatial", Container())
            else:
                raise AttributeError(err)
        try:
            self.set(
                "temporal", 
                self.data_var(Container(), dataset_module.TemporalDataVariables())
            )
        except AttributeError as err:
            if "has no attribute \'TemporalDataVariables\'" in str(err):
                if False and self.warn_msgs:
                    warnings.warn(warn_msg % ("TemporalDataVariables", dataset_module.__file__), UserWarning)
                self.set("temporal", Container())
            else:
                raise AttributeError(err)
        try:
            self.set(
                "graph", 
                self.data_var(Container(), dataset_module.GraphDataVariables())
            )
        except AttributeError as err:
            if "has no attribute \'GraphDataVariables\'" in str(err):
                if False and self.warn_msgs:
                    warnings.warn(warn_msg % ("GraphDataVariables", dataset_module.__file__), UserWarning)
                self.set("graph", Container())
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
        return con

    def caching_var(self, con, data_var):
        con.copy(data_var.get("caching"))
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
        return con

    def partitioning_var(self, con, data_var):
        con.copy(data_var.get("partitioning"))
        return con

    def structure_var(self, con, data_var):
        con.copy(data_var.get("structure"))
        return con


if __name__ == "__main__":
    var = Variables()
    print(var)
