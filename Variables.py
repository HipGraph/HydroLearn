import os
import warnings
import importlib
import Utility as util
from Container import Container


class Variables(Container):

    warn_msgs = 1

    def __init__(self, incl_datasets_var=1, incl_models_var=1):
        self.models = Container()
        self.execution = self.execution_var(Container())
        if incl_datasets_var:
            self.datasets = self.datasets_var(Container())
        if incl_models_var:
            self.models = self.models_var(Container())
        self.mapping = self.mapping_var(Container())
        self.plotting = self.plotting_var(Container())
        self.training = self.training_var(Container())
        self.evaluating = self.evaluating_var(Container())
        self.checkpointing = self.checkpointing_var(Container())
        self.distribution = self.distribution_var(Container())
        self.processing = self.processing_var(Container())
        self.debug = self.debug_var(Container())
        self.meta = self.meta_var(Container())

    def debug_var(self, con):
        con.print_args = False
        con.print_args = True
        con.print_vars = False
        con.print_dataset = None
        con.dataset_memory = False
        con.print_spatial_errors = False
        con.print_errors = True
        con.view_model_forward = 0
        return con

    def meta_var(self, con):
        con.id_var_additions = ["training"]
        con.n_id_digits = 5
        con.load_spatial = True
        con.load_temporal = True
        con.load_spatiotemporal = True
        con.load_graph = True
        return con

    def execution_var(self, con):
        con.train = True
        con.evaluate = True
        con.model = "LSTM"
        dataset = "littleriver"
        con.dataset = dataset
        con.set("dataset", dataset, "train")
        con.set("dataset", dataset, "valid")
        con.set("dataset", dataset, "test")
        con.principle_data_type = "spatiotemporal"
        con.principle_data_form = "original"
        con.use_gpu = True
        con.gpu_data_mapping = "all"
        return con

    def plotting_var(self, con):
        con.plot_dir = "Plots"
        con.plot_model_fit = False
        con.plot_model_fit = True
        con.plot_graph = True
        con.plot_graph = False
        con.plot_partitions = ["test"]
        con.spatial_selection = ["random", 5, 0]
        con.n_spatial_per_plot = 1
        con.line_options = ["groundtruth", "prediction"]
        con.figure_options = ["xlabel", "ylabel", "xticks", "yticks", "lims", "legend", "save"]
        con.line_kwargs = {
            "prediction": {"color": "Reds", "label": "Prediction"},
            "groundtruth": {"color": "Greys", "label": "Groundtruth"},
        }
        return con

    def training_var(self, con):
        con.n_epochs = 100
        con.patience = 10
        con.mbatch_size = 64
        con.lr = 0.001
        con.lr_scheduler = None
        con.lr_scheduler_kwargs = {}
        con.param_lr_map = {}
        con.grad_clip = None # None, "norm", or "value"
        con.grad_clip_kwargs = {}
        con.regularization = 0.0
        con.l1_reg = 0.0
        con.l2_reg = 0.0
        con.optimizer = "Adam"
        con.loss = "MSELoss"
        con.loss_kwargs = {}
        con.initializer = None
        con.initialization_seed = 0
        con.batch_shuffle_seed = 0
        return con

    def evaluating_var(self, con):
        con.evaluated_partitions = ["train", "valid", "test"]
        con.evaluation_range = [0.0, 1.0]
        con.evaluation_dir = "Evaluations"
        con.evaluated_checkpoint = "Best.pth"
        con.method = "direct"
        con.metrics = "regression"
        con.mask_metrics = True
        con.cache = False
        con.cache_partitions = ["train", "valid", "test"]
        return con

    def checkpointing_var(self, con):
        con.checkpoint_dir = "Checkpoints"
        con.checkpoint_epochs = -1
        con.log_to_tensorboard = False
        con.tensorboard = Container().set(
            [
                "log_text", 
                "log_losses", 
                "log_gradients", 
                "log_parameters", 
                "log_computational_graph", 
            ], 
            True, 
        )
        con.tensorboard.log_computational_graph = False
        return con

    def distribution_var(self, con):
        con.root_process_rank = 0
        con.process_rank = 0
        con.n_processes = 1
        con.nonroot_process_ranks = util.list_subtract(list(range(con.n_processes)), [con.root_process_rank])
        con.backend = "gloo"
        con.nersc = False
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
        exclusions = ["Model", "ModelTemplate", "Utility", "Trainer", "__init__"] + ["STemGNN", "Radflow", "GEOMAN", "ARIMA"]
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
        con.spatial = Container()
        con.spatial.predictor_features = None
        con.spatial.response_features = None
        con.temporal = Container()
        con.temporal.predictor_features = None
        con.temporal.response_features = None
        con.spatiotemporal = Container()
        con.spatiotemporal.predictor_features = None
        con.spatiotemporal.response_features = None
        con.graph = Container()
        con.graph.edge_weight_feature = None
        con.graph.edge_attr_features = None
        con.temporal_mapping = [7, 1]
        con.horizon = 1
        return con

    def processing_var(self, con):
        con.spatial = Container()
        con.spatial.transform_resolution = ["feature"]
        con.spatial.feature_transform_map = {}
        con.spatial.default_feature_transform = None
        con.temporal = Container()
        con.temporal.transform_resolution = ["feature"]
        con.temporal.feature_transform_map = {}
        con.temporal.default_feature_transform = None
        con.spatiotemporal = Container()
        con.spatiotemporal.transform_resolution = ["spatial", "feature"]
        con.spatiotemporal.feature_transform_map = {}
        con.spatiotemporal.default_feature_transform = None
        con.prediction_adjustment_map = {}
        con.transform_features = True
        con.default_feature_transform = "zscore"
        con.adjust_predictions = True
        con.default_prediction_adjustment = ["identity"]
        con.temporal_reduction = ["avg", 1, 1]
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
                self.training = Container()
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
                self.spatiotemporal = Container()
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
                self.spatial = Container()
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
                self.temporal = Container()
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
                self.graph = Container()
            else:
                raise AttributeError(err)

    def data_var(self, con, data_var):
        con.loading = self.loading_var(Container(), data_var)
        con.caching = self.caching_var(Container(), data_var)
        con.partitioning = self.partitioning_var(Container(), data_var)
        con.structure = self.structure_var(Container(), data_var)
        return con

    def loading_var(self, con, data_var):
        con.copy(data_var.loading)
        return con

    def caching_var(self, con, data_var):
        con.copy(data_var.caching)
        cache_filename_format = data_var.caching.data_type + "_Form[%s]_Data[%s].pkl"
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
        cache_filename_format = data_var.caching.data_type + "_Form[%s]_Data[%s]_%s_%s.pkl"
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
        con.copy(data_var.partitioning)
        return con

    def structure_var(self, con, data_var):
        con.copy(data_var.structure)
        return con


if __name__ == "__main__":
    var = Variables()
    print(var)
