import pandas as pd
import numpy as np
import os
import itertools
import Utility as util
import Gather
from Variables import Variables
from Container import Container
from Data.Data import Data
from Plotting import Plotting
from Experimentation.ENVSOFT import *
from Analysis.Analysis import Analysis


class ENVSOFT_Analysis(Analysis):

    def result_dir(self):
        return os.sep.join([__file__.replace(".py", ""), self.name()])


class Analysis6(ENVSOFT_Analysis):

    def __init__(self):
        self.exp = Experiment6()

    def name(self):
        return "InterWatershedFeatureSimilarity"

    def steps(self, args):
#        self.temporal_feature_similarity()
        self.static_feature_similarity()

    def temporal_feature_similarity(self):
        # Init
        var = Variables()
        var.execution.set("dataset", "wabashriver_swat")
        var.execution.set("dataset", "wabashriver_swat", ["train", "valid"])
        var.execution.set("dataset", "littleriver_observed", "test")
        train_spatial_labels = []
        for i in range(8, 16):
            work = self.exp.jobs[i].work
            label = work.datasets.wabashriver_swat.spatiotemporal.partitioning.train__spatial_selection[1]
            train_spatial_labels.append(label)
        test_spatial_labels = self.exp.jobs[8].work.datasets.littleriver_observed.spatiotemporal.partitioning.test__spatial_selection[1:]
        train_spatial_selection = ["literal"] + train_spatial_labels
        test_spatial_selection = ["literal"] + test_spatial_labels
        print(train_spatial_selection)
        print(test_spatial_selection)
        var.datasets.wabashriver_swat.spatiotemporal.partitioning.set(
            "spatial_selection", 
            train_spatial_selection, 
            "*"
        )
        var.datasets.littleriver_observed.spatiotemporal.partitioning.set(
            "spatial_selection", 
            test_spatial_selection, 
            "*"
        )
        data = Data(var)
        # Get data
        train_dataset, test_dataset = data.get("dataset", ["train", "test"])
        train_spatmp = train_dataset.spatiotemporal.original.get("features", "train")
        test_spatmp = test_dataset.spatiotemporal.original.get("features", "test")
        self.results = []
        self.paths = []
        features = ["FLOW_OUTcms", "PRECIPmm", "tmin", "tmax"]
        feature_bins_map = {
            features[0]: 100, 
            features[1]: 150, 
            features[2]: 20, 
            features[3]: 20, 
        }
        feature_xlim_map = {
            features[0]: [-1, 8], 
            features[1]: [-2.5, 15], 
            features[2]: [-30, 30], 
            features[3]: [-20, 45], 
        }
        feature_ylim_map = {
            features[0]: [-0.01, 1.25], 
            features[1]: [-0.001, 0.333], 
            features[2]: [-0.001, 0.1], 
            features[3]: [-0.001, 0.1], 
        }
        for feature in features:
            for i in range(0, train_spatmp.shape[1], 2):
                for j in range(0, test_spatmp.shape[1], 2):
                    line_kwargs = {
                        "color": "black", 
                        "marker": "", 
                        "alpha": 1.0, 
                        "linestyle": "-", 
                        "linewidth": 1.5, 
                    }
                    plt = Plotting()
                    spatial_labels = train_spatial_labels[i:i+2] + test_spatial_labels[j:j+2]
                    n_bins = feature_bins_map[feature]
                    # Plot training distribution(s)
                    f = train_dataset.spatiotemporal.misc.feature_index_map[feature]
                    label = "WRB_{%s}" % (train_spatial_labels[i])
                    plt.plot_density(train_spatmp[:,i,f], label, n_bins=n_bins, line_kwargs=line_kwargs)
                    label = "WRB_{%s}" % (train_spatial_labels[i+1])
                    line_kwargs["color"] = "grey"
#                line_kwargs["marker"] = "*"
                    plt.plot_density(train_spatmp[:,i+1,f], label, n_bins=n_bins, line_kwargs=line_kwargs)
                    # Plot testing distribution(s)
                    f = test_dataset.spatiotemporal.misc.feature_index_map[feature]
                    label = "LRW_{%s}" % (test_spatial_labels[j])
                    line_kwargs["color"] = "maroon"
                    line_kwargs["linestyle"] = "--"
                    plt.plot_density(test_spatmp[:,j,f], label, n_bins=n_bins, line_kwargs=line_kwargs)
                    label = "LRW_{%s}" % (test_spatial_labels[j+1])
                    line_kwargs["marker"] = ""
                    line_kwargs["markevery"] = 2
                    line_kwargs["markersize"] = 3 * line_kwargs["linewidth"]
                    line_kwargs["linestyle"] = "--"
                    line_kwargs["color"] = "orangered"
#                line_kwargs["marker"] = "*"
                    plt.plot_density(test_spatmp[:,j+1,f], label, n_bins=n_bins, line_kwargs=line_kwargs)
                    # Set plotting params
                    if 0:
                        plt.lims(feature_xlim_map[feature], feature_ylim_map[feature])
                    else:
                        plt.lims(feature_xlim_map[feature])
#                    plt.ticks(yticks=[])
                    plt.labels(xlabel=plt.feature_ylabel_map[feature])
                    plt.legend(12)
                    # Write results
                    self.results.append(plt)
                    fname = "Density_Feature[%s]_SpatialElements[%s].png" % (
                        feature, 
                        ",".join(spatial_labels)
                    )
                    print(fname)
                    self.paths.append(os.sep.join([self.result_dir(), fname]))
                    self.results[-1].save_figure(self.paths[-1])

    def static_feature_similarity(self):
        from Data import Data
        from Data.SpatialData import SpatialData
        partitions = ["train", "valid", "test"]
        datasets = ["wabashriver_swat", "wabashriver_swat", "wabashriver_swat"] 
        var = Variables().merge(Container().set("from_cache", False))
        init_var = Data.pull_init_var(var, "wabashriver_swat", "spatial")
        spa = SpatialData(init_var)
        print(spa)
        features = spa.original.features
        # Compute landuse proportions at watershed level
        j = spa.misc.feature_index_map["landuse_types"]
        landuse_prop_map = {}
        for i in range(features.shape[0]):
            landuses = features[i,j]
            props = features[i,j+1]
            for landuse, prop in zip(landuses, props):
                if not landuse in landuse_prop_map:
                    landuse_prop_map[landuse] = 0
                landuse_prop_map[landuse] += prop
        for landuse in landuse_prop_map.keys():
            landuse_prop_map[landuse] /= features.shape[0]
        print("LANDUSES(%d):" % (len(landuse_prop_map)))
        for landuse in util.sort_dict(landuse_prop_map, by="value", ascending=False).keys():
            print("    %s: %.2f" % (landuse, landuse_prop_map[landuse]))
        # Compute soil proportions at watershed level
        j = spa.misc.feature_index_map["soil_types"]
        soil_prop_map = {}
        for i in range(features.shape[0]):
            soils = features[i,j]
            props = features[i,j+1]
            for soil, prop in zip(soils, props):
                if not soil in soil_prop_map:
                    soil_prop_map[soil] = 0
                soil_prop_map[soil] += prop
        for soil in soil_prop_map.keys():
            soil_prop_map[soil] /= features.shape[0]
        print("SOILS(%d):" % (len(soil_prop_map)))
        for soil in util.sort_dict(soil_prop_map, by="value", ascending=False).keys():
            print("    %s: %.2f" % (soil, soil_prop_map[soil]))
        # Compute stats on other numerical properties
        print("Feature Statistics:")
        idx = spa.misc.feature_index_map["river_slope"]
        features[:,idx] /= 100
        for j in spa.transformed.numerical_feature_indices:
            print(4*" "+"%s:" % (spa.misc.index_feature_map[j]))
            stats = util.get_stats(features[:,j])
            for metric, value in stats.items():
                print(8*" "+"%s: %.3f" % (metric, value))


class Analysis11(ENVSOFT_Analysis):

    def __init__(self):
        self.exp = Experiment11()

    def name(self):
        return self.exp.name()

    def steps(self, args):
        self.temporal_reduction_tables()
#        self.temporal_resolution_plots()

    def temporal_reduction_tables(self):
        eval_dir = self.exp.jobs[0].work.evaluation_dir
        chkpt_dir = self.exp.jobs[0].work.checkpoint_dir
        configs = Gather.get_all_configs(eval_dir)
        errors = Gather.get_all_errors(eval_dir)
        infos = Gather.get_all_info(chkpt_dir)
        temporal_reductions = self.exp.temporal_reductions
        mappings = []
        for job in self.exp.jobs[:len(temporal_reductions)]:
            n_temporal_in, n_temporal_out = job.work.get(["n_temporal_in", "n_temporal_out"])
            mappings.append("%d $\longrightarrow$ %d" % (n_temporal_in, n_temporal_out))
        window_sizes = [red[1] for red in temporal_reductions]
        window_strides = [red[2] for red in temporal_reductions]
        features = ["SWmm", "FLOW_OUTcms"]
        feature_abbrev_map = {"SWmm": "SW", "FLOW_OUTcms": "SF"}
        feature_dataset_map = {"SWmm": "wabashriver_swat", "FLOW_OUTcms": "wabashriver_observed"}
        self.paths, self.results = [], []
        param_settings_map = {"Mapping": mappings, "W": window_sizes, "S": window_strides}
        param_settings_map = {"W": window_sizes, "S": window_strides}
        model_errors_map = {}
        for feature in features:
            for model in configs.get_keys():
                key = "%s(%s)" % (model, feature_abbrev_map[feature])
                model_errors_map[key] = []
        info_result_map = {"Time (sec)": []}
        for feature in features:
            dataset = feature_dataset_map[feature]
            for model in configs.get_keys():
                for tmp_red in temporal_reductions:
                    found_con = configs.find("temporal_reduction", tmp_red, path=[model])
                    found_con = found_con.find("dataset", dataset, partition="train")
                    found_con = found_con.find("response_features", [feature])
                    for model_id, config_var in found_con.get_key_values():
                        err_con = errors.get("NRMSE", "test", [model, model_id])
                        info_con = infos.get(model_id, path=[model])
                        for feature, err_map in err_con.get_key_values():
                            key = "%s(%s)" % (model, feature_abbrev_map[feature])
                            model_errors_map[key].append(err_con.get(feature)["529"])
                        if model == "LSTM" and dataset == "wabashriver_swat":
                            runtime = round(float(info_con.get("Runtime(seconds)")), 1)
                            info_result_map["Time (sec)"].append(runtime)
        data_map = util.merge_dicts(param_settings_map, info_result_map)
        data_map = util.merge_dicts(data_map, model_errors_map)
        try:
            df = pd.DataFrame(data_map)
        except ValueError as err:
            print(data_map)
            raise ValueError(err)
        df.columns = [col.replace("GEOMAN", "GeoMAN") for col in df.columns]
        print(df)
        self.paths.append(os.sep.join([self.result_dir(), self.name() + ".tex"]))
        self.results.append(df)
        for path, result in zip(self.paths, self.results):
            with open(path, "w") as f:
                col_frmt = len(result.columns) * "c" 
                f.write(result.to_latex(index=False, escape=False, column_format=col_frmt))

    def temporal_resolution_plots(self):
        eval_dir = self.exp.jobs[0].work.evaluation_dir
        configs = Gather.get_all_configs(eval_dir)
        errors = Gather.get_all_errors(eval_dir)
        evals = Gather.get_all_evals(eval_dir)
        temporal_reductions = [self.exp.temporal_reductions[i] for i in [0,1,3,7]]
#        temporal_reductions = [self.exp.temporal_reductions[i] for i in [7]]
        var = Variables()
        plt = Plotting()
        orig_line_opts, orig_fig_opts = var.plotting.line_options, var.plotting.figure_options
        orig_fig_opts.remove("xticks")
        orig_fig_opts.remove("xlabel")
        feature_ylim_map = {
            "SWmm": [200, 550], 
            "FLOW_OUTcms": [0, 300], 
        }
        for feature in ["SWmm", "FLOW_OUTcms"][:]:
            dataset = "wabashriver_observed"
            if feature == "SWmm":
                dataset = "wabashriver_swat"
            for tmp_red in temporal_reductions:
                for job in self.exp.jobs:
                    if job.work.temporal_reduction == tmp_red and feature in job.work.response_features:
                        work = job.work
                plt.figure((8,4))
                _var = Container().copy(var).merge(work)
                data = Data(_var)
                lstm_con = configs.find("temporal_reduction", tmp_red, path=["LSTM"]).find("dataset", dataset, partition="train").find("response_features", [feature])
                arima_con = configs.find("temporal_reduction", tmp_red, path=["ARIMA"]).find("dataset", dataset, partition="train").find("response_features", [feature])
                lstm_id = list(lstm_con.get_key_values())[0][0]
                arima_id = list(arima_con.get_key_values())[0][0]
                lstm_Yhat = evals.get("Yhat", "test", ["LSTM", lstm_id])
                arima_Yhat = evals.get("Yhat", "test", ["ARIMA", arima_id])
                _var.plotting.plot_dir = os.sep.join([self.result_dir()])
                _var.plotting.figure_options = []
                _var.plotting.line_kwargs["prediction"]["color"] = "Reds"
                _var.plotting.line_kwargs["prediction"]["label"] = "LSTM"
                plt.plot_model_fit(
                    lstm_Yhat, 
                    data.test__dataset.spatiotemporal, 
                    "test", 
                    _var
                )
                _var.plotting.line_options = ["prediction"]
                _var.plotting.figure_options = list(orig_fig_opts)
                if not tmp_red == temporal_reductions[0]:
                    _var.plotting.figure_options.remove("legend")
                _var.plotting.figure_options.remove("save")
                _var.plotting.line_kwargs["prediction"]["color"] = "Blues"
                _var.plotting.line_kwargs["prediction"]["label"] = "ARIMA"
                plt.plot_model_fit(
                    arima_Yhat, 
                    data.test__dataset.spatiotemporal, 
                    "test", 
                    _var
                )
                _var.plotting.line_options = [str(tmp_red)]
                _var.plotting.figure_options = ["save"]
                left, right, bottom, top = plt.lims()
                temporal_labels = data.test__dataset.spatiotemporal.reduced.test__temporal_labels[0]
                if dataset == "wabashriver_observed":
                    start = np.where(temporal_labels == "2008-01-01")[0][0]
                    end = np.where(temporal_labels == "2012-01-24")[0][0]
                else:
                    for tmp_lab in temporal_labels:
                        continue
                        print(tmp_lab)
                    start = np.where(temporal_labels == "2008-01-15")[0][0]
                    end = np.where(temporal_labels == "2012-01-10")[0][0]
                plt.lims([start, end], feature_ylim_map[feature])
                indices = np.linspace(start, end, 7, endpoint=True, dtype=int)
                plt.ticks(
                    [indices, temporal_labels[indices]], 
                    xtick_kwargs={"rotation": 60, "fontsize": 7}
                )
                plt.ticks([])
                plt.labels({1: "Daily", 7: "Weekly", 14: "Bi-Weekly", 28: "Monthly"}[tmp_red[1]])
                if tmp_red == temporal_reductions[-1]:
                    _var.plotting.figure_options.append("xticks")
                plt.plot_model_fit(
                    arima_Yhat, 
                    data.test__dataset.spatiotemporal, 
                    "test", 
                    _var
                )
        self.paths, self.results = [], []


class Analysis12(ENVSOFT_Analysis):

    def __init__(self):
        self.exp = Experiment12()

    def name(self):
        return self.exp.name()

    def steps(self, args):
        eval_dir = self.exp.jobs[0].work.evaluation_dir
        configs = Gather.get_all_configs(eval_dir)
        errors = Gather.get_all_errors(eval_dir)
        evals = Gather.get_all_evals(eval_dir)
        var = Variables()
        plt = Plotting()
        orig_line_opts, orig_fig_opts = var.plotting.line_options, var.plotting.figure_options
        orig_fig_opts.remove("xticks")
        orig_fig_opts.remove("xlabel")
        feature_ylim_map = {
            "SWmm": [200, 550], 
            "FLOW_OUTcms": [0, 300], 
        }
        tmp_red = ["avg", 7, 1]
        for feature in ["FLOW_OUTcms", "SWmm"][:]:
            dataset = "wabashriver_observed"
            if feature == "SWmm":
                dataset = "wabashriver_swat"
            for job in self.exp.jobs:
                if job.work.temporal_reduction == tmp_red and feature in job.work.response_features:
                    work = job.work
            _var = Container().copy(var).merge(work)
            data = Data(_var)
            lstm_con = configs.find("temporal_reduction", tmp_red, path=["LSTM"]).find("dataset", dataset, partition="train").find("response_features", [feature])
            arima_con = configs.find("temporal_reduction", tmp_red, path=["ARIMA"]).find("dataset", dataset, partition="train").find("response_features", [feature])
            lstm_id = list(lstm_con.get_key_values())[0][0]
            arima_id = list(arima_con.get_key_values())[0][0]
            lstm_Yhat = evals.get("Yhat", "test", ["LSTM", lstm_id])
            arima_Yhat = evals.get("Yhat", "test", ["ARIMA", arima_id])
            _var.plotting.plot_dir = os.sep.join([self.result_dir()])
            _var.plotting.figure_options = []
            _var.plotting.line_kwargs["prediction"]["color"] = "Reds"
            _var.plotting.line_kwargs["prediction"]["label"] = "LSTM"
            plt.plot_model_fit(
                lstm_Yhat, 
                data.test__dataset.spatiotemporal, 
                "test", 
                _var
            )
            _var.plotting.line_options = ["prediction"]
            _var.plotting.figure_options = list(orig_fig_opts)
            _var.plotting.figure_options.remove("save")
            _var.plotting.line_kwargs["prediction"]["color"] = "Blues"
            _var.plotting.line_kwargs["prediction"]["label"] = "ARIMA"
            plt.plot_model_fit(
                arima_Yhat, 
                data.test__dataset.spatiotemporal, 
                "test", 
                _var
            )
            _var.plotting.line_options = [str(tmp_red)]
            _var.plotting.figure_options = ["save"]
            left, right, bottom, top = plt.lims()
            temporal_labels = data.test__dataset.spatiotemporal.reduced.test__temporal_labels[0]
            if dataset == "wabashriver_observed":
                start = np.where(temporal_labels == "2008-01-01")[0][0]
                end = np.where(temporal_labels == "2012-01-24")[0][0]
            else:
                for tmp_lab in temporal_labels:
                    continue
                    print(tmp_lab)
                start = np.where(temporal_labels == "2008-01-15")[0][0]
                end = np.where(temporal_labels == "2012-01-10")[0][0]
            plt.lims([start, end], feature_ylim_map[feature])
            indices = np.linspace(start, end, 7, endpoint=True, dtype=int)
            plt.ticks(
                [indices, temporal_labels[indices]], 
                xtick_kwargs={"rotation": 60, "fontsize": 7}
            )
            plt.plot_model_fit(
                arima_Yhat, 
                data.test__dataset.spatiotemporal, 
                "test", 
                _var
            )
        self.paths, self.results = [], []
