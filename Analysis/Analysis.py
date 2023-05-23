import os
import sys
import numpy as np
import pandas as pd

import Gather
import Utility as util
from Container import Container
from Plotting import Plotting


class Analysis:

    def __init__(self):
        pass

    def name(self):
        return "Analysis"

    def run(self, args):
        os.makedirs(self.result_dir(), exist_ok=True)
        self.steps(args)
        self.write(args)

    def steps(self, args):
        pass

    def write(self, args):
        pass

    def result_dir(self):
        return os.sep.join([__file__.replace(".py", ""), self.name()])


class ValuesOverParameterAnalysis(Analysis):

    def name(self):
        raise NotImplementedError()

    def default_var(self):
        return Container().set(
            [
                "exp", 
                "values",
                "value_partitions",
                "value_containers",
                "parameter",
                "parameter_partition", 
                "parameter_container",
                "response_feature",
                "model",
                "plot_types",
                "plot_kwargs",
                "normalize",
                "debug",
            ],
            [
                None, 
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "bar",
                {"alpha": 0.5},
                False,
                0,
            ],
            multi_value=True
        )

    def var(self):
        return self.default_var()

    def steps(self, args):
        values_over_parameter_analysis(self, self.var().merge(args))


class ValuesOverParameterGridAnalysis(Analysis):

    def name(self):
        raise NotImplementedError()

    def default_var(self):
        return Container().set(
            [
                "exp", 
                "values",
                "value_partitions",
                "value_containers",
                "parameters",
                "parameter_partitions", 
                "parameter_containers",
                "response_feature",
                "model",
                "plot_types",
                "plot_kwargs",
                "normalize",
                "debug",
            ],
            [
                None, 
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "bar",
                {"alpha": 0.5},
                False,
                0,
            ],
            multi_value=True
        )

    def var(self):
        return self.default_var()

    def steps(self, args):
        values_over_parameter_grid_analysis(self, self.var().merge(args))


def values_and_partitions(mode=0):
    if mode == 0:
        values = ["n_parameters", "epoch_runtime", "UPR", "OPR", "MR", "RMSE"]
        partitions = [None, None, "test", "test", "test", "*"]
    elif mode == 1:
        values = ["n_parameters", "epoch_runtime", "MAE", "RMSE", "MAPE", "RRSE", "CORR"]
        partitions = [None, None, "test", "test", "test", "test", "test"]
    elif mode == 2:
        values = ["MAE", "RMSE", "MAPE", "RRSE", "CORR"]
        partitions = ["test", "test", "test", "test", "test"]
    elif mode == 3:
        values = ["MAE", "RMSE", "MAPE"]
        partitions = ["*", "*", "*"]
    else:
        raise ValueError(mode)
    return [values, partitions]


class ModelComparison(ValuesOverParameterAnalysis):

    def name(self):
        return "ModelComparison"

    def var(self):
        var = self.default_var()
        var.set(["values", "value_partitions"], values_and_partitions(1), multi_value=1)
        var.parameter = "model"
        var.model = ""
        var.round_to = 3
        return var


class ParameterSearch(ValuesOverParameterAnalysis):

    def name(self):
        return "ParameterSearch"

    def var(self):
        var = self.default_var()
        var.set(["values", "value_partitions"], values_and_partitions(3), multi_value=1)
        var.parameter = None
        var.model = None
        var.round_to = 3
        return var


class ParameterGridSearch(ValuesOverParameterGridAnalysis):

    def name(self):
        return "ParameterGridSearch"

    def var(self):
        var = self.default_var()
        var.set(["values", "value_partitions"], values_and_partitions(2), multi_value=1)
        var.value_containers = None
        var.parameters = None
        var.parameter_container = None
        var.model = None
        var.round_to = 3
        return var


def values_over_parameter_grid_analysis(ana, var, path=None):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    # Gather all data from cache
    eval_dir, chkpt_dir = ana.exp.jobs[0].work.get(["evaluation_dir", "checkpoint_dir"])
    cache = Gather.get_cache(eval_dir, chkpt_dir)
    # Handle broadcasting for all variables
    #   handle values : str or list of str -> list of str w/ shape=(k,)
    if isinstance(var.values, str):
        if isinstance(var.value_partitions, list):
            var.values = [var.values for _ in var.value_partitions]
        else:
            var.values = [var.values]
    elif not isinstance(var.values, list):
        raise ValueError("Input values must be str or list of str, received %s" % (var.values))
    #   handle value_partitions : str or list of str -> list of str w/ shape=(k,)
    if isinstance(var.value_partitions, str): # single partition - broadcast to all values
        var.value_partitions = [var.value_partitions for _ in var.values]
    elif isinstance(var.value_partitions, list):
        if len(var.value_partitions) != len(var.values):
            var.value_partitions = [var.value_partitions for _ in var.values]
    elif not isinstance(var.value_partitions, list):
        raise ValueError("Input value_partitions must be str or list of str, received %s" % (var.value_partitions))
    for i, partition in enumerate(var.value_partitions):
        if partition == "*":
            var.value_partitions[i] = ["train", "valid", "test"]
        elif not isinstance(partition, list):
            var.value_partitions[i] = [var.value_partitions[i]]
    #   handle value_containers : str or list of str -> list of str w/ shape=(k,)
    if 0:
        if isinstance(var.value_containers, str):
            var.value_containers = [var.value_containers for _ in var.values]
        elif not isinstance(var.value_containers, list):
            raise ValueError("Input value_containers must be str or list of str, received %s" % (var.value_containers))
    #   handle plot_types : str or list of str -> list of str w/ shape=(k,)
    if isinstance(var.plot_types, str):
        var.plot_types = [var.plot_types for _ in var.values]
    elif not isinstance(var.plot_types, list):
        raise ValueError("Input plot_types must be str or list of str, received %s" % (var.plot_types))
    if var.debug:
        print(var)
    # Collect values from cache
    param_values = [job.work.get(var.parameters) for job in ana.exp.jobs]
    vop_dicts = collect_values_over_parameter_grid(
        var.values,
        var.value_partitions,
        var.parameters,
        param_values,
        var.parameter_partitions, 
        cache,
        var,
    )
    if var.debug > 1:
        print(Container().from_dict(vop_dicts))
    dfs = []
    for value_name, vop_dict in vop_dicts.items():
        df = pd.DataFrame(vop_dict).drop(var.parameter, axis=1)
        dfs.append(df)
    df = pd.concat(dfs, axis=1, keys=var.values)
    df.insert(0, var.parameter, param_values)
    print(df)
    # Plot the value(s) as a function of the parameter
    path = os.sep.join(
        [
            ana.result_dir(),
            "ValuesOverParameter_Values[%s]_Parameter[%s].png" % (",".join(vop_dicts.keys()), var.parameter),
        ]
    )
    plot_values_over_parameter(vop_dicts, var.value_partitions, var, path)
    # Save the value(s) as a Latex table
    path = path.replace(".png", ".tex")
    table = df.style.to_latex()
    with open(path, "w") as f:
        f.write(table)


def collect_values_over_parameter_grid(value_names, value_partitions, param_names, param_values, param_partitions, cache, var):
    import json
    print(value_names, value_partitions, param_names, param_values, param_partitions, sep="\n")
    model_id_map = {}
    vop_dicts = {}
    for value_name, _value_partitions in zip(value_names, value_partitions):
        vop_dicts[value_name] = util.to_dict(
            [",".join(param_names)] + _value_partitions, [[] for _ in range(1+len(_value_partitions))]
        )
    for value_name, _value_partitions in zip(value_names, value_partitions):
        vop_dicts[value_name][",".join(param_names)] = param_values
        for partition in _value_partitions:
            for _param_values in param_values:
                model, model_id = var.model, None
                try:
                    model_id, model_con = Gather.find_model_id(
                        cache, 
                        model, 
                        {"name": param_names, "value": _param_values, "partition": param_partitions}, 
                        return_channel_con=True
                    )
                except ValueError as err:
                    if var.debug > 1:
                        print(err)
                        print(model, model_id, partition)
                        input()
                # Get the cache channel that this value is in
                channel_name, channel_con = None, None
                for channel_name, channel_con, _ in cache.get_name_value_partitions(sort=False):
                    if channel_con.has(value_name, partition): break
                # Get the instance of this value
                value = np.nan
                try:
                    if channel_name in ["errors"]:
                        errors = cache.get(channel_name).get(value_name, partition, path=[model, model_id])
                        if var.debug > 1:
                            print("get(", value_name, partition, [model, model_id], ")")
                            print(errors)
                            input()
                        if var.response_feature is None:
                            response_feature, error_dict, _ = errors[0]
                        else:
                            error_dict = errors.get(var.response_feature)
                        value = np.mean(list(error_dict.values()))
                    else:
                        value = cache.get(channel_name).get(value_name, partition, path=[model, model_id])
                except ValueError as err:
                    if var.debug > 1:
                        print(err)
                        input()
                vop_dicts[value_name][partition].append(value)
    return vop_dicts


def values_over_parameter_analysis(ana, var, path=None):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    # Gather all data from cache
    eval_dir, chkpt_dir = ana.exp.jobs[0].work.get(["evaluation_dir", "checkpoint_dir"])
    cache = Gather.get_cache(eval_dir, chkpt_dir)
    # Handle broadcasting for all variables
    #   handle values : str or list of str -> list of str w/ shape=(k,)
    if isinstance(var.values, str):
        if isinstance(var.value_partitions, list):
            var.values = [var.values for _ in var.value_partitions]
        else:
            var.values = [var.values]
    elif not isinstance(var.values, list):
        raise ValueError("Input values must be str or list of str, received %s" % (var.values))
    #   handle value_partitions : str or list of str -> list of str w/ shape=(k,)
    if isinstance(var.value_partitions, str): # single partition - broadcast to all values
        var.value_partitions = [var.value_partitions for _ in var.values]
    elif isinstance(var.value_partitions, list):
        if len(var.value_partitions) != len(var.values):
            var.value_partitions = [var.value_partitions for _ in var.values]
    elif not isinstance(var.value_partitions, list):
        raise ValueError("Input value_partitions must be str or list of str, received %s" % (var.value_partitions))
    for i, partition in enumerate(var.value_partitions):
        if partition == "*":
            var.value_partitions[i] = ["train", "valid", "test"]
        elif not isinstance(partition, list):
            var.value_partitions[i] = [var.value_partitions[i]]
    #   handle plot_types : str or list of str -> list of str w/ shape=(k,)
    if isinstance(var.plot_types, str):
        var.plot_types = [var.plot_types for _ in var.values]
    elif not isinstance(var.plot_types, list):
        raise ValueError("Input plot_types must be str or list of str, received %s" % (var.plot_types))
    if var.model is None:
        var.model = input("Please provide a model: ")
    if var.parameter is None:
        var.parameter = input("Please provide a parameter: ")
    if var.debug:
        print(var)
    # Collect values from cache
    param_values = [job.work.get(var.parameter, var.parameter_partition) for job in ana.exp.jobs]
    vop_dicts = collect_values_over_parameter(
        var.values,
        var.value_partitions,
        var.parameter,
        param_values,
        var.parameter_partition, 
        cache,
        var,
    )
    if var.debug > 1:
        print(Container().from_dict(vop_dicts))
    # Join all data dicts into one DataFrame
    dfs = []
    for value_name, vop_dict in vop_dicts.items():
        df = pd.DataFrame(vop_dict).drop(var.parameter, axis=1)
        dfs.append(df)
    df = pd.concat(dfs, axis=1, keys=var.values)
    #   Add parameter column
    df.insert(0, var.parameter, param_values)
    #   Remove NaN instances (from None partition) from the columns
#    print([tuple(map(str.capitalize, col)) for col in df.columns])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_frame(
            df.columns.to_frame().fillna("")
        )
    else:
        df.columns = df.index.fillna("")
#    df.columns = pd.MultiIndex.from_frame(df.columns.to_frame().applymap(str.capitalize))
    #   Drop secondary columns if it simply repeats one element
    cols = np.array([list(col) for col in df.columns])
    if not len(np.unique(np.delete(cols[:,1], cols[:,1]==""))) > 1:
        df.columns = cols[:,0]
    #   Apply misc changes
    df = df.round(var.round_to)
    print(df)
    # Plot the value(s) as a function of the parameter
    path = os.sep.join(
        [
            ana.result_dir(),
            "ValuesOverParameter_Values[%s]_Parameter[%s].png" % (",".join(vop_dicts.keys()), var.parameter),
        ]
    )
    plot_values_over_parameter(vop_dicts, var.value_partitions, var, path)
    # Save the value(s) as a Latex table
    import re
    path = path.replace(".png", ".tex")
    table = df.to_latex(index=False, na_rep="N/A")
    table = table.replace(" model ", " Model ")
    table = table.replace(" n\\_parameters ", " Parameters ")
    table = table.replace(" runtime ", " Runtime ")
    table = table.replace(" epoch\\_runtime ", " Epoch Runtime ")
    table = re.sub("\.0+ ", " ", table)
    if var.debug:
        print(table)
    with open(path, "w") as f:
        f.write(table)


def collect_values_over_parameter(value_names, value_partitions, param_name, param_values, param_partition, cache, var):
    model_id_map = {}
    vop_dicts = {}
    for value_name, _value_partitions in zip(value_names, value_partitions):
        vop_dicts[value_name] = util.to_dict(
            [param_name] + _value_partitions, [[] for _ in range(1+len(_value_partitions))]
        )
    for value_name, _value_partitions in zip(value_names, value_partitions):
        vop_dicts[value_name][param_name] = param_values
        for partition in _value_partitions:
            for param_value in param_values:
                model, model_id = var.model, None
                try:
                    if param_name in ["model"]:
                        model = param_value
                        if model in model_id_map:
                            model_id = model_id_map[model]
                        else:
                            model_id, model_con = Gather.find_model_id(
                                cache, 
                                model, 
                                return_channel_con=True
                            )
                            model_id_map[model] = model_id
                        if var.debug > 1:
                            print(util.make_msg_block(param_value + " " + model_id))
    #                        print(model_params)
                            input()
                    else:
                        model_id, model_con = Gather.find_model_id(
                            cache, 
                            model, 
                            {"name": param_name, "value": param_value, "partition": param_partition}, 
                            return_channel_con=True
                        )
                except ValueError as err:
                    if var.debug > 1:
                        print(err)
                        print(model, model_id, partition)
                        input()
                # Get the cache channel that this value is in
                channel_name, channel_con = None, None
                for channel_name, channel_con, _ in cache.get_name_value_partitions(sort=False):
                    if channel_con.has(value_name, partition): break
                # Get the instance of this value
                value = np.nan
                try:
                    if channel_name in ["errors"]:
                        errors = cache.get(channel_name).get(value_name, partition, path=[model, model_id])
                        if var.debug > 1:
                            print("get(", value_name, partition, [model, model_id], ")")
                            print(errors)
                            input()
                        if var.response_feature is None:
                            response_feature, error_dict, _ = errors[0]
                        else:
                            error_dict = errors.get(var.response_feature)
                        value = np.mean(list(error_dict.values()))
                    else:
                        value = cache.get(channel_name).get(value_name, partition, path=[model, model_id])
                except ValueError as err:
                    if var.debug > 1:
                        print(err)
                        input()
                vop_dicts[value_name][partition].append(value)
    return vop_dicts


def plot_values_over_parameter(vop_dicts, partitions, var, path):
    plt = Plotting()
    normalize = var.normalize if "normalize" in var else True
    n_bars = 0
    for _partitions, plot_type in zip(partitions, var.plot_types):
        n_bars += len(_partitions) if plot_type == "bar" else 0
    i, bar_idx = 0, 0
    colors = plt.default_colors
    for (value_name, vop_dict), _partitions, plot_type in zip(vop_dicts.items(), partitions, var.plot_types):
        df = pd.DataFrame(vop_dict)
        if var.debug:
            print(util.make_msg_block(value_name))
            print(df)
        df.fillna(np.nan, inplace=True)
        for partition in _partitions:
            linestyle = "-"
            if partition in ["test"]:
                linestyle = "--"
            color = colors[i % len(colors)]
            label = value_name
            if not partition is None:
                label = "%s %s" % (partition, value_name)
            values = df[partition].to_numpy()
            if normalize and not all(np.logical_and(values >= 0, values <= 1)):
                values = util.minmax_transform(values, np.min(values), np.max(values), a=1/10, b=1)
            param_values = df[var.parameter]
            if not isinstance(param_values[0], float) and not isinstance(param_values[0], int):
                param_values = [str(param_value) for param_value in param_values]
            kwargs = {} if not "plot_kwargs" in var else var.plot_kwargs
            if plot_type == "line":
                kwargs = util.merge_dicts(kwargs, {"color": color, "label": label, "linestyle": linestyle})
                plt.plot_line(param_values, values, **kwargs)
            elif plot_type == "bar":
                if n_bars > 0:
                    bar_sep = 25 / len(param_values)
                    bar_extent = [-3/8 * bar_sep, 3/8 * bar_sep]
                    bar_width = plt.defaults.bars.width
                    if n_bars == 1:
                        bar_offsets = np.array([0])
                    else:
                        bar_width = plt.defaults.bars.width / n_bars
                        bar_width = (bar_extent[1] - bar_extent[0]) / (n_bars * 3/4)
                        bar_offsets = np.linspace(
                            bar_extent[0] + bar_width / 2,
                            bar_extent[1] - bar_width / 2,
                            n_bars,
                        )
                locs = param_values
                if isinstance(param_values[0], str):
                    locs = np.arange(len(param_values), dtype=float) * bar_sep
                locs += bar_offsets[bar_idx]
                kwargs = util.merge_dicts(
                    kwargs, 
                    {"width": bar_width, "color": color, "label": label, "linestyle": linestyle})
                plt.plot_bar(locs, values, **kwargs)
                bar_idx += 1
            elif plot_type == "scatter":
                kwargs = util.merge_dicts(kwargs, {"color": color, "label": label, "linestyle": linestyle})
                plt.plot_satter(param_values, values, **kwargs)
            else:
                raise NotImplementedError("Unknown plot_type \"%s\"" % (var.plot_type))
            if np.nan in values and value_name in ["MAE", "MSE", "MAPE", "RMSE", "NRMSE", "OPR", "UPR", "RRSE"]:
                _values = np.copy(values)
                _values[_values == 0] = int(sys.float_info.max)
                plt.plot_axis(min(values), color=color, linestyle=":")
            i += 1
    plt.style("grid")
    locs = param_values
    if isinstance(param_values[0], str):
        locs = np.arange(len(param_values), dtype=float) * bar_sep
        plt.lim([locs[0]-1, locs[-1]+1])
    fontsize = 11 * (25 / len(param_values))
    plt.xticks(locs, param_values, rotation=90, fontsize=fontsize)
    if normalize:
        plt.lim(None, [0, 1.025], ymargin=["1%", "2.5%"])
        plt.labels(var.parameter, "Normalized Value")
    else:
        plt.labels(var.parameter, None)
    plt.legend(prop={"size": 6.5}, ncol=len(partitions))
    plt.save_figure(path)
    plt.close()


def get_graph_data(dataset, partition, var):
    if dataset.graph.is_empty():
        raise ValueError("Cannot plot a graph without dataset.graph")
    graph = dataset.graph
    G = graph.original.get("nx_graph", partition)
    node_positions, node_sizes = None, None
    node_labels = graph.original.get("node_labels", partition)
    node_selection = ["literal"] + node_labels.tolist()
    n_nodes = len(node_labels)
    if not dataset.spatial.is_empty(): # Get node positions and sizes from spatial data
        spatial = dataset.spatial
        spatial_labels = spatial.original.spatial_labels
        spatial_indices = spatial.indices_from_selection(spatial_labels, node_selection)
        position_indices = np.array(util.get_dict_values(spatial.misc.feature_index_map, var.node_position_features))
        node_positions = spatial.original.filter_axis(
            spatial.original.features, 
            [-2, -1], 
            [spatial_indices, position_indices]
        )
        if var.node_size_feature in spatial.misc.feature_index_map:
            size_idx = spatial.misc.feature_index_map[var.node_size_feature]
            node_sizes = spatial.original.filter_axis(
                spatial.original.features, 
                [-2, -1], 
                [spatial_indices, size_idx]
            )
            node_sizes = node_sizes / np.max(node_sizes)
            node_sizes = node_sizes[spatial_indices]
    # Get node sizes from spatiotemporal data
    if not var.node_size_feature is None and not dataset.spatiotemporal.is_empty():
        spatiotemporal = dataset.spatiotemporal
        spatial_labels = spatiotemporal.original.spatial_labels
        spatial_indices = spatial.indices_from_selection(spatial_labels, node_selection)
        size_idx = spatiotemporal.misc.feature_index_map[var.node_size_feature]
        node_sizes = np.mean(
            spatiotemporal.filter_axis(
                spatiotemporal.statistics.means, 
                [-2, -1], 
                [spatial_indices, size_idx]
            ), 
            axis=0
        )
    else:
        node_sizes = np.ones((len(node_labels),))
    node_sizes = node_sizes**(1/var.node_size_root)
    node_sizes = node_sizes * var.node_size_mult
    return G, node_positions, node_sizes


def visualize_graph(dataset, partition, var, path):
    from Plotting import Plotting
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=var.figsize)
    plot = Plotting()
    if "shapes" in var and not var.shapes is None:
        plot.plot_shapes(var.shapes, ax=ax, filled=True, color="k", alpha=1/8, linewidth=1)
    if partition is None or isinstance(partition, str):
        partition = [partition]
    elif not isinstance(partition, list):
        raise ValueError("Input \"partition\" may be None, str, or list of None/str. Received %s" % (type(partition)))
    for _partition in partition:
        alpha = 2/8 if _partition is None and len(partition) > 1 else 1
        G, node_positions, node_sizes = get_graph_data(dataset, _partition, var)
        plot.plot_networkx_graph(
            G, 
            node_positions, 
            node_sizes, 
            node_alpha=alpha, 
            edge_alpha=alpha, 
            plot_edges=var.plot_args.edges, 
            plot_labels=var.plot_args.labels, 
            ax=ax
        )
    xlim, ylim = plot.lim()
    plot.save_figure(path, dpi=200)
    plot.close()
    plt.close()
