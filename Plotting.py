import time
import os
import sys
import re
import seaborn as sns
import pandas as pd
import geopandas as gpd
import numpy as np
import shapefile
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import make_axes_locatable
import polylabel
import networkx as nx

import Utility as util
from Container import Container


np.set_printoptions(precision=4, suppress=True, linewidth=200)
class PolygonN(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        aspect = height/float(width)
        verts = orig_handle.get_xy()
        closed = orig_handle.get_closed()
        minx, miny = verts[:,0].min(), verts[:,1].min()
        maxx, maxy = verts[:,0].max(), verts[:,1].max()
        aspect= (maxy-miny)/float((maxx-minx))
        nvx = (verts[:,0]-minx)*float(height)/aspect/(maxx-minx)-x0
        nvy = (verts[:,1]-miny)*float(height)/(maxy-miny)-y0

        p = Polygon(np.c_[nvx, nvy], closed)
        p.update_from(orig_handle)
        p.set_transform(handlebox.get_transform())

        handlebox.add_artist(p)
        return p


class Plotting(Container):

    debug = 0
    default_colors = np.array(plt.rcParams["axes.prop_cycle"].by_key()["color"])
#    defaults = {
#        "bar.width": 0.8, 
#        "bar.bottom": 0.0, 
#        "bar.align": "center", 
#        "color.cycle": plt.rcParams["axes.prop_cycle"].by_key()["color"], 
#    }
    line_width = 1
    gt_line_width = 1.25 * line_width
    pred_line_width = 0.75 * line_width
    marker_size = 7
    month_labels = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "January"]
    feature_idx_map = {"FLOW_OUTcms": 3, "SWmm": 10, "PRECIPmm": 7}
    feature_fullname_map = {
        "FLOW_OUTcms": "Streamflow",
        "dv_va": "Streamflow",
        "SWmm": "Soil Moisture",
        "PRECIPmm": "Precipitation",
        "tmin": "Minimum Temperature",
        "tmax": "Maximum Temperature",
        "speedmph": "Vehicle Speed",
        "occupancy": "Road Occupancy",
        "power_MW": "Power",
        "power_kWh": "Power",
        "exchange_rate": "Exchange Rate",
        "signal_mV": "Signal",
        "confirmed": "Confirmed Cases",
        "Avg_Speed": "Average Speed",
        "return_air_humidity": "Return Air Humidity", 
    }
    feature_SIunit_map = {
        "FLOW_OUTcms": "$m^{3}/s$",
        "dv_va": "$ft^{3}/s$", 
        "SWmm": "$mm$",
        "PRECIPmm": "$mm$",
        "tmin": "$\degree C$",
        "tmax": "$\degree C$",
        "speedmph": "$mph$",
        "occupancy": "$\%$",
        "power_MW": "$MW$",
        "power_kWh": "$kWh$",
        "signal_mV": "$mV$",
        "Avg_Speed": "$mph$",
        "return_air_humidity": "?", 
    }
    feature_ylabel_map = {}
    for feature in feature_fullname_map.keys():
        feature_ylabel_map[feature] = feature_fullname_map[feature]
        if feature in feature_SIunit_map:
            feature_ylabel_map[feature] += " (%s)" % (feature_SIunit_map[feature])
    dataset_legend_map = {
        "littleriver_observed": "Little",
        "wabashriver_swat": "Wabash(SWAT)",
        "wabashriver_observed": "Wabash(GT)",
        "los-loop_observed": "Los-Loop",
        "sz-taxi_observed": "SX-Taxi",
        "metr-la": "METR-LA",
        "_metr-la": "_METR-LA",
        "new-metr-la": "NEW_METR-LA",
        "pems-bay": "PEMS-BAY",
        "_pems-bay": "_PEMS-BAY",
        "new-pems-bay": "NEW_PEMS-BAY",
        "traffic": "Traffic",
        "solar-energy": "Solar-Energy",
        "electricity": "Electricity",
        "exchange-rate": "Exchange Rate",
        "ecg5000_observed": "ECG5000",
        "covid-19_observed": "COVID-19",
        "caltranspems_d05": "Caltrans PeMS District 5",
        "us-streams": "US National Streams", 
        "us-streams-al": "Alaska", 
        "power": "HVAC", 
    }
    partition_fullname_map = {"train": "Training", "valid": "Validation", "test": "Testing"}
    partition_codename_map = {"train": "Train", "valid": "Valid", "test": "Test"}
    feature_plot_order_map = {
        "FLOW_OUTcms": "descending",
        "dv_va": "descending",
        "SWmm": "descending",
        "speedmph": "descending",
        "occupancy": "descending",
        "power_MW": "descending",
        "power_kWh": "descending",
        "exchange_rate": "descending",
        "signal_mV": "descending",
        "confirmed": "descending",
        "Avg_Speed": "descending",
        "return_air_humidity": "descending", 
    }

    def __init__(self):
        self.set("plot_dir", "Plots")
        self.set("lines", [])
        plt.rcdefaults()
        self.defaults = Container()
        for key, value in plt.rcParams.items():
            fields = key.split(".")
            if len(fields) > 1:
                self.defaults.set(fields[-1], value, path=fields[:-1])
            else:
                self.defaults.set(fields[-1], value)
        self.defaults.bars = Container().set(
            ["width", "bottom", "align"],
            [0.8, 0, "center"],
            multi_value=True
        )


    def plot_networkx_graph(
        self,
        G,
        node_positions=None,
        node_sizes=None,
        node_list="all",
        edge_list="all",
        node_alpha=1.0,
        edge_alpha=1.0,
        node_kwargs={},
        edge_kwargs={},
        label_kwargs={},
        plot_edges=True, 
        plot_labels=True, 
        ax=None,
    ):
        # Get node and edge sets to be plotted
        if isinstance(node_list, str) and node_list == "all":
            node_list = list(G.nodes())
        if isinstance(edge_list, str) and edge_list == "all":
            edge_list = list(G.edges())
        elif isinstance(edge_list, np.ndarray):
            if edge_list.shape[0] == 2:
                edge_list = [(src, dst) for src, dst in np.transpose(edge_list)]
        n_nodes = len(node_list)
        n_edges = len(edge_list)
        # Generate/prepare node and edge positions/sizes
        if node_positions is None:
            node_positions = np.random.uniform(size=(n_nodes, 2))
        elif not isinstance(node_positions, dict):
            node_positions = util.to_dict(node_list, node_positions)
        if node_sizes is None:
            node_sizes = 300
        elif isinstance(node_sizes, dict):
            node_sizes = util.get_dict_values(node_sizes, node_list)
        if 0:
            for node, pos in node_positions.items():
                print(node, pos)
            for node, size in zip(G.nodes(), node_sizes):
                print(node, size)
        # Plot it
        node_kwargs = util.merge_dicts({"alpha": 3/4*node_alpha}, node_kwargs)
        edge_kwargs = util.merge_dicts({"alpha": edge_alpha, "node_size": 1/4, "arrows": True}, edge_kwargs)
        label_kwargs = util.merge_dicts({"alpha": node_alpha, "font_size": 5, "font_color": "r"}, label_kwargs)
        if ax is None:
            nx.draw_networkx_nodes(G, node_positions, nodelist=node_list, node_size=node_sizes, **node_kwargs)
            if plot_edges:
                nx.draw_networkx_edges(G, node_positions, edgelist=edge_list, **edge_kwargs)
            if plot_labels:
                nx.draw_networkx_labels(G, node_positions, **label_kwargs)
        else:
            nx.draw_networkx_nodes(G, node_positions, nodelist=node_list, node_size=node_sizes, ax=ax, **node_kwargs)
            if plot_edges:
                nx.draw_networkx_edges(G, node_positions, edgelist=edge_list, ax=ax, **edge_kwargs)
            if plot_labels:
                nx.draw_networkx_labels(G, node_positions, ax=ax, **label_kwargs)
#        plt.axis("off")

    def plot_shapes(self, shapes, ax=None, filled=False, **kwargs):
        from shapely.geometry import Point, Polygon, MultiPolygon
        if isinstance(shapes, str):
            shapes = gpd.read_file(shapes)
        elif not isinstance(shapes, gpd.GeoDataFrame):
            raise ValueError("Input \"shapes\" may be str or geopandas.GeoDataFrame. Received %s" % (type(shapes)))
        shapes = shapes.to_crs("epsg:4326")
        polys = []
        for i in range(len(shapes)):
            poly = shapes.loc[i,:].geometry
            if isinstance(poly, Polygon):
                polys.append(np.array(poly.exterior.coords.xy))
            elif isinstance(poly, MultiPolygon):
                polys += [np.array(_poly.exterior.coords.xy) for _poly in poly.geoms]
            else:
                raise NotImplementedError("Unknown geometry type %s" % (type(poly)))
        if ax is None:
            ax = plt.gca()
        for i, poly in enumerate(polys):
            if filled:
                ax.add_patch(plt.Polygon(poly.T, **kwargs))
            else:
                self.plot_line(poly[0,:], poly[1,:], ax=ax, **kwargs)

    def plot_watershed(self, item_shapes_map, subbasin_river_map, path, highlight=True, watershed="", river_opts={"color_code": True, "name": True}):
        cache_path = "Data" + os.sep + "SubbasinLabelCoordinateMap_Watershed[%s].pkl" % (watershed)
        if os.path.exists(cache_path):
            subbasin_coordinate_map = util.from_cache(cache_path)
        else:
            subbasin_coordinate_map = {}
            for shape_rec in item_shapes_map["subbasins"].shapeRecords():
                x = [i[0] for i in shape_rec.shape.points[:]]
                y = [i[1] for i in shape_rec.shape.points[:]]
                if watershed == "WabashRiver":
                    subbasin = shape_rec.record[0]
                    lon, lat = shape_rec.record[10], shape_rec.record[11]
                elif watershed == "LittleRiver":
                    subbasin = shape_rec.record[8]
                else:
                    raise NotImplementedError()
                print("Computing Pole of Inaccessibility for ", subbasin)
                coords = [[x_i, y_i] for x_i, y_i in zip(x, y)]
                subbasin_coords = polylabel.polylabel([coords], 0.1)
                subbasin_coordinate_map[subbasin] = subbasin_coords
            util.to_cache(subbasin_coordinate_map, cache_path)
        n_subbasins = len(subbasin_coordinate_map.keys())
        # Normalize coordinates to [0, 1]
        min_x, max_x, min_y, max_y = sys.float_info.max, -sys.float_info.max, sys.float_info.max, -sys.float_info.max
        min_lon, max_lon, min_lat, max_lat = sys.float_info.max, -sys.float_info.max, sys.float_info.max, -sys.float_info.max
        for item, shapes in item_shapes_map.items():
            for shape in shapes.shapeRecords():
                x = [i[0] for i in shape.shape.points[:]]
                y = [i[1] for i in shape.shape.points[:]]
                min_x, max_x, min_y, max_y = min(min(x), min_x), max(max(x), max_x), min(min(y), min_y), max(max(y), max_y)
        # Get river color cycle
        if len(subbasin_river_map) > 0:
            if 1:
                colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
                indices = np.delete(np.arange(len(colors)), [3, 7])
            else:
                colors = np.array(plt.get_cmap("tab20b").colors)
                indices = np.delete(np.arange(20), np.arange(0, 20, 4))
            colors = colors[indices]
            river_color_map = {}
            i = 0
            n_rivers, n_colors = len(set(subbasin_river_map.values())), len(colors)
            indices = np.arange(n_rivers) % n_colors
            np.random.seed(1)
            indices = np.random.permutation(indices)
            for river in subbasin_river_map.values():
                if river not in river_color_map:
                    river_color_map[river] = colors[indices[i]]
                    i += 1
            tmp = river_color_map["Flatrock"]
            river_color_map["Flatrock"] = river_color_map["Muscatatuck"]
            river_color_map["Muscatatuck"] = tmp
            # River labeling setup
            river_subbasins_map = {}
            for subbasin, river in subbasin_river_map.items():
                if river in river_subbasins_map:
                    river_subbasins_map[river] += [subbasin]
                else:
                    river_subbasins_map[river] = [subbasin]
        # Setup
        train_color, test_color = "tab:grey", "tab:red"
        np.random.seed(1)
        n, r = 1276, 20
        subbasins = np.arange(n) + 1
        train_indices = sorted(np.random.choice(n, size=r, replace=False))
        train_subbasins = subbasins[train_indices]
        test_subbasins = sorted(np.delete(subbasins, train_indices)[np.random.choice(n-r, size=r, replace=False)])
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        subbasin_patches, river_patches = [], []
        train_patches, test_patches, other_patches = [], [], []
        subbasin_for_subbasin_legend = 1014
        subbasin_for_river_legend = 644
        highlight_train, highlight_test = highlight, highlight
        highlight_train, highlight_test = highlight, highlight
        for item, shapes in item_shapes_map.items():
            i = 0
            for shape_rec in shapes.shapeRecords():
                i += 1
                x = np.array([i[0] for i in shape_rec.shape.points[:]])
                y = np.array([i[1] for i in shape_rec.shape.points[:]])
                x = util.minmax_transform(x, min_x, max_x)
                y = util.minmax_transform(y, min_y, max_y)
                color = "k"
                if "subbasins" in item:
                    print(dir(shape_rec))
                    print(shape_rec.record)
                    print(shape_rec.shape)
                    if watershed == "WabashRiver":
                        subbasin = shape_rec.record[0]
                        lon, lat = shape_rec.record[10], shape_rec.record[11]
                    elif watershed == "LittleRiver":
                        subbasin = shape_rec.record[8]
                    else:
                        raise NotImplementedError()
                    subbasin_loc = subbasin_coordinate_map[subbasin]
                    subbasin_x = util.minmax_transform(subbasin_loc[0], min_x, max_x)
                    subbasin_y = util.minmax_transform(subbasin_loc[1], min_y, max_y)
                    xy = np.stack([x, y], axis=1)
                    if subbasin in train_subbasins:
                        color = "w"
                        train_patches += [Polygon(xy)]
                    elif subbasin in test_subbasins:
                        color = "w"
                        test_patches += [Polygon(xy)]
                    else:
                        other_patches += [Polygon(xy)]
                    if subbasin == subbasin_for_subbasin_legend:
                        subbasin_legend_xy = xy
                    color = "k"
                    font_size = {"WabashRiver": 2.5, "LittleRiver": 20}[watershed]
                    plt.text(subbasin_x, subbasin_y, "%s" % (str(subbasin)), color=color, ha="center", va="center", fontsize=font_size)
                    line_width = {"WabashRiver": 0.5, "LittleRiver": 2.5}[watershed]
                    plt.plot(x, y, color="k", linewidth=line_width)
                if "rivers" in item:
                    print(dir(shape_rec))
                    print(shape_rec.record)
                    print(shape_rec.shape)
                    if watershed == "WabashRiver":
                        subbasin = shape_rec.record[5]
                    elif watershed == "LittleRiver":
                        subbasin = shape_rec.record[0]
                    else:
                        raise NotImplementedError()
                    if 1:
                        max_d = {"WabashRiver": 0.01, "LittleRiver": sys.float_info.max}[watershed]
                        mask = [False]
                        for i in range(len(x)-1):
                            d = ((x[i] - x[i+1])**2 + (y[i] - y[i+1])**2)**(1/2)
                            mask += [d > max_d]
                        x, y = np.ma.masked_array(x, mask), np.ma.masked_array(y, mask)
                    if False:
                        if x[0] == x[-1] or y[0] == y[-1]:
                            x, y = x[:-1], y[:-1]
                    color = "tab:blue"
                    if len(subbasin_river_map) > 0 and river_opts["color_code"]:
                        river = subbasin_river_map[subbasin]
                        color = river_color_map[river]
                    line_width = {"WabashRiver": 0.625, "LittleRiver": 1.5}[watershed]
                    plt.plot(x, y, color=color, linewidth=line_width)
                    if subbasin == subbasin_for_river_legend:
                        xy = np.ma.stack([x, y], axis=1)
                        river_legend_xy = xy
            if "subbasins" in item:
                subbasin_patches = []
                if highlight_train:
                    ax.add_collection(PatchCollection(train_patches, facecolor=train_color))
                    train_patch = Polygon(subbasin_legend_xy, color=train_color, label="Training Subbasin")
                    subbasin_patches += [train_patch]
                if highlight_test:
                    ax.add_collection(PatchCollection(test_patches, facecolor=test_color))
                    test_patch = Polygon(subbasin_legend_xy, color=test_color, label="Testing Subbasin")
                    subbasin_patches += [test_patch]
            if "rivers" in item and len(subbasin_river_map) > 0 and river_opts["name"]:
                for river, subbasins in river_subbasins_map.items():
                    x = np.array([subbasin_coordinate_map[subbasin][0] for subbasin in subbasins])
                    y = np.array([subbasin_coordinate_map[subbasin][1] for subbasin in subbasins])
                    x = util.minmax_transform(x, min_x, max_x)
                    y = util.minmax_transform(y, min_y, max_y)
                    river_x, river_y = np.mean(x), np.mean(y)
                    color = river_color_map[river]
                    size = 4 * (np.log(len(subbasins)) / np.log(5))
                    if river == "Eel":
                        txt = plt.text(river_x+0.25, river_y+0.45, "%s" % (str(river)), color=color, ha="center", va="center", fontsize=size)
                        txt.set_path_effects([PathEffects.withStroke(linewidth=0.1*size, foreground="k")])
                        txt = plt.text(river_x-0.15, river_y-0.3, "%s" % (str(river)), color=color, ha="center", va="center", fontsize=size)
                        txt.set_path_effects([PathEffects.withStroke(linewidth=0.1*size, foreground="k")])
                    else:
                        txt = plt.text(river_x, river_y, "%s" % (str(river)), color=color, ha="center", va="center", fontsize=size)
                        txt.set_path_effects([PathEffects.withStroke(linewidth=0.1*size, foreground="k")])
                river_patch_map = {}
                for river in subbasin_river_map.values():
                    color = river_color_map[river]
                    if river not in river_patch_map:
                        river_patch_map[river] = Polygon(river_legend_xy, False, fill=False, color=color, label=river)
                river_patches = list(river_patch_map.values())
        handles = subbasin_patches# + river_patches[:]
        if len(handles) > 0:
            size = 15
            if 1:
                plt.legend(
                    handles=handles,
                    handler_map={Polygon: PolygonN()},
                    prop={"size": size}
                )
            else:
                n_col = len(handles) // 3
                n_col = 4
                plt.legend(
                    handles=handles,
                    handler_map={Polygon: PolygonN()},
                    loc="upper center",
                    bbox_to_anchor=(0.5,1.1),
                    ncol=n_col,
                    fancybox=True,
                    shadow=True,
                    prop={"size": size}
                )
        dpi = 500
        dpi = 1000
        plt.axis("off")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close()

    def plot_cluster_heatmap(self, mat, xtick_labels=[], ytick_labels=[], xlabel="", ylabel="", plot_numbers=True, transpose=False, size=(12, 12), rowcol_cluster=[True, True], ax=None):
        data = {}
        if xtick_labels == []:
            xtick_labels = list(range(mat.shape[1]))
        if ytick_labels == []:
            ytick_labels = list(range(mat.shape[0]))
        if transpose:
            for i in range(mat.shape[0]):
                data[ytick_labels[i]] = mat[i,:]
        else:
            for i in range(mat.shape[1]):
                data[xtick_labels[i]] = mat[:,i]
        data = pd.DataFrame(data)
        if transpose:
            data.index = xtick_labels
        else:
            data.index = ytick_labels
        cg = sns.clustermap(data, method="average", metric="correlation", figsize=size, annot=plot_numbers, annot_kws={"fontsize": 0.7*size[0]}, row_cluster=rowcol_cluster[0], col_cluster=rowcol_cluster[1])
        plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        if transpose:
            cg.ax_heatmap.set_xlabel(ylabel, fontsize=1.5*size[0])
            cg.ax_heatmap.set_ylabel(xlabel, fontsize=1.5*size[0])
        else:
            cg.ax_heatmap.set_xlabel(xlabel, fontsize=1.5*size[0])
            cg.ax_heatmap.set_ylabel(ylabel, fontsize=1.5*size[0])
        return ax

    def plot_heatmap(self, mat, xtick_locs=None, xtick_labels=[], ytick_locs=None, ytick_labels=[], xlabel="", ylabel="", cbar_label="", cmap="viridis", plot_cbar=True, plot_numbers=False, transpose=False, ax=None):
        if ax is None:
            ax = plt.gca()
        if transpose:
            mat = np.transpose(mat)
            xtick_labels, ytick_labels = ytick_labels, xtick_labels
            xlabel, ylabel = ylabel, xlabel
        size = (mat.shape[1], mat.shape[0])
        if plot_numbers:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    plt.text(
                        j,
                        mat.shape[0]-1-i,
                        "%.4f" % (mat[i,j]),
                        ha="center",
                        va="center",
                        fontsize=0.5*size[0]
                    )
        im = ax.imshow(mat, cmap=cmap, extent=[-0.5, mat.shape[1]-0.5, -0.5, mat.shape[0]-0.5])
        if plot_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=16)
        if ytick_locs is None:
            ytick_locs = np.flip(np.arange(len(ytick_labels)))
        if 1:
            self.xticks(xtick_locs, xtick_labels, ax=ax, rotation=90)
            self.yticks(ytick_locs, ytick_labels, ax=ax)
        else:
            self.xticks(xtick_locs, xtick_labels, ax=ax, fontsize=0.75*size[0], rotation=90)
            self.yticks(ytick_locs, ytick_labels, ax=ax, fontsize=0.75*size[1])
        self.xlabel(xlabel, ax=ax, fontsize=16)
        self.ylabel(ylabel, ax=ax, fontsize=16)
        return ax

    def plot_graph_distributions(self, G, distributions, graph_name, plot_dir="Plots"):
        # Degrees
        path = plot_dir + os.sep + "GraphDistribution_Graph[%s]_Type[%s].png" % (
            graph_name,
            "degree"
        )
        if "degree" in distributions and not os.path.exists(path):
            degree_dist = netprop.Degree_Distribution(G)
            self.plot_distribution(
                degree_dist,
                path,
                xlabel="Degree ($k$)",
                ylabel="Nodes with Degree $k$ ($N_k$)",
                title="Degree Distribution"
            )
        # Connected Components
        path = plot_dir + os.sep + "GraphDistribution_Graph[%s]_Type[%s].png" % (
            graph_name,
            "connected_component"
        )
        if "connected_component" in distributions and not os.path.exists(path):
            CC_dist = netprop.CC_Distribution(G)
            self.plot_distribution(
                CC_dist,
                path,
                xlabel="Weekly Connected Component Size",
                ylabel="Count",
                title="Connected Component Size Distribution",
                xlog=False,
                ylog=False,
                intAxis=False
            )
        # Local Clustering Coefficients
        path = plot_dir + os.sep + "GraphDistribution_Graph[%s]_Type[%s].png" % (
            graph_name,
            "local_clustering_coefficient"
        )
        if "local_clustering_coefficient" in distributions and not os.path.exists(path):
            LCC_dist = netprop.Clustering_Analysis(G)
            self.plot_distribution(
                LCC_dist,
                path,
                xlabel="Clustering Coefficient",
                ylabel="Number of Nodes",
                title="Clustering Coefficient Distribution",
                xlog=False,
                ylog=False,
                showLine=False
            )
        # Shortest Paths
        path = plot_dir + os.sep + "GraphDistribution_Graph[%s]_Type[%s].png" % (
            graph_name,
            "shortest_path"
        )
        if "shortest_path" in distributions and not os.path.exists(path):
            SPL_dist = netprop.ShortestPaths_Analysis(G)
            self.plot_distribution(
                SPL_dist,
                path,
                xlabel="Shortest Path Lengths (hops)",
                ylabel="Number of Paths",
                title="Shortest Path Length Distribution",
                xlog=False,
                ylog=False,
                showLine=False,
                intAxis=True
            )


    def plot_distribution(self, data, path="", xlabel="", ylabel="", title="", xlog = True, ylog= True, showLine=False, intAxis=False) :
        counts = {}
        for item in data :
            if item not in counts :
                counts[item] = 0
            counts[item] += 1
        counts = sorted(counts.items())
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter([k for (k, v) in counts] , [v for (k, v) in counts])
        if(len(counts)<20):  # for tiny graph
            showLine=True
        if showLine:
            ax.plot([k for (k, v) in counts] , [v for (k, v) in counts])
        if xlog:
            ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
        if intAxis:
            gca = fig.gca()
            gca.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2e"))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=60)
        plt.title(title)
        if isinstance(path, str) and path != "":
            fig.savefig(path, bbox_inches="tight")
        else:
            fig.show()
        plt.close()

    # Purpose:
    #   Plot the probability density function for given data
    # Notes:
    #   If receiving "underflow encountered in exp" error, increase/decrease bandwidth/n_bins value
    #   If PDF looks noisy, increase/decrease bandwidth/n_bins value
    #   If PDF doesn't sufficiently span the known value range, decrease/increase bandwidth/n_bins value
    def plot_density(self, data, source="", bandwidth=None, n_bins=100, plt_mean=False, plt_median=False, plt_stddev=False, fill=False, line_kwargs={}):
        from sklearn.neighbors import KernelDensity
        from scipy.stats import gaussian_kde
        from statsmodels.nonparametric.kde import KDEUnivariate
        if self.debug:
            print(data.shape, np.min(data), np.max(data))
        X = np.reshape(data, -1)
        x_range = np.max(X) - np.min(X)
        if bandwidth is None:
            bandwidth = x_range / n_bins
        xs = np.linspace(np.min(X)-x_range/10, np.max(X)+x_range/10, n_bins+1, endpoint=True)
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(np.reshape(X, [-1, 1]))
        pdf = np.exp(kde.score_samples(np.reshape(xs, [-1, 1])))
        # Interpolate line
        from scipy.interpolate import make_interp_spline, BSpline
        new_xs = np.linspace(np.min(xs), np.max(xs), 4*n_bins+1, endpoint=True)
        new_ys = make_interp_spline(xs, pdf, k=3)(new_xs)
#        pdf = pdf / np.sum(pdf)
        label = "$%s$" % (source.replace(" ", " \\ "))
        lines = plt.plot(
            new_xs,
            new_ys,
            label=label,
            **line_kwargs,
        )
        self.set("lines", self.get("lines") + lines)
        if fill:
            plt.fill_between(xs, 0, pdf, facecolor=color, alpha=0.2)
        mean, median, std = np.mean(X), np.median(X), np.std(X)
        if plt_mean:
            height = np.interp(mean, xs, pdf)
            label = "$\mu$"
            if source in self.partition_codename_map:
                label = "$\mu_{%s}$" % (self.partition_codename_map[source].replace(" ", " \\ "))
            elif source != "":
                label = "$\mu_{%s}$" % (source.replace(" ", " \\ "))
            plt.vlines(mean, 0, height, label=label, color=color, ls="-.", linewidth=self.line_width)
        if plt_median:
            height = np.interp(median, xs, pdf)
            label = "$Median$"
            if source in self.partition_codename_map:
                label = "$Median_{%s}$" % (self.partition_codename_map[source].replace(" ", " \\ "))
            elif source != "":
                label = "$Median_{%s}$" % (source.replace(" ", " \\ "))
            plt.vlines(median, 0, height, label=label, color=color, ls=":", linewidth=self.line_width)
        if plt_stddev:
            height = np.interp(mean-std, xs, pdf)
            plt.vlines(mean-std, 0, height, color=color, ls=":", linewidth=self.line_width)
            height = np.interp(mean+std, xs, pdf)
            label = "$\mu \pm \sigma$"
            if source in self.partition_codename_map:
                label = "$\mu_{%s} \pm \sigma_{%s}$" % (
                    self.partition_codename_map[source].replace(" ", " \\ "),
                    self.partition_codename_map[source].replace(" ", " \\ ")
                )
            elif source != "":
                label = "$\mu_{%s} \pm \sigma_{%s}$" % (
                    source.replace(" ", " \\ "),
                    source.replace(" ", " \\ ")
                )
                label = "$\sigma_{%s}$" % (source.replace(" ", " \\ "))
            plt.vlines(mean+std, 0, height, label=label, color=color, ls=":", linewidth=self.line_width)
        y_max = -sys.float_info.max
        for line in self.get("lines"):
            y_max = max(y_max, np.max(line.get_ydata()))
        self.lims(ylim=[-0.002*y_max, y_max*1.01])
        self.legend()
        return self.lines

    def plot_time_series(self, x, y, temporal_labels=None, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        if x is None:
            x = range(len(y))
        line = self.plot_line(x, y, ax=ax, **kwargs)
        if not temporal_labels is None:
            indices = np.linspace(0, len(temporal_labels)-1, 7, dtype=int)
            self.xticks(indices, temporal_labels[indices], ax=ax, rotation=60)
        return line

    def plot_line(self, x, y, ax=None, **kwargs):
        if x is None:
            x = range(len(y))
        if ax is None:
            ax = plt.gca()
        kwargs = util.merge_dicts(
            {},
            kwargs,
        )
        return ax.plot(x, y, **kwargs)

    def plot_bar(self, x, y, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        kwargs = util.merge_dicts(
            {"width": 0.8, "bottom": 0.0, "align": "center"},
            kwargs,
        )
        if x is None:
            x = np.arange(len(y))
        elif isinstance(x[0], str):
            x = np.arange(len(x))
        ax.bar(x, y, **kwargs)

    def plot_scatter(self, x, y, labels=None, ax=None, scatter_kwargs={}, label_kwargs={}):
        if ax is None:
            ax = plt.gca()
        scatter_kwargs = util.merge_dicts(
            {
                "s": 75, 
                "edgecolors": "none", 
            },
            scatter_kwargs
        )
        label_kwargs = util.merge_dicts(
            {"color": "w", "va": "center", "ha": "center", "fontsize": 5},
            label_kwargs
        )
        if not labels is None:
            for _x, _y, _label in zip(x, y, labels):
                plt.text(_x, _y, _label, **label_kwargs)
        ax.scatter(x, y, **scatter_kwargs)

    def plot_trend(self, x, y, ax=None, **kwargs):
        kwargs = util.merge_dicts(
            {"color": "k", "linestyle": "--", "linewidth": 0.75},
            kwargs,
        )
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        self.plot_line(x, p(x), ax=ax, **kwargs)

    def plot_axis(self, loc, axis="x", **kwargs):
        if axis != "x" and axis != "y":
            raise ValueError("Only x and y axis lines supported. Received axis=\"%s\"" % (axis))
        kwargs = util.merge_dicts({"color": "k"}, kwargs)
        plot_fn = plt.axhline
        if axis == "y":
            plot_fn = plt.axvline
        plot_fn(loc, **kwargs)

    def xlim(self, left=None, right=None, margin=0.0, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        curr_left, curr_right = ax.get_xlim()
#        if not (left is None or right is None):
        if not left is None:
            ax.set_xlim(left=left, **kwargs)
        if not right is None:
            ax.set_xlim(right=right, **kwargs)
        return curr_left, curr_right 

    def ylim(self, bottom=None, top=None, margin=0.0, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        curr_bottom, curr_top = ax.get_ylim()
#        if not (bottom is None or top is None):
        if not bottom is None:
            ax.set_ylim(bottom=bottom, **kwargs)
        if not top is None:
            ax.set_ylim(top=top, **kwargs)
        return curr_bottom, curr_top 

    def lim(self, xlim=None, ylim=None, xmargin=0.0, ymargin=0.0, ax=None, xlim_kwargs={}, ylim_kwargs={}):
        if xlim is None:
            xlim = (None, None)
        if ylim is None:
            ylim = (None, None)
        left, right = self.xlim(xlim[0], xlim[1], xmargin, ax, **xlim_kwargs)
        bottom, top = self.ylim(ylim[0], ylim[1], ymargin, ax, **ylim_kwargs) 
        return (left, right), (bottom, top)

    def xticks(self, locs=None, labels=None, ax=None, **kwargs):
        if locs is None and not labels is None:
            locs = range(len(labels))
        if ax is None:
            ax = plt.gca()
        _locs, _labels = ax.get_xticks(), ax.get_xticklabels()
        if not locs is None:
            ax.set_xticks(locs, labels, **kwargs)
        return _locs, _labels

    def yticks(self, locs=None, labels=None, ax=None, **kwargs):
        if locs is None and not labels is None:
            locs = range(len(labels))
        if ax is None:
            ax = plt.gca()
        _locs, _labels = ax.get_yticks(), ax.get_yticklabels()
        if not locs is None:
            ax.set_yticks(locs, labels, **kwargs)
        return _locs, _labels

    def ticks(self, xticks=None, yticks=None, ax=None, xtick_kwargs={}, ytick_kwargs={}):
        if xticks is None:
            xticks = (None, None)
        elif xticks == []:
            xticks = ([], [])
        else:
            raise ValueError(xticks)
        if yticks is None:
            yticks = (None, None)
        elif yticks == []:
            yticks = ([], [])
        else:
            raise ValueError(yticks)
        xlocs, xlabels = self.xticks(xticks[0], xticks[1], ax, **xtick_kwargs)
        ylocs, ylabels = self.yticks(yticks[0], yticks[1], ax, **ytick_kwargs)
        return (xlocs, xlabels), (ylocs, ylabels)

    def xlabel(self, label, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        prev_label = ax.get_xlabel()
        if not label is None:
            ax.set_xlabel(label, **kwargs)
        return prev_label

    def ylabel(self, label, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        prev_label = ax.get_ylabel()
        if not label is None:
            ax.set_ylabel(label, **kwargs)
        return prev_label

    def labels(self, xlabel=None, ylabel=None, ax=None, xlabel_kwargs={}, ylabel_kwargs={}):
        prev_xlabel = self.xlabel(xlabel, ax=ax, **xlabel_kwargs)
        prev_ylabel = self.ylabel(ylabel, ax=ax, **ylabel_kwargs)
        return prev_xlabel, prev_ylabel

    def title(self, label=None, fontdict=None, loc="center", y=None, pad=6.0, ax=None):
        if ax is None:
            ax = plt.gca()
        if label is None:
            return ax.get_title()
        return ax.set_title(label, fontdict=fontdict, loc=loc, y=y, pad=pad)

    def style(self, style, ax=None, **kwargs):
        if style in plt.style.available:
            if ax is None:
                plt.style.use(style, **kwargs)
            else:
                raise ValueError()
        else:
            if style =="grid":
                kwargs = util.merge_dicts({"linestyle": ":"}, kwargs)
                if ax is None:
                    ax = plt.gca()
                ax.grid(**kwargs)
            else:
                raise ValueError()

    def legend(self, handles=None, labels=None, ax=None, style="standard", **kwargs):
        if style == "center":
            kwargs = util.merge_dicts(
                {
                    "loc": "upper center", 
                    "bbox_to_anchor": (0.5, 1.05), 
                    "ncol": 3, 
                    "fancybox": True, 
                    "prop": {"size": 7}, 
                }, 
                kwargs
            )
        elif style == "standard":
            kwargs = util.merge_dicts(
                {}, 
                kwargs
            )
        else:
            raise ValueError(style)
#        kwargs = util.merge_dicts({"size": 7}, kwargs)
        if ax is None:
            ax = plt.gca()
        if not (handles is None or labels is None):
            ax.legend(handles, labels, **kwargs)
        elif not handles is None:
            ax.legend(handle=handles, **kwargs)
        elif not labels is None:
            ax.legend(labels, **kwargs)
        else:
            ax.legend(**kwargs)

    def figure(self, size=(8, 8)):
        return plt.figure(figsize=size)

    def subplots(self, n=1, size=(8, 8)):
        return plt.subplots(n, figsize=size)

    def save_figure(self, path, dpi=200, fig=None):
        if fig is None:
            fig = plt
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close()

    def show_figure(self):
        plt.show()
        plt.close()

    def close(self):
        plt.close()

    def plot_learning_curve(self, train, valid, test, path):
        plt.plot(train, color="k", label="Training", linewidth=self.line_width)
        plt.plot(valid, color="r", label="Validation", linewidth=self.line_width)
        plt.plot(test, color="g", label="Testing", linewidth=self.line_width)
        plt.axvline(np.argsort(valid)[0], linestyle=":", color="k")
        ymax = max(max(train), max(valid), max(test))
        bottom, top = plt.ylim()
        plt.ylim(bottom=0, top=0.8*ymax)
        plt.ylabel("Mean Squared Error")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(path)
        plt.close()


    # Precondition: predictions and groundtruth have the following shapes
    #   Yhats.shape=[n_windows, n_temporal_in, n_spatial, n_response]
    #   groundtruth.shape=[n_windows, n_temporal_in, n_spatial, n_response]
    def plot_model_fit(self, Yhat, spatmp, partition, var):
        # Unpacke variables
        exec_var, plt_var, proc_var = var.execution, var.plotting, var.processing
        plt_dir, line_opts, fig_opts = plt_var.get(["plot_dir", "line_options", "figure_options"])
        line_kwargs = plt_var.line_kwargs
        dataset = exec_var.get("dataset", partition)
        n_temporal_in, n_temporal_out = var.mapping.temporal_mapping
        n_predictor, n_response = spatmp.misc.get(["n_predictor", "n_response"])
        response_features, response_indices = spatmp.misc.get(["response_features", "response_indices"])
        n_spatial, spatial_labels, spatial_indices = spatmp.original.get(
            ["n_spatial", "spatial_labels", "spatial_indices"],
            partition
        )
        # Filter statistics
        statistics = spatmp.statistics
        if exec_var.principle_data_form == "reduced":
            statistics = spatmp.reduced_statistics
        mins = spatmp.filter_axis(
            statistics.minimums,
            [0, 1],
            [spatial_indices, response_indices]
        )
        maxes = spatmp.filter_axis(
            statistics.maximums,
            [0, 1],
            [spatial_indices, response_indices]
        )
        meds = spatmp.filter_axis(
            statistics.medians,
            [0, 1],
            [spatial_indices, response_indices]
        )
        means = spatmp.filter_axis(
            statistics.means,
            [0, 1],
            [spatial_indices, response_indices]
        )
        stddevs = spatmp.filter_axis(
            statistics.standard_deviations,
            [0, 1],
            [spatial_indices, response_indices]
        )
        # Get groundtruth data (Y, etc) then filter and reformat predictions (Yhat) to match groundtruth
        #   Pull groundtruth data - response_features Y, temporal_labels, and their periodic_indices
        if exec_var.principle_data_form == "original":
            Y = spatmp.original.get("response_features", partition)
            temporal_labels = spatmp.original.get("temporal_labels", partition)
            periodic_indices = spatmp.original.get("periodic_indices", partition)
            #   Reformat and filter predictions - filter for contiguous outputs then reshape
            n_sample, n_temporal_out, n_spatial, n_feature = Yhat.shape
            contiguous_window_indices = util.contiguous_window_indices(n_sample, n_temporal_out, 1)
            Yhat = Yhat[contiguous_window_indices,:,:,:]
            Yhat = np.reshape(Yhat, (-1,) + Yhat.shape[-2:])
        elif exec_var.principle_data_form == "reduced":
            Y = spatmp.reduced.get("response_features", partition)
            temporal_labels = spatmp.reduced.get("temporal_labels", partition)
            periodic_indices = spatmp.reduced.get("periodic_indices", partition)
            Y, temporal_labels, periodic_indices = Y[0,:,:,:], temporal_labels[0,:], periodic_indices[0,:]
            #   Reformat and filter predictions - filter for contiguous outputs then reshape
            n_channel, n_sample, n_temporal_out, n_spatial, n_feature = Yhat.shape
            contiguous_window_indices = util.contiguous_window_indices(n_sample, n_temporal_out, 1)
            Yhat = Yhat[0,contiguous_window_indices,:,:,:]
            Yhat = np.reshape(Yhat, (-1,) + Yhat.shape[-2:])
        # Pull spatial elements selected for plotting
        try:
            plotted_spatial_indices = spatmp.indices_from_selection(
                spatmp.original.get("spatial_labels", partition), 
                plt_var.spatial_selection
            )
        except:
            plotted_spatial_indices = np.arange(n_spatial)
#        Y, Yhat = Y[:,plotted_spatial_indices,:], Yhat[:,plotted_spatial_indices,:]
#        n_spatial = len(plotted_spatial_indices)
        #####################
        event_data = np.zeros((Y.shape[0],n_spatial,n_response,2))
        cache_plot_data = False
#        cache_plot_data = True
        plt_range = [0.0, 1.0]
        # Set colors
        alpha_min = 0.40
        alpha_max = 0.80
        alpha_interval = [alpha_min, alpha_max]
        if n_spatial > 1:
            alphas = np.linspace(alpha_interval[0], alpha_interval[1], n_spatial)
            alphas = np.flip(alphas)
        else:
            alphas = np.array([alpha_max])
        cmap = plt.get_cmap(line_kwargs["groundtruth"]["color"])
        groundtruth_colors = [cmap(1.25*i) for i in alphas]
        cmap = plt.get_cmap(line_kwargs["prediction"]["color"])
        Yhat_colors = [cmap(i) for i in alphas]
        cmap = plt.get_cmap("Blues")
        median_colors = [cmap(i) for i in alphas]
        cmap = plt.get_cmap("Blues")
        mean_colors = [cmap(i) for i in alphas]
        cmap = plt.get_cmap("Blues")
        std_colors = [cmap(0.25*i) for i in alphas]
        cmap = plt.get_cmap("Reds")
        extreme_colors = [cmap(1.0*i) for i in alphas]
        for o in range(n_response):
            y_min, y_max = sys.float_info.max, sys.float_info.min
            spatial_indices = plotted_spatial_indices
            if self.feature_plot_order_map.get(response_features[o], "ascending") == "descending":
                spatial_indices = np.flip(spatial_indices)
            n_spa_plotted, n_spa_per_plt = 0, plt_var.get("n_spatial_per_plot")
            for s in spatial_indices:
                if "confusion" in line_opts:
                    self._plot_zscore_confusion(
                        Yhat[:,s,o],
                        Y[:,s,o][n_temporal_in:],
                        means[periodic_indices[n_temporal_in:],s,o],
                        stddevs[periodic_indices[n_temporal_in:],s,o]
                    )
                    path = plt_dir + os.sep + "Confusion_Partition[%s]_Subbasins[%s]_Response[%s].png" % (
                        partition,
                        ",".join(spatial_labels[s:s+1]),
                        response_features[o]
                    )
                    self.save_figure(path)
                if "groundtruth" in line_opts:
                    self._plot_groundtruth(
                        Y[:,s,o],
                        dataset,
                        n_spatial,
                        spatial_labels[s],
                        n_spa_per_plt,
                        groundtruth_colors[n_spa_plotted%n_spa_per_plt],
                    )
                if "prediction" in line_opts:
                    self._plot_prediction(
                        Yhat[:,s,o],
                        n_temporal_in,
                        n_temporal_out,
                        n_spatial,
                        spatial_labels[s],
                        n_spa_per_plt,
                        Yhat_colors[n_spa_plotted%n_spa_per_plt],
                        line_kwargs["prediction"]["label"],
                    )
                if "groundtruth_extremes" in line_opts:
                    self._plot_groundtruth_extremes(
                        Y[:,s,o],
                        means[periodic_indices,s,o],
                        stddevs[periodic_indices,s,o],
                        dataset,
                        n_spatial,
                        spatial_labels[s],
                        n_spa_per_plt,
                        plt_range
                    )
                if "prediction_extremes" in line_opts:
                    self._plot_prediction_extremes(
                        Yhat[:,s,o],
                        means[windowed_output_periodic_indices,s,o],
                        stddevs[windowed_output_periodic_indices,s,o],
                        n_temporal_in,
                        n_temporal_out,
                        n_spatial,
                        spatial_labels[s],
                        n_spa_per_plt,
                        plt_range
                    )
                if len(re.findall(".*median", ",".join(line_opts))) > 0:
                    if "median" in line_opts:
                        spatial_response_meds = meds[periodic_indices,s,o]
                    elif "temporal_median" in line_opts:
                        spatial_response_meds = np.median(meds[periodic_indices,:,o], axis=1)
                    elif "spatial_median" in line_opts:
                        spatial_response_med = np.median(meds[:,s,o], axis=0)
                        spatial_response_meds = np.tile(spatial_response_med, temporal_labels.shape[0])
                    elif "feature_median" in line_opts:
                        spatial_response_med = np.median(meds[:,s,o], axis=(0,1))
                        spatial_response_meds = np.tile(spatial_response_med, temporal_labels.shape[0])
                    label = "Median"
                    if n_spatial > 1 and n_spa_per_plt > 1:
                        label = ("Subbasin %s " % (spatial_labels[s])) + label
                    plt.plot(spatial_response_meds, color=median_colors[s], linestyle="-.", label=label, linewidth=self.line_width)
                    y_min = min(y_min, np.min(spatial_response_meds))
                    y_max = max(y_max, np.max(spatial_response_meds))
                if len(re.findall(".*mean", ",".join(line_opts))) > 0:
                    if "mean" in line_opts:
                        spatial_response_means = means[periodic_indices,s,o]
                    elif "temporal_mean" in line_opts:
                        spatial_response_means = np.mean(means[periodic_indices,:,o], axis=1)
                    elif "spatial_mean" in line_opts:
                        spatial_response_mean = np.mean(means[:,s,o], axis=0)
                        spatial_response_means = np.tile(spatial_response_mean, temporal_labels.shape[0])
                    elif "feature_mean" in line_opts:
                        spatial_response_mean = np.mean(means[:,s,o], axis=(0,1))
                        spatial_response_means = np.tile(spatial_response_mean, temporal_labels.shape[0])
                    label = "Mean"
                    label = "$\mu$"
                    if n_spatial > 1 and n_spa_per_plt > 1:
                        label = ("Subbasin %s " % (spatial_labels[s])) + label
                    start = round(plt_range[0] * spatial_response_means.shape[0])
                    end = round(plt_range[1] * spatial_response_means.shape[0])
                    plt.plot(spatial_response_means, color=mean_colors[s], linestyle="-", label=label, linewidth=self.line_width)
                    y_min = min(y_min, np.min(spatial_response_means))
                    y_max = max(y_max, np.max(spatial_response_means))
                if len(re.findall(".*stddev", ",".join(line_opts))) > 0:
                    if "stddev" in line_opts:
                        spatial_response_stddevs = stddevs[periodic_indices,s,o]
                        spatial_response_means = means[periodic_indices,s,o]
                    elif "temporal_stddev" in line_opts:
                        spatial_response_stddevs = np.std(stddevs[periodic_indices,:,o], axis=1)
                        spatial_response_means = np.mean(means[periodic_indices,:,o], axis=1)
                    elif "spatial_stddev" in line_opts:
                        spatial_response_stddev = np.std(stddevs[:,s,o], axis=0)
                        spatial_response_stddevs = np.tile(spatial_response_stddev, temporal_labels.shape[0])
                        spatial_response_mean = np.mean(means[:,s,o], axis=0)
                        spatial_response_means = np.tile(spatial_response_mean, temporal_labels.shape[0])
                    elif "feature_stddev" in line_opts:
                        spatial_response_stddev = np.std(stddevs[:,s,o], axis=(0,1))
                        spatial_response_stddevs = np.tile(spatial_response_stddev, temporal_labels.shape[0])
                        spatial_response_mean = np.mean(means[:,s,o], axis=(0,1))
                        spatial_response_means = np.tile(spatial_response_mean, temporal_labels.shape[0])
                    start = round(plt_range[0] * spatial_response_stddevs.shape[0])
                    end = round(plt_range[1] * spatial_response_stddevs.shape[0])
                    indices = np.arange(end-start)
                    indices = np.arange(0, spatial_response_stddevs.shape[0])
                    cmap = plt.get_cmap("Blues")
                    stddev_alphas = np.array([0.25,0.4,0.55])
                    stddev_alphas = np.flip(stddev_alphas)
                    std_colors = [cmap(i) for i in stddev_alphas]
                    z_scores = [2.0,1.5,1.0]
                    for i in range(3):
                        label = "Standard Deviation %.1f" % (z_scores[i])
                        label = "$\mu \pm %.1f \cdot \sigma$" % (z_scores[i])
                        if n_spatial > 1 and n_spa_per_plt > 1:
                            label = ("Subbasin %s " % (spatial_labels[s])) + label
                        std_interval = np.array([-1, 1]) * z_scores[i]
                        lower_bounds = spatial_response_means + std_interval[0] * spatial_response_stddevs
                        if response_features[o] == "FLOW_OUTcms":
                            lower_bounds[lower_bounds < 0] = 0
                        upper_bounds = spatial_response_means + std_interval[1] * spatial_response_stddevs
                        plt.fill_between(indices, lower_bounds, upper_bounds, color=std_colors[i], linestyle="-", label=label, linewidth=self.line_width)
                    y_min = min(y_min, np.min(lower_bounds))
                    y_max = max(y_max, np.max(upper_bounds))
                y_min = min(y_min, np.min(Yhat[:,s,o]))
                y_max = max(y_max, np.max(Yhat[:,s,o]))
                y_min = min(y_min, np.min(Y[:,s,o]))
                y_max = max(y_max, np.max(Y[:,s,o]))
                n_spa_plotted += 1
                if (n_spa_per_plt > 0 and n_spa_plotted % n_spa_per_plt == 0) or (n_spa_per_plt < 0 and n_spa_plotted == len(spatial_labels)):
                    if "xlabel" in fig_opts:
                        plt.xlabel("Time", fontsize=8)
                    if "ylabel" in fig_opts:
                        
                        plt.ylabel(
                            self.feature_ylabel_map.get(response_features[o], response_features[o]), 
                            fontsize=8
                        )
                    if "xticks" in fig_opts:
                        self._plot_xticks(temporal_labels, plt_range)
                    if "yticks" in fig_opts:
                        self._plot_yticks([y_min, y_max])
                    if "lims" in fig_opts:
                        self._plot_xylim([0, temporal_labels.shape[0]], [y_min, y_max], plt_range)
                    if "legend" in fig_opts:
                        self._plot_legend()
                    if "save" in fig_opts:
                        self._savefigs(
                            partition,
                            spatmp.misc.spatial_label_field, 
                            spatial_labels[s:s+1],
                            response_features[o],
                            line_opts,
                            plt_range,
                            plt_dir
                        )
                    y_min, y_max = sys.float_info.max, sys.float_info.min


    def _plot_groundtruth(self, Y, dataset, n_spa, spa_label, n_spa_per_plt, color):
        label = self.dataset_legend_map.get(dataset, "Groundtruth")
        if n_spa > 1 and n_spa_per_plt > 1:
            label = ("Subbasin %s " % (str(spa_label))) + label
        plt.plot(Y, color=color, linestyle="-", label=label, linewidth=self.gt_line_width)


    def _plot_prediction(self, Yhat, n_tmp_in, n_tmp_out, n_spa, spa_label, n_spa_per_plt, color, label):
        if n_spa > 1 and n_spa_per_plt > 1:
            label = ("Subbasin %s " % (str(spa_label))) + label
        indices = np.arange(n_tmp_in, n_tmp_in + Yhat.shape[0])
        plt.plot(indices, Yhat, color=color, linestyle="-", label=label, linewidth=self.pred_line_width)


    def _plot_groundtruth_extremes(self, Y, means, stddevs, dataset, n_spa, spa_label, n_spa_per_plt, plt_range):
        interval_events_map = util.compute_events(Y, means, stddevs)
        interval_label_map = {
            "-8,-2": "Extremely Dry",
            "-2,-1.5": "Severely Dry",
            "-1.5,-1": "Moderately Dry",
            "-1,1": "Near Normal",
            "1,1.5": "Moderately Wet",
            "1.5,2": "Severely Wet",
            "2,8": "Extremely Wet"
        }
        for key, value in interval_label_map.items():
            interval_label_map[key] = value.replace("ly Dry", "").replace("ly Wet", "")
        colors = ["red", "darkorange", "gold", "green", "gold", "darkorange", "red"]
        plot_mask = [False, False, False, True, False, False, False]
        marker, lw = "x", 0
        for interval, color, mask in zip(interval_label_map.keys(), colors, plot_mask):
            events = interval_events_map[interval]
            start = round(plt_range[0] * events.shape[0])
            end = round(plt_range[1] * events.shape[0])
            if np.ma.count(events[start:end]) > 0 and not mask:
                label = "%s %s" % (self.dataset_legend_map[dataset], interval_label_map[interval])
                if n_spa > 1 and n_spa_per_plt > 1:
                    label = ("Subbasin %s " % (spa_label)) + label
                plt.plot(
                    events,
                    color=color,
                    linestyle="-",
                    marker=marker,
                    label=label,
                    linewidth=2*lw,
                    markersize=0.75*self.marker_size
                )


    def _plot_prediction_extremes(self, Yhat, means, stddevs, n_tmp_in, n_tmp_out, n_spa, spa_label, n_spa_per_plt, plt_range):
        interval_events_map = util.compute_events(Yhat, means, stddevs)
        interval_label_map = {
            "-8,-2": "Extremely Dry",
            "-2,-1.5": "Severely Dry",
            "-1.5,-1": "Moderately Dry",
            "-1,1": "Near Normal",
            "1,1.5": "Moderately Wet",
            "1.5,2": "Severely Wet",
            "2,8": "Extremely Wet"
        }
        for key, value in interval_label_map.items():
            interval_label_map[key] = value.replace("ly Dry", "").replace("ly Wet", "")
        colors = ["red", "darkorange", "gold", "green", "gold", "darkorange", "red"]
        plot_mask = [False, False, False, True, False, False, False]
        marker, lw = "+", 0
        for interval, color, mask in zip(interval_label_map.keys(), colors, plot_mask):
            events = interval_events_map[interval]
            start = round(plt_range[0] * events.shape[0])
            end = round(plt_range[1] * events.shape[0])
            if np.ma.count(events[start:end]) > 0 and not mask:
                label = "Prediction %s" % (interval_label_map[interval])
                if n_spa > 1 and n_spa_per_plt > 1:
                    label = ("Subbasin %s " % (spa_label)) + label
                indices = np.arange(n_tmp_in, n_tmp_in + events.shape[0])
                plt.plot(
                    indices,
                    events,
                    color=color,
                    linestyle="-",
                    marker=marker,
                    label=label,
                    linewidth=2*lw,
                    markersize=0.75*self.marker_size
                )


    def _plot_zscore_confusion(self, Yhat, Y, means, stddevs):
        confusion = util.compute_zscore_confusion(Yhat, Y, means, stddevs, normalize=False)
        print(confusion)
        interval_labels = [
            "Extremely Dry",
            "Severely Dry",
            "Moderately Dry",
            "Near Normal",
            "Moderately Wet",
            "Severely Wet",
            "Extremely Wet"
        ]
        df = pd.DataFrame(confusion, index=interval_labels, columns=interval_labels)
        from pretty_confusion_matrix import pp_matrix
        pp_matrix(df, cmap="Blues", pred_val_axis="col")
        self.labels("Predicted Class", "Actual Class")
        self.ticks(xtick_kwargs={"rotation": 45, "fontsize": 7}, ytick_kwargs={"fontsize": 7})


    def _plot_legend(self, standard=False):
        size = 9
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) > 3:
            size = 7
        if standard:
            plt.legend(prop={"size": size})
        else:
            by_label = dict(zip(labels, handles))
            n_col = len(by_label.keys()) // 3 + 1
            n_col = 4
            plt.legend(
                by_label.values(),
                by_label.keys(),
                loc="upper center",
                bbox_to_anchor=(0.5,1.1),
                ncol=n_col,
                fancybox=True,
                shadow=True,
                prop={"size": size}
            )

    def _plot_xylim(self, x_interval, y_interval, plt_range):
        x_range = x_interval[1] - x_interval[0]
        start = round(plt_range[0] * x_range + x_interval[0])
        end = round(plt_range[1] * x_range + x_interval[0])
        plt.xlim(start, end)
        y_range = y_interval[1] - y_interval[0]
        plt.ylim(y_interval[0]-y_range/20, y_interval[1]+y_range/20)


    def _plot_xticks(self, xlabels, plt_range):
        xtick_indices = np.arange(xlabels.shape[0])
        xtick_labels = xlabels[xtick_indices]
        n_xticks = 8
        n_xticks = int(n_xticks / (plt_range[1] - plt_range[0]))
        start = round(plt_range[0] * (xtick_indices.shape[0] - 1))
        end = round(plt_range[1] * (xtick_indices.shape[0] - 1))  + 1
        indices = np.linspace(0, xtick_indices.shape[0]-1, n_xticks, endpoint=True, dtype=np.int)
        xtick_indices = xtick_indices[indices]
        xtick_labels = xtick_labels[indices]
        plt.xticks(xtick_indices, xtick_labels, fontsize=7, rotation=45)


    def _plot_yticks(self, y_interval):
        ytick_indices, ytick_labels = plt.yticks()
#        ytick_labels = np.linspace(y_interval[0], y_interval[1], 7)
#        plt.yticks(ytick_indices, ytick_labels, fontsize=6)
        plt.yticks(fontsize=7)


    def _savefigs(self, partition, spatial_label_field, spatial_labels, feature, line_opts, plt_range, plt_dir):
        spatials = ",".join(map(str, spatial_labels))
        opts = list(line_opts)
        if "confusion" in opts:
            opts.remove("confusion")
        fname = "Evaluation_Partition[%s]_%s[%s]_Response[%s]_Options[%s].png" % (
            partition, spatial_label_field.capitalize(), spatials, feature, ",".join(opts)
        )
        path = os.sep.join([plt_dir, fname])
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    def plot_error_scatter(self, model_i_errors, model_j_errors, datasets, var, path):
        def get_node_indegrees(datasets, var):
            dataset = datasets.get("dataset", "train")
            return np.array(list(dataset.graph.original.get("node_indegree_map", var.partition).values()), dtype=int)
        def get_node_metrics(model_errors, var):
            return np.array(list(model_errors.get(var.metric, var.partition).get(var.response_feature).values()))
        def get_node_stds(datasets, var):
            dataset = datasets.get("dataset", "train")
            return dataset.spatiotemporal.filter_axis(
                dataset.spatiotemporal.statistics.standard_deviations, 
                [0, 1], 
                [
                    dataset.spatiotemporal.original.get("spatial_indices", var.partition), 
                    dataset.spatiotemporal.misc.response_indices
                ], 
            ) 
        # Get spatial resolution errors
        points_legend_label = datasets.get("dataset", var.partition).spatiotemporal.misc.spatial_label_field
        spatial_error_map_i = model_i_errors.get(var.metric, var.partition).get(var.response_feature)
        spatial_error_map_j = model_j_errors.get(var.metric, var.partition).get(var.response_feature)
        spatial_labels_i, errors_i = list(spatial_error_map_i.keys()), np.array(list(spatial_error_map_i.values()))
        spatial_labels_j, errors_j = list(spatial_error_map_j.keys()), np.array(list(spatial_error_map_j.values()))
        if 0:
            print("model_i_errors =", spatial_error_map_i)
            print("model_j_errors =", spatial_error_map_j)
            print("No. Nodes =", len(spatial_labels_i))
        # Unpack the data
        dataset = datasets.get("dataset", var.partition)
        spatiotemporal = dataset.spatiotemporal
        spatial = dataset.spatial
        temporal = dataset.temporal
        graph = dataset.graph
        #
        if var.plot_args.x_axis == "in-degree":
            x_i = get_node_indegrees(datasets, var)
            x_j = get_node_indegrees(datasets, var)
            xlabel = "Node In-degree"
        elif var.plot_args.x_axis == "metric":
            x_i = get_node_metrics(model_i_errors, var)
            x_j = get_node_metrics(model_j_errors, var)
            xlabel = "$%s_{%s}$" % (var.metric, var.model_i)
        elif var.plot_args.x_axis == "stddev":
            x_i = get_node_stds(datasets, var)
            x_j = get_node_stds(datasets, var)
            x_i, x_j = np.squeeze(x_i), np.squeeze(x_j)
            feature_name = var.response_feature
            if var.response_feature == "FLOW_OUTcms":
                feature_name = "Streamflow"
            elif var.response_feature == "SWmm":
                feature_name = "Soil Water"
            elif var.response_feature == "speedmph":
                feature_name = "MPH"
            xlabel = "$\sigma_{%s}$" % (feature_name)
        else:
            raise NotImplementedError("var.plot_args.x_axis=%s" % (var.plot_args.x_axis))
        if var.plot_args.z_axis is None:
            z_i = np.ones(x_i.shape)
            z_j = np.ones(x_j.shape)
        elif var.plot_args.z_axis == "in-degree":
            z_i = get_node_indegrees(datasets, var)
            z_j = get_node_indegrees(datasets, var)
            xlabel = "Node In-degree"
        elif var.plot_args.z_axis == "metric":
            z_i = get_node_metrics(model_i_errors, var)
            z_j = get_node_metrics(model_j_errors, var)
            xlabel = "$%s_{%s}$" % (var.metric, var.model_i)
        elif var.plot_args.z_axis == "stddev":
            z_i = get_node_stds(datasets, var)
            z_j = get_node_stds(datasets, var)
            z_i, z_j = np.squeeze(z_i), np.squeeze(z_j)
        else:
            raise NotImplementedError("var.plot_args.z_axis=%s" % (var.plot_args.z_axis))
        if var.plot_args.y_axis == "diff":
            x = x_i
            y = errors_j - errors_i
            z = z_i
            if var.plot_args.z_axis is None:
                z_alpha = z
                z_scale = z * 75
                z_color = "b"
            else:
                z_alpha = util.minmax_transform(z, min(z), max(z), a=1/10, b=1)
                z_scale = util.minmax_transform(z, min(z), max(z), a=10, b=100)
                z_alpha = np.ones(z.shape)
                z_color = [
                    plt.get_cmap("Blues")(fac) for fac in util.minmax_transform(z, min(z), max(z), a=3/8, b=1)
                ]
            plt.axhline(0, color="k", linestyle="-", alpha=4/8, zorder=0)
            labels_i, labels_j = None, None
            if "point_labels" in var.plot_args.plot_options:
                labels_i, labels_j = spatial_labels_i, spatial_labels_j
            self.plot_scatter(
                x,
                y,
                labels=labels_i,
                scatter_kwargs={
                    "c": z_color, 
                    "label": points_legend_label.capitalize(), 
                    "alpha": z_alpha, 
                    "s": z_scale, 
                }
            )
            if "trend" in var.plot_args.plot_options:
                self.plot_trend(x, y, label="Trend")
            plt.ylabel("$\Delta$ %s (%s $\longrightarrow$ %s)" % (var.metric, var.model_i, var.model_j))
        elif var.plot_args.y_axis == "raw":
            y_i, y_j = errors_i, erorrs_j
            self.plot_scatter(x_i, y_i, scatter_kwargs={"c": "r", "label": var.model_i})
            if "trend" in var.plot_args.plot_options:
                self.plot_trend(x, y, color="r", label="%s Trend" % (var.model_i))
            x, y = indegrees, errors_j
            self.plot_scatter(x_j, y_j, scatter_kwargs={"c": "b", "label": var.model_j})
            if "trend" in var.plot_args.plot_options:
                self.plot_trend(x, y, color="b", label="%s Trend" % (var.model_j))
            plt.ylabel(var.metric)
        else:
            raise NotImplementedError("var.plot_args.y_axis=%s" % (var.plot_args.y_axis))
        if var.plot_args.x_axis == "in-degree":
            self.xticks(None, np.arange(0, max(x)+1))
            self.xlim(-0.5, max(x)+0.5)
#        else:
#            self.ticks([None, None])
#        plt.xlim(min(x), max(indegrees)+0.25)
        plt.xlabel(xlabel)
        plt.legend()
        plt.savefig(path, bbox_inches="tight")
        plt.close()


def plot_wabash():
    watershed = "WabashRiver"
    import pandas as pd
    plt = Plotting()
    path = "Data" + os.sep + "WabashSubbasins_IDtoHUC.csv"
    df = pd.read_csv(path)
    subbasin_river_map = pd.Series(df.NameHUC8.values, index=df.subbasin.values).to_dict()
    replaced_names = ["Wabash", "White"]
    for subbasin, river in subbasin_river_map.items():
        for replaced_name in replaced_names:
            if replaced_name in river:
                subbasin_river_map[subbasin] = replaced_name
            subbasin_river_map[subbasin] = subbasin_river_map[subbasin].replace("-Haw", "")
    fnames = ["Wabash_HUC12", "subs1", "riv1", "outlets1", "monitoring_points1"]
    fnames.remove("Wabash_HUC12")
    fnames.remove("riv1")
    fnames.remove("outlets1")
    fnames.remove("monitoring_points1")
    fname_path_map = {
        fname: os.sep.join(["Data", "LittleRiverWatershedShapes", fname+".shp"]) for fname in fnames
    }
    fname_item_map = {"riv1": "rivers", "subs1": "subbasins"}
    item_shapes_map = {fname_item_map[fname]: shapefile.Reader(path) for fname, path in fname_path_map.items()}
    path = os.sep.join(["Plots", "Watershed_Components[%s]_Watershed[%s].png"]) % (
        ",".join(item_shapes_map.keys()),
        watershed
    )
    plt.plot_watershed(item_shapes_map, subbasin_river_map, path, highlight=False, watershed=watershed, river_opts={"color_code": False, "name": False})


def plot_little():
    watershed = "LittleRiver"
    plt = Plotting()
    subbasin_river_map = {}
    fnames = ["basins", "gis_streams"]
#    fnames.remove("gis_streams")
    fname_path_map = {
        fname: os.sep.join(["Data", "LittleRiverWatershedShapes", fname+".shp"]) for fname in fnames
    }
    fname_item_map = {"gis_streams": "rivers", "basins": "subbasins"}
    item_shapes_map = {fname_item_map[fname]: shapefile.Reader(path) for fname, path in fname_path_map.items()}
    path = os.sep.join(["Plots", "Watershed_Components[%s]_Watershed[%s].png"]) % (
        ",".join(item_shapes_map.keys()),
        watershed
    )
    plt.plot_watershed(item_shapes_map, subbasin_river_map, path, highlight=False, watershed=watershed)


class Spatial(Plotting):

    def plot_model_fit(self, Y, Yhat, dataset, partition, var):
        spa = dataset.spatial
        for i, feature in enumerate(spa.misc.response_features):
            self.plot_bar(None, Y[:,i], alpha=7/8, color="k", label="Gt[%s]" % (feature))
            self.plot_bar(None, Yhat[:,i], alpha=5/8, color="r", label="Pr[%s]" % (feature))
        self.plot_axis(10, "x", linestyle="--", linewidth=5/8)
        self.lim([-1, Y.shape[0]], [spa.statistics.minimums[i], 21/20*spa.statistics.maximums[i]])
        self.legend()
        path = os.sep.join(
            [
                var.plotting.plot_dir, 
                "Evaluation_Partition[%s]_Response[%s].png" % (partition, ",".join(spa.misc.response_features))
            ]
        )
        self.save_figure(path)


if __name__ == "__main__":
    pass
