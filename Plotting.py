import progressbar as pb
import time
import argparse
import os
import sys
import re
import seaborn as sns
import pandas as pd
import numpy as np
from progressbar import ProgressBar
import shapefile
import pickle
import matplotlib.pyplot as plt
from scipy.stats import skew
import Utility as util
import NetworkProperties as netprop
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
os.environ["PROJ_LIB"] = "C:\\Users\\Nicholas\\Anaconda3\\Library\\share"
from mpl_toolkits.basemap import Basemap
from Container import Container
import polylabel
import matplotlib.patheffects as PathEffects


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


    line_width = 1
    gt_line_width = 1.25 * line_width
    pred_line_width = 0.75 * line_width
    marker_size = 7
    month_labels = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "January"]
    feature_idx_map = {"FLOW_OUTcms": 3, "SWmm": 10, "PRECIPmm": 7}
    feature_fullname_map = {"FLOW_OUTcms": "Streamflow", "SWmm": "Soil Moisture"}
    feature_SIunit_map = {"FLOW_OUTcms": "$m^{3}/s$", "SWmm": "$mm$"}
    feature_ylabel_map = {}
    for feature in feature_fullname_map.keys():
        feature_ylabel_map[feature] = "%s (%s)" % (feature_fullname_map[feature], feature_SIunit_map[feature])
    dataset_legend_map = {"littleriver_observed": "Little(GT)", "wabashriver_swat": "Wabash(SWAT)", "wabashriver_observed": "Wabash(GT)"}
    partition_fullname_map = {"train": "Training", "valid": "Validation", "test": "Testing"}
    partition_codename_map = {"train": "Train", "valid": "Valid", "test": "Test"}


    def __init__(self):
        self.set("plot_dir", "Plots")
        self.set("lines", [])


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



    def plot_cluster_heatmap(self, mat, xtick_labels=[], ytick_labels=[], x_label="", y_label="", plot_numbers=True, transpose=True, size=(12, 12), path=None):
        data = {}
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
        rowcol_cluster = [True, True]
        cg = sns.clustermap(data, method="average", metric="correlation", figsize=size, annot=plot_numbers, annot_kws={"fontsize": 0.7*size[0]}, row_cluster=rowcol_cluster[0], col_cluster=rowcol_cluster[1])
        plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        if transpose:
            cg.ax_heatmap.set_xlabel(y_label, fontsize=1.5*size[0])
            cg.ax_heatmap.set_ylabel(x_label, fontsize=1.5*size[0])
        else:
            cg.ax_heatmap.set_xlabel(x_label, fontsize=1.5*size[0])
            cg.ax_heatmap.set_ylabel(y_label, fontsize=1.5*size[0])
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches="tight")
        plt.close()


    def plot_heatmap(self, mat, xtick_labels=[], ytick_labels=[], x_label="", y_label="", cbar_label="", plot_numbers=True, transpose=True, size=(12, 12), path=None):
        ma = (np.transpose(mat) if transpose else mat)
        plt.figure(figsize=size)
        if transpose:
            tmp = xtick_labels
            xtick_labels = ytick_labels
            ytick_labels = tmp
            tmp = x_label
            x_label = y_label
            y_label = tmp
        plt.xlabel(x_label, fontsize=1.5*size[0])
        plt.ylabel(y_label, fontsize=1.5*size[0])
        plt.xticks(np.arange(len(xtick_labels)), xtick_labels, rotation=0, fontsize=0.9*size[0])
        plt.yticks(np.flip(np.arange(len(ytick_labels))), ytick_labels, rotation=0, fontsize=0.9*size[0])
        if plot_numbers:
            for i in range(ma.shape[0]):
                for j in range(ma.shape[1]):
                    plt.text(
                        j, 
                        mat.shape[0]-1-i, 
                        "%.4f" % (mat[i,j]), 
                        ha="center", 
                        va="center", 
                        fontsize=0.7*size[0]
                    )
        ax = plt.gca()
        im = plt.imshow(ma, extent=[-0.5, ma.shape[1]-0.5, -0.5, ma.shape[0]-0.5])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=1.5*size[1])
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches="tight")
        plt.close()


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


    def plot_density(self, data, color="b", source="", marker="", band_width=1/20, plt_mean=False, plt_median=False, plt_stddev=False, fill=False):
        from sklearn.neighbors import KernelDensity
        from scipy.stats import gaussian_kde
        from statsmodels.nonparametric.kde import KDEUnivariate
        print(data.shape, np.min(data), np.max(data))
        X = np.reshape(data, -1)
        x_range = np.max(X) - np.min(X)
        xs = np.linspace(np.min(X)-x_range/10, np.max(X)+x_range/10, 101, endpoint=True)
        kde = KernelDensity(bandwidth=band_width)
        kde.fit(np.reshape(X, [-1, 1]))
        pdf = np.exp(kde.score_samples(np.reshape(xs, [-1, 1])))
#        pdf = pdf / np.sum(pdf)
        label = "$%s$" % (source.replace(" ", " \\ "))
        lines = plt.plot(xs, pdf, label=label, color=color, marker=marker, markevery=2, markersize=2*self.line_width, linewidth=self.line_width)
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
        self.lims(y_lim=[-0.002*y_max, y_max*1.01])
        self.legend()
        return self.lines


    def legend(self, size=7):
        plt.legend(prop={"size": size})


    def lims(self, x_lim=[None, None], y_lim=[None, None]):
        left, right = plt.xlim()
        if not x_lim[0] is None:
            if x_lim[0] == "min":
                x_lim[0] = left
            plt.xlim(left=x_lim[0])
        if not x_lim[1] is None:
            if x_lim[1] == "max":
                x_lim[1] = right
            plt.xlim(right=x_lim[1])
        bottom, top = plt.ylim()
        if not y_lim[0] is None:
            if y_lim[0] == "min":
                y_lim[0] = bottom
            plt.ylim(bottom=y_lim[0])
        if not y_lim[1] is None:
            if y_lim[1] == "max":
                y_lim[1] = top
            plt.ylim(top=y_lim[1])
        return left, right, bottom, top


    def ticks(self, xticks=[None, None], yticks=[None, None], xtick_kwargs={}, ytick_kwargs={}):
        xtick_indices, xtick_labels = plt.xticks(**xtick_kwargs)
        if not xticks[1] is None:
            if xticks[0] is None:
                xticks[0] = np.arange(len(xticks[1]))
            plt.xticks(xticks[0], xticks[1])
        ytick_indices, ytick_labels = plt.yticks(**ytick_kwargs)
        if not yticks[1] is None:
            if yticks[0] is None:
                yticks[0] = np.arange(len(yticks[1]))
            plt.yticks(yticks[0], yticks[1])
        return xtick_indices, xtick_labels, ytick_indices, ytick_labels


    def labels(self, xlabel=None, ylabel=None, xlabel_kwargs={}, ylabel_kwargs={}):
        if not xlabel is None:
            plt.xlabel(xlabel, **xlabel_kwargs)
        if not ylabel is None:
            plt.ylabel(ylabel, **ylabel_kwargs)


    def save_figure(self, path, dpi=200):
        plt.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close()


    def show_figure(self):
        plt.show()
        plt.close()


    def plot_learning_curve(self, train, valid, test, path):
        plt.plot(train, color="k", label="Training", linewidth=self.line_width)
        plt.plot(valid, color="r", label="Validation", linewidth=self.line_width)
        plt.plot(test, color="g", label="Testing", linewidth=self.line_width)
        ymax = max(max(train), max(valid), max(test))
        plt.ylim(bottom=0, top=0.75*ymax)
        plt.ylabel("Mean Squared Error")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(path)
        plt.close()


    def _plot_spatiotemporal(self, name, var, partition):
        if name == "original":
            spatiotemporal = var.get(name, partition)
            temporal_labels = var.get(name+"_temporal_labels", partition)
            spatial_labels = var.get(name+"_spatial_labels", partition)
            feature_labels = list(var.get("feature_index_map").keys())
        elif name == "reduced":
            channel_indices = var.get_reduced_channel_indices(
                var.get("reduced_n_temporal", partition), 
                1
            )
            response_indices = var.get("response_indices")
            spatiotemporal = var.get(name, partition)[channel_indices][:,:,response_indices]
            temporal_labels = var.get(name+"_temporal_labels", partition)[channel_indices]
            spatial_labels = var.get("original_spatial_labels", partition)
            feature_labels = list(var.get("feature_index_map").keys())
            feature_labels = var.get("response_features")
        elif "windowed" in name:
            windowed = var.get(name, partition)
            if "input" in name:
                windowed_temporal_labels = var.get("input_windowed_temporal_labels", partition)
                feature_labels = var.get("predictor_features")
                window_n_temporal = var.get("n_temporal_in")
            elif "output" in name:
                windowed_temporal_labels = var.get("output_windowed_temporal_labels", partition)
                feature_labels = var.get("response_features")
                window_n_temporal = var.get("n_temporal_out")
            else:
                raise ValueError()
            contiguous_window_indices = var.get_contiguous_window_indices(
                window_n_temporal, 
                var.get("n_windows", partition), 
                1
            )
            spatiotemporal = np.reshape(
                windowed[contiguous_window_indices,:,:,:], 
                (-1, windowed.shape[2], windowed.shape[3])
            )
            temporal_labels = np.reshape(
                windowed_temporal_labels[contiguous_window_indices,:], 
                (-1)
            )
            spatial_labels = var.get("original_spatial_labels", partition)
        path = var.get("plot_dir") + os.sep + "Spatiotemporal_Partition[%s]_Feature[%s]_Subbasins[%s]_Form[%s]_Source[%s].png" % (
            partition,
            "%s",
            ",".join(spatial_labels),
            name,
            var.get("dataset", partition)
        )
        if len(spatial_labels) > 100:
            return
        print("NAME =", name, ",", "Partition =", partition)
        self.plot_spatiotemporal(spatiotemporal, temporal_labels, spatial_labels, feature_labels, path)


    def plot_spatiotemporal(self, spatiotemporal, temporal_labels, spatial_labels, feature_labels, path=""):
        n_temporal = spatiotemporal.shape[0]
        n_spatial = spatiotemporal.shape[1]
        n_features = spatiotemporal.shape[2]
        for i in range(n_features):
            for j in range(n_spatial):
                plt.plot(spatiotemporal[:,j,i], label=spatial_labels[j], linewidth=self.line_width)
                y_min, y_max = np.min(spatiotemporal[:,j,i]), np.max(spatiotemporal[:,j,i])
                print("SUBBASIN =", spatial_labels[j], "FEATURE =", feature_labels[i], "min-max = [%.3f, %.3f]" % (y_min, y_max))
            temporal_indices = np.linspace(0, n_temporal-1, 8, dtype=np.int)
            plt.xticks(temporal_indices, temporal_labels[temporal_indices], rotation=45)
            y_min, y_max = np.min(spatiotemporal[:,:,i]), np.max(spatiotemporal[:,:,i])
            n_decimals = max(3-len(str(int(y_max))), 0)
            dtype, y_min, y_max = ([int, int(y_min), int(y_max)] if n_decimals == 0 else [float, y_min, y_max])
            ytick_locs = np.around(np.linspace(y_min, y_max, 7, endpoint=True, dtype=dtype), n_decimals)
            self.ticks(yticks=[ytick_labels, ytick_labels])
            plt.legend()
            if path == "":
                plt.show()
            else:
                feature_path = path % (feature_labels[i])
                self.save_figure(feature_path)
        

    def plot_reduced_historical(self, reduced_historical, reduced_historical_dates, subbasin_labels, feature, modifications="", plot_range=[0.0, 1.0]):
        n_subbasins = reduced_historical.shape[1]
        y_min = sys.float_info.max
        y_max = sys.float_info.min
        subbasin_indices = subbasin_labels - 1
        alphas = [0.65, 0.65]
        overlap_color = "brown"
        overlap_color = "green"
        overlap_color = "purple"
        if overlap_color == "brown":
            # Color pairs that create brown when combined
            colors = ["purple", "yellow"]
            colors = ["red", "green"]
            colors = ["blue", "orange"]
        if overlap_color == "green":
            # Color pairs that create green when combined
            colors = ["blue", "yellow"]
        if overlap_color == "purple":
            # Color pairs that create purple when combined
            colors = ["blue", "red"]
#            alphas = [1, 1]
        background_color = "snow"
        background_color = "whitesmoke"
        background_color = "white"
        plt.rcParams["axes.facecolor"] = background_color
        start = round(plot_range[0] * reduced_historical.shape[0])
        end = round(plot_range[1] * reduced_historical.shape[0])
        for s in range(subbasin_labels.shape[0]):
            subbasin_idx = subbasin_indices[s]
            subbasin_label = subbasin_labels[s]
            color = colors[s]
            alpha = alphas[s]
            plt.plot(reduced_historical[start:end,subbasin_idx], color=color, label="Subbasin %d" % (subbasin_label), linewidth=.65, alpha=alpha)
#            plt.plot(reduced_historical[start:end,subbasin_idx], label="Subbasin %d" % (subbasin_label), linewidth=.65, alpha=alpha)
            y_min = min(y_min, np.min(reduced_historical[:,subbasin_idx]))
            y_max = max(y_max, np.max(reduced_historical[:,subbasin_idx]))
        y_range = y_max - y_min
        plt.ylim(y_min-0.05*y_range, y_max+0.05*y_range)
        print("reduced_historical_dates =", reduced_historical_dates.shape, "=")
        print(reduced_historical_dates)
        print("start,end = %d,%d" % (start,end))
        start = round(plot_range[0] * reduced_historical_dates.shape[0])
        end = round(plot_range[1] * reduced_historical_dates.shape[0])
        indices = np.linspace(start, end-1, 10, dtype=np.int)
        plt.xticks(indices-start, reduced_historical_dates[indices], rotation=45, fontsize=8)
        plt.xlabel("Time (%d:1 Reduction)" % (timestep_reduction_factor))
        mods = re.sub("difference\d", "difference", modifications)
        modification_ylabel_map = {
            "": "%s (%s)",
            "min-max": "Min-Max Normalized %s",
            "z-score": "%s Z-score",
            "tanh": "Tanh Normalized %s",
            "difference": "Differenced %s",
            "z-score,difference": "Differenced %s Z-score",
            "log": "Log Transformed %s",
            "cube-root": "Cube-Root Transformed %s"
        }
        ylabel = modification_ylabel_map[mods] % (feature_fullname_map[feature], feature_SIunit_map[feature])
        plt.ylabel(ylabel)
        plt.legend(prop={"size":5})
        filename = "ReducedHistorical_Subbasins[%s]_Feature[%s]_DatesInterval[%s,%s]_TimeReduction[%s,%d,%d]_Modifications[%s]_Range[%.3f,%.3f].png" % (
            ",".join(map(str, subbasin_labels)),
            feature,
            train_dates_interval[0],
            train_dates_interval[1],
            timestep_reduction_method,
            timestep_reduction_factor,
            timestep_reduction_stride,
            modifications,
            plot_range[0],
            plot_range[1]
        )
        path = plot_dir + os.sep + filename
        plt.savefig(path, bbox_inches="tight", dpi=200)
        plt.close()


    def plot_subbasin_correlation(self, feature_corrcoefs, feature, modifications, vmin_vmax=[-1.0,1.0], plot_range=[0.0, 1.0]):
        n_subbasins = int(feature_corrcoefs.shape[1] * 1)
        start = round(plot_range[0] * n_subbasins) 
        end = round(plot_range[1] * n_subbasins) 
        corrcoefs = feature_corrcoefs
        vmin = vmin_vmax[0]
        vmax = vmin_vmax[1]
        cmap = "hot"
        cmap = "seismic"
        cmap = "coolwarm"
        cmap = "inferno"
        cmap = "afmhot"
        im = plt.imshow(corrcoefs[start:end,:], cmap=cmap, aspect="auto", interpolation=None, vmin=vmin, vmax=vmax, origin="lower")
        xtick_labels = np.linspace(1, n_subbasins, 10, dtype=np.int16)
        xtick_indices = xtick_labels - 1
        ytick_labels = np.linspace(start+1, end, 10, dtype=np.int16)
        ytick_indices = ytick_labels - 1 - start
        plt.xticks(xtick_indices, xtick_labels, fontsize=5, rotation=45)
        plt.yticks(ytick_indices, ytick_labels, fontsize=5)
        plt.ylabel("Subbasin ID", fontsize=10)
        plt.xlabel("Subbasin ID", fontsize=10)
        cbar = plt.colorbar(im)
        cbar_label = "Pearson Correlation"
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=10)
        filename = "SubbasinSimilarityHeatmap_Feature[%s]_Modifications[%s]_Metric[%s].png" % (
            feature,
            modifications,
            "PearsonCorrelation"
        )
        path = plot_dir + os.sep + filename
        plt.savefig(path, bbox_inches="tight", dpi=200)
        plt.close()


    def plot_subbasin_dynamictimewarpings(self, subbasin_dtws, feature, modifications, plot_range=[0.0, 1.0]):
        n_subbasins = int(subbasin_dtws.shape[0] * 1)
        start = round(plot_range[0] * n_subbasins) 
        end = round(plot_range[1] * n_subbasins) 
        vmin = np.min(subbasin_dtws)
        vmax = np.max(subbasin_dtws)
#        vmin = 0.0
#        vmin = -1.0
#        vmax = 1.0
        cmap = "hot"
        cmap = "seismic"
        cmap = "coolwarm"
        cmap = "inferno"
        cmap = "afmhot"
        im = plt.imshow(subbasin_dtws[start:end,:], cmap=cmap, aspect="auto", interpolation=None, vmin=vmin, vmax=vmax, origin="lower")
        xtick_labels = np.linspace(1, n_subbasins, 10, dtype=np.int16)
        xtick_indices = xtick_labels - 1
        ytick_labels = np.linspace(start+1, end, 10, dtype=np.int16)
        ytick_indices = ytick_labels - 1 - start
        plt.xticks(xtick_indices, xtick_labels, fontsize=5, rotation=45)
        plt.yticks(ytick_indices, ytick_labels, fontsize=5)
        plt.ylabel("Subbasin ID", fontsize=10)
        plt.xlabel("Subbasin ID", fontsize=10)
        cbar = plt.colorbar(im)
        cbar_label = "Normalized Dynamic Time Warping Distance"
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=10)
        filename = "SubbasinDistanceHeatmap_Feature[%s]_Modifications[%s]_Metric[%s].png" % (
            feature,
            modifications,
            "DynamicTimeWarping"
        )
        path = plot_dir + os.sep + filename
        plt.savefig(path, bbox_inches="tight", dpi=200)
        plt.close()


    def plot_subbasin_mean_similarities(self, feature_mean_similarities, feature):
        n_daysofyear = feature_mean_similarities.shape[0]
        n_subbasins = int(feature_mean_similarities.shape[1] * 1)
        n_features = feature_mean_similarities.shape[2]
        xtick_labels = month_labels
        idx = feature_idx_map[feature]
        cmap = "seismic"
        cmap = "hot"
        print("Min Mean Similarity =", np.min(feature_mean_similarities[:,:n_subbasins,idx]))
        print("Max Mean Similarity =", np.max(feature_mean_similarities[:,:n_subbasins,idx]))
        ytick_labels = np.linspace(1, n_subbasins, 10, dtype=np.int16)
        ytick_indices = ytick_labels - 1
        mean_similarities = feature_mean_similarities[:,:n_subbasins,idx]
        # Plot Means
        mean_similarities = np.swapaxes(mean_similarities, 0, 1)
        min_mean = np.min(mean_similarities)
        max_mean = np.max(mean_similarities)
        im = plt.imshow(mean_similarities, cmap=cmap, vmin=min_mean, vmax=max_mean, aspect="auto", interpolation=None, origin="lower")
        plt.xticks(xtick_indices, xtick_labels, fontsize=5, rotation=45)
        plt.yticks(ytick_indices, ytick_labels, fontsize=5)
        plt.xlabel("Subbasin ID", fontsize=10)
        plt.ylabel("Month", fontsize=10)
        cbar = plt.colorbar(im)
        cbar_label = "Mean" + " " + feature_fullname_map[feature] + " " + "MSE"
        cbar_label = "RMSE"
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=10)
        filename = "SubbasinSimilarityHeatmap_Feature[%s]_Metric[%s].png" % (feature, "Mean-RMSE")
        path = plot_dir + os.sep + filename
        plt.savefig(path, bbox_inches="tight", dpi=200)
        plt.close()


    def plot_subbasin_coefvar_similarities(self, feature_coefvar_similarities, feature):
        n_daysofyear = feature_coefvar_similarities.shape[0]
        n_subbasins = int(feature_coefvar_similarities.shape[1] * 1)
        n_features = feature_coefvar_similarities.shape[2]
        xtick_labels = month_labels
        idx = feature_idx_map[feature]
        cmap = "seismic"
        cmap = "hot"
        print("Min Coefficient of Variation Similarity =", np.min(feature_similarities[:,:n_subbasins,idx]))
        print("Max Coefficient of Variation Similarity =", np.max(feature_similarities[:,:n_subbasins,idx]))
        ytick_labels = np.linspace(1, n_subbasins, 10, dtype=np.int16)
        ytick_indices = ytick_labels - 1
        coefvar_similarities = feature_coefvar_similarities[:,:n_subbasins,idx]
        # Plot Coefficients of Variation
        coefvar_similarities = np.swapaxes(coefvar_similarities, 0, 1)
        min_coefvar = np.min(coefvar_similarities)
        max_coefvar = np.max(coefvar_similarities)
        im = plt.imshow(coefvar_similarities, cmap=cmap, vmin=min_coefvar, vmax=max_coefvar, aspect="auto", interpolation=None, origin="lower")
        plt.yticks(ytick_indices, ytick_labels, fontsize=5)
        plt.ylabel("Subbasin ID", fontsize=10)
        plt.xlabel("Subbasin ID", fontsize=10)
        plt.xticks(xtick_indices, xtick_labels, fontsize=5, rotation=45)
        cbar = plt.colorbar(im)
        cbar_label = feature_fullname_map[feature] + " " + "Coefficient of Variation $(\\frac{\sigma}{\mu})$ MSE"
        cbar_label = "RMSE"
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=10)
        filename = "SubbasinSimilarityHeatmap_Feature[%s]_Metric[%s].png" % (feature, "CoefficientOfVariation-RMSE")
        path = plot_dir + os.sep + filename
        plt.savefig(path, bbox_inches="tight", dpi=200)
        plt.close()


    def plot_subbasin_seasonal_PDFs(self, PDFs, ranges, subbasin_idx, feature):
        feature_idx = feature_idx_map[feature]
        feature_min = ranges[0,subbasin_idx,feature_idx]
        feature_max = ranges[1,subbasin_idx,feature_idx]
        ytick_labels = np.round(np.linspace(feature_min, feature_max, 10), 1)
        ytick_indices = np.linspace(0, PDFs.shape[0]-1, 10)
        n_daysofyear = PDFs.shape[1]
        xtick_labels = month_labels
        xtick_indices = np.linspace(1, n_daysofyear, len(xtick_labels))
        # Plot PDFs
        print(PDFs[:,:,subbasin_idx,feature_idx])
        cmap = "afmhot"
        vmin = np.min(PDFs[:,:,subbasin_idx,feature_idx])
        vmax = np.max(PDFs[:,:,subbasin_idx,feature_idx])
        im = plt.imshow(PDFs[:,:,subbasin_idx,feature_idx], cmap=cmap, aspect="auto", interpolation="gaussian", vmin=vmin, vmax=vmax, origin="lower")
        plt.yticks(ytick_indices, ytick_labels, fontsize=5)
        y_label = feature_fullname_map[feature] + " " + feature_SIunit_map[feature]
        plt.ylabel(y_label, fontsize=10)
        plt.xticks(xtick_indices, xtick_labels, fontsize=5, rotation=45)
        cbar = plt.colorbar(im)
        cbar_label = "Probability"
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=10)
        filename = "SubbasinSeasonalityHeatmap_Subbasin[%d]_Feature[%s]_Metric[%s].png" % (subbasin_idx+1, feature, "PDF")
        path = plot_dir + os.sep + filename
        plt.savefig(path, bbox_inches="tight", dpi=200)
        plt.close()


    def plot_subbasin_seasonal_means(self, feature_means, feature):
        n_daysofyear = feature_means.shape[0]
        n_subbasins = int(feature_means.shape[1] * 1)
        n_features = feature_means.shape[2]
        xtick_labels = month_labels
        idx = feature_idx_map[feature]
        print("Min Mean =", np.min(feature_means[:,:n_subbasins,idx]))
        print("Max Mean =", np.max(feature_means[:,:n_subbasins,idx]))
        ytick_labels = np.linspace(1, n_subbasins, 10, dtype=np.int16)
        ytick_indices = ytick_labels - 1
        xtick_indices = np.linspace(1, n_daysofyear, len(xtick_labels))
        means = feature_means[:,:n_subbasins,idx]
        # Plot Means
        means = np.swapaxes(means, 0, 1)
        min_mean = np.min(means)
        max_mean = np.max(means)
        im = plt.imshow(means, cmap="hot", aspect="auto", interpolation=None, vmin=min_mean, vmax=max_mean, origin="lower")
        plt.yticks(ytick_indices, ytick_labels, fontsize=5)
        plt.ylabel("Subbasin ID", fontsize=10)
        plt.xticks(xtick_indices, xtick_labels, fontsize=5, rotation=45)
        cbar = plt.colorbar(im)
        cbar_label = "Mean" + " " + feature_fullname_map[feature] + " " + feature_SIunit_map[feature]
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=10)
        filename = "SubbasinSeasonalityHeatmap_Feature[%s]_Metric[%s].png" % (feature, "Mean")
        path = plot_dir + os.sep + filename
        plt.savefig(path, bbox_inches="tight", dpi=200)
        plt.close()


    def plot_subbasin_seasonal_coefvars(self, feature_coefvars, feature):
        n_daysofyear = feature_coefvars.shape[0]
        n_subbasins = int(feature_coefvars.shape[1] * 1)
        n_features = feature_coefvars.shape[2]
        xtick_labels = month_labels
        idx = feature_idx_map[feature]
        print("Min Coefficient of Variation =", np.min(feature_coefvars[:,:n_subbasins,idx]))
        print("Max Coefficient of Variation =", np.max(feature_coefvars[:,:n_subbasins,idx]))
        ytick_labels = np.linspace(1, n_subbasins, 10, dtype=np.int16)
        ytick_indices = ytick_labels - 1
        xtick_indices = np.linspace(1, n_daysofyear, len(xtick_labels))
        coefvars = feature_coefvars[:,:n_subbasins,idx]
        # Plot Coefficients of Variation
        coefvars = np.swapaxes(coefvars, 0, 1)
        min_cov = np.min(coefvars)
        max_cov = np.max(coefvars)
        im = plt.imshow(coefvars, cmap="hot", aspect="auto", interpolation=None, vmin=min_cov, vmax=max_cov, origin="lower")
        plt.yticks(ytick_indices, ytick_labels, fontsize=5)
        plt.ylabel("Subbasin ID", fontsize=10)
        plt.xticks(xtick_indices, xtick_labels, fontsize=5, rotation=45)
        cbar = plt.colorbar(im)
        cbar_label = feature_fullname_map[feature] + " " + "Coefficient of Variation $(\\frac{\sigma}{\mu})$"
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=10)
        filename = "SubbasinSeasonalityHeatmap_Feature[%s]_Metric[%s].png" % (feature, "CoefficientOfVariation")
        path = plot_dir + os.sep + filename
        plt.savefig(path, bbox_inches="tight", dpi=200)
        plt.close()


    def plot_miscellaneous(self):

        plot_all = True
        plot_individual = True
        plot_regions = True

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        print("Colors =", colors)
        all_subbasins = np.arange(1, n_subbasins)
        all_subbasins = [1, 2, 7, 9, 19, 52]
        std_alphas = {1:0.5, 2:0.25, 3:0.125}
#std_alphas = {1:0.5, 2:0.5, 3:0.5}
        days_to_month = [1, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
        days_to_month = np.array(days_to_month, dtype=np.int16)
        xtick_indices = days_to_month - 1
        xtick_labels = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "January"]
        xtick_labels = np.array(xtick_labels, dtype=object)
        n_stds = 1

        n_subbasins = len(all_subbasins)
        if plot_all:
            for j in range(2, n_subbasins+1):
                subbasins = all_subbasins[:j]
                c = 0
                for subbasin in subbasins:
                    s = subbasin - 1
                    means = feature_means[:,s,streamflow_idx]
                    stds = feature_stds[:,s,streamflow_idx]
                    plt.plot(means, linestyle="-", color=colors[c], label="Subbasin %d Streamflow Mean"%(subbasin))
                    if plot_regions:
                        for i in range(n_stds,0,-1):
                            lower_bounds = means - stds * i
                            upper_bounds = means + stds * i
                            plt.fill_between(np.arange(n_daysofyear), lower_bounds, upper_bounds, alpha=std_alphas[i], color=colors[c])
                    plt.xticks(xtick_indices, xtick_labels, fontsize=5, rotation=45)
                    plt.ylabel("Stream Flow (cm^3/s)")
                    plt.xlabel("Day of Year")
                    plt.legend(prop={"size":5})
                    c += 1
                subbasin_labels = map(str, subbasins)
                plt.savefig("./Test/Subbasins[%s].png"%(",".join(subbasin_labels)), dpi=200)
                plt.close()
        if plot_individual:
            c = 0
            for subbasin in subbasins:
                s = subbasin - 1
                means = feature_means[:,s,streamflow_idx]
                stds = feature_stds[:,s,streamflow_idx]
                plt.plot(np.arange(n_daysofyear), means, linestyle="-", label="Subbasin %d Mean"%(subbasin), color=colors[c])
                if plot_regions:
#                for i in range(n_stds,0,-1):
                    for i in range(1,n_stds+1):
                        lower_bounds = means - stds * i
                        upper_bounds = means + stds * i
                        plt.fill_between(np.arange(n_daysofyear), lower_bounds, upper_bounds, alpha=std_alphas[i], label="Subbasin %d Standard Deviation %d"%(subbasin,i), color=colors[c])
                plt.xticks(xtick_indices, xtick_labels, fontsize=5, rotation=45)
                plt.ylabel("Stream Flow (cm^3/s)")
                plt.xlabel("Day of Year")
                plt.legend(prop={"size":5})
                plt.savefig("./Test/Subbasin[%d].png"%(subbasin), dpi=200)
                plt.close()


    def plot_correlation_error_pairs(self, correlations, errors, partition, wabash_data, args):
        n_responses = wabash_data.get("n_responses")
        n_subbasins = wabash_data.get("n_subbasins", partition=partition)
        subbasin_labels = wabash_data.get("subbasin_labels", partition=partition)
        subbasin_indices = wabash_data.get("subbasin_indices", partition=partition)
        response_features = wabash_data.get("response_features")
        historical_feature_index_map = wabash_data.get("historical_feature_index_map")
        train_subbasin_indices = wabash_data.get("subbasin_indices", partition="train")
        train_subbasin_idx = train_subbasin_indices[0]
        for o in range(n_responses):
            feature = response_features[o]
            corrs = correlations[train_subbasin_idx,subbasin_indices,o]
            errs = errors[:,o]
            plt.scatter(corrs, errs)
            common_subbasin_indices = np.intersect1d(train_subbasin_indices, subbasin_indices)
            if common_subbasin_indices.shape[0] > 0:
                corrs = correlations[train_subbasin_idx,common_subbasin_indices,o]
                errs = errors[common_subbasin_indices,o]
                plt.scatter(corrs, errs, color="r")
            xlabel = "Pearson Correlation"
            plt.xlabel(xlabel)
            ylabel = feature_fullname_map[feature] + " " + "NRMSE"
            plt.ylabel(ylabel)
#            plt.xlim(-1, 1)
            filename = "SubbasinCorrelationInferenceErrorScatter_Partition[%s]_Feature[%s].png" % (partition, feature)
            path = plot_dir + os.sep + filename
            plt.savefig(path, bbox_inches="tight", dpi=200)
            plt.close()


    # Precondition: predictions and groundtruth have the following shapes
    #   predictions.shape=[n_windows, n_temporal_in, n_spatial, n_responses]
    #   groundtruth.shape=[n_windows, n_temporal_in, n_spatial, n_responses]
    def plot_model_fit(self, prediction, spatmp, partition, plt_range, plt_dir, var):
        exec_var, plt_var, proc_var = var.get("execution"), var.get("plotting"), var.get("processing")
        dataset, options = exec_var.get("dataset", partition), plt_var.get("options")
        n_temporal_in = spatmp.get("windowed").get("n_temporal_in")
        n_temporal_out = spatmp.get("windowed").get("n_temporal_out")
        n_predictors = spatmp.get("misc").get("n_predictors")
        n_responses = spatmp.get("misc").get("n_responses")
        response_features = spatmp.get("misc").get("response_features")
        response_indices = spatmp.get("misc").get("response_indices")
        n_spatial = spatmp.get("original").get("n_spatial", partition)
        spatial_labels = spatmp.get("original").get("spatial_labels", partition)
        spatial_indices = spatmp.get("original").get("spatial_indices", partition)
        mins = spatmp.filter_axes(
            spatmp.get("metrics").get("minimums"),
            [1, 2],
            [spatial_indices, response_indices]
        )
        maxes = spatmp.filter_axes(
            spatmp.get("metrics").get("maximums"),
            [1, 2],
            [spatial_indices, response_indices]
        )
        meds = spatmp.filter_axes(
            spatmp.get("metrics").get("medians"),
            [1, 2],
            [spatial_indices, response_indices]
        )
        means = spatmp.filter_axes(
            spatmp.get("metrics").get("means"),
            [1, 2],
            [spatial_indices, response_indices]
        )
        stddevs = spatmp.filter_axes(
            spatmp.get("metrics").get("standard_deviations"),
            [1, 2],
            [spatial_indices, response_indices]
        )
        transformation_resolution = proc_var.get("transformation_resolution")
        original_temporal_labels = spatmp.get("original").get("temporal_labels", partition)
        reduced_features = spatmp.get("reduced").get("features", partition)
        reduced_temporal_labels = spatmp.get("reduced").get("temporal_labels", partition)
        reduced_temporal_indices = spatmp.get("reduced").get("temporal_indices", partition)
        reduced_n_temporal = spatmp.get("reduced").get("n_temporal", partition)
        windowed_output_features = spatmp.get("windowed").get("output_features", partition)
        windowed_output_temporal_labels = spatmp.get("windowed").get(
            "output_temporal_labels", 
            partition
        )
        n_windows = spatmp.get("windowed").get("n_windows", partition)
        temporal_channel_indices = spatmp.get_reduced_channel_indices(reduced_n_temporal, 1)
        contiguous_window_indices = spatmp.get_contiguous_window_indices(
            n_temporal_out,
            n_windows,
            1
        )
        prediction = prediction[contiguous_window_indices,:,:,:]
        temporal_channel_idx = 0
        gt = spatmp.filter_axes(
            reduced_features[temporal_channel_indices],
            [2],
            [spatmp.get("misc").get("response_indices")]
        )
        windowed_output_temporal_labels = windowed_output_temporal_labels[contiguous_window_indices]
        temporal_labels = reduced_temporal_labels[temporal_channel_idx,:]
        dayofyear_indices = util.convert_dates_to_daysofyear(temporal_labels) - 1
        windowed_output_dayofyear_indices = util.convert_dates_to_daysofyear(windowed_output_temporal_labels).reshape(-1) - 1
        event_data = np.zeros((gt.shape[0],n_spatial,n_responses,2))
        cache_plot_data = False
        cache_plot_data = True
        # Set colors
        alpha_min = 0.40
        alpha_max = 0.80
        alpha_interval = [alpha_min, alpha_max]
        if n_spatial > 1:
            alphas = np.linspace(alpha_interval[0], alpha_interval[1], n_spatial)
            alphas = np.flip(alphas)
        else:
            alphas = np.array([alpha_max])
        cmap = plt.get_cmap("Greys")
        groundtruth_colors = [cmap(1.25*i) for i in alphas]
        cmap = plt.get_cmap("Reds")
        prediction_colors = [cmap(i) for i in alphas]
        cmap = plt.get_cmap("Blues")
        median_colors = [cmap(i) for i in alphas]
        cmap = plt.get_cmap("Blues")
        mean_colors = [cmap(i) for i in alphas]
        cmap = plt.get_cmap("Blues")
        std_colors = [cmap(0.25*i) for i in alphas]
        cmap = plt.get_cmap("Reds")
        extreme_colors = [cmap(1.0*i) for i in alphas]
        feature_plot_order_map = {"SWmm": "ascending", "FLOW_OUTcms": "descending"}
        for o in range(n_responses):
            y_min, y_max = sys.float_info.max, sys.float_info.min
            response_prediction = prediction[:,:,:,o]
            response_gt = gt[:,:,o]
            spatial_indices = np.arange(n_spatial)
            if feature_plot_order_map[response_features[o]] == "descending":
                spatial_indices = np.flip(spatial_indices)
            n_spa_plotted, n_spa_per_plt = 0, plt_var.get("n_spatial_per_plot")
            for s in spatial_indices:
                spatial_response_prediction = np.reshape(response_prediction[:,:,s], (-1))
                spatial_response_gt = np.reshape(response_gt[:,s], (-1))
                if "confusion" in options:
                    self._plot_zscore_confusion(
                        spatial_response_prediction, 
                        spatial_response_gt[n_temporal_in:],
                        means[dayofyear_indices[n_temporal_in:],s,o], 
                        stddevs[dayofyear_indices[n_temporal_in:],s,o]
                    )
                    path = plt_dir + os.sep + "Confusion_Partition[%s]_Subbasins[%s]_Response[%s].png" % (
                        partition, 
                        ",".join(spatial_labels[s:s+1]), 
                        response_features[o]
                    )
                    self.save_figure(path)
                if "groundtruth" in options:
                    self._plot_groundtruth(
                        spatial_response_gt, 
                        dataset, 
                        n_spatial, 
                        spatial_labels[s], 
                        n_spa_per_plt,
                        groundtruth_colors[n_spa_plotted%n_spa_per_plt]
                    )
                if "prediction" in options:
                    self._plot_prediction(
                        spatial_response_prediction, 
                        n_temporal_in, 
                        n_temporal_out, 
                        n_spatial, 
                        spatial_labels[s], 
                        n_spa_per_plt,
                        prediction_colors[n_spa_plotted%n_spa_per_plt]
                    )
                if "groundtruth_extremes" in options:
                    self._plot_groundtruth_extremes(
                        spatial_response_gt, 
                        means[dayofyear_indices,s,o], 
                        stddevs[dayofyear_indices,s,o], 
                        dataset, 
                        n_spatial, 
                        spatial_labels[s], 
                        n_spa_per_plt,
                        plt_range
                    )
                if "prediction_extremes" in options:
                    self._plot_prediction_extremes(
                        spatial_response_prediction, 
                        means[windowed_output_dayofyear_indices,s,o], 
                        stddevs[windowed_output_dayofyear_indices,s,o], 
                        n_temporal_in, 
                        n_temporal_out, 
                        n_spatial, 
                        spatial_labels[s], 
                        n_spa_per_plt,
                        plt_range
                    )
                if len(re.findall(".*median", ",".join(options))) > 0:
                    if "median" in options:
                        spatial_response_meds = meds[dayofyear_indices,s,o]
                    elif "temporal_median" in options:
                        spatial_response_meds = np.median(meds[dayofyear_indices,:,o], axis=1)
                    elif "spatial_median" in options:
                        spatial_response_med = np.median(meds[:,s,o], axis=0)
                        spatial_response_meds = np.tile(spatial_response_med, temporal_labels.shape[0])
                    elif "feature_median" in options:
                        spatial_response_med = np.median(meds[:,s,o], axis=(0,1))
                        spatial_response_meds = np.tile(spatial_response_med, temporal_labels.shape[0])
                    label = "Median"
                    if n_spatial > 1 and n_spa_per_plt > 1:
                        label = ("Subbasin %s " % (spatial_labels[s])) + label
                    plt.plot(spatial_response_meds, color=median_colors[s], linestyle="-.", label=label, linewidth=self.line_width)
                    y_min = min(y_min, np.min(spatial_response_meds))
                    y_max = max(y_max, np.max(spatial_response_meds))
                if len(re.findall(".*mean", ",".join(options))) > 0:
                    if "mean" in options:
                        spatial_response_means = means[dayofyear_indices,s,o]
                    elif "temporal_mean" in options:
                        spatial_response_means = np.mean(means[dayofyear_indices,:,o], axis=1)
                    elif "spatial_mean" in options:
                        spatial_response_mean = np.mean(means[:,s,o], axis=0)
                        spatial_response_means = np.tile(spatial_response_mean, temporal_labels.shape[0])
                    elif "feature_mean" in options:
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
                if len(re.findall(".*stddev", ",".join(options))) > 0:
                    if "stddev" in options:
                        spatial_response_stddevs = stddevs[dayofyear_indices,s,o]
                        spatial_response_means = means[dayofyear_indices,s,o]
                    elif "temporal_stddev" in options:
                        spatial_response_stddevs = np.std(stddevs[dayofyear_indices,:,o], axis=1)
                        spatial_response_means = np.mean(means[dayofyear_indices,:,o], axis=1)
                    elif "spatial_stddev" in options:
                        spatial_response_stddev = np.std(stddevs[:,s,o], axis=0)
                        spatial_response_stddevs = np.tile(spatial_response_stddev, temporal_labels.shape[0])
                        spatial_response_mean = np.mean(means[:,s,o], axis=0)
                        spatial_response_means = np.tile(spatial_response_mean, temporal_labels.shape[0])
                    elif "feature_stddev" in options:
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
                y_min = min(y_min, np.min(spatial_response_prediction))
                y_max = max(y_max, np.max(spatial_response_prediction))
                y_min = min(y_min, np.min(spatial_response_gt))
                y_max = max(y_max, np.max(spatial_response_gt))
                """
                start = round(plt_range[0] * (spatial_response_prediction.shape[0] + n_temporal_in))
                end = round(plt_range[1] * (spatial_response_prediction.shape[0] + n_temporal_in))
                val_indices = np.arange(start, end) - n_temporal_in
                loc_indices = np.arange(val_indices.shape[0]) + max(n_temporal_in-start, 0)
                prediction_interval_markers = np.arange(loc_indices[0], loc_indices[-1]+1+1, n_temporal_out)
                # Add at most 50 output markers so its not too crowded
                max_prediction_interval_markers = 50
                if prediction_interval_markers.shape[0] <= max_prediction_interval_markers and "prediction intervals" in options:
                    for i in range(prediction_interval_markers.shape[0]-1):
                        plt.axvline(prediction_interval_markers[i], color="k", linestyle="--", linewidth=self.line_width)
                    i = prediction_interval_markers.shape[0] - 1
                    plt.axvline(prediction_interval_markers[i], color="k", linestyle="--", label="Inference Interval Markers", linewidth=self.line_width)
                """
                n_spa_plotted += 1
                if (n_spa_per_plt > 0 and n_spa_plotted % n_spa_per_plt == 0) or (n_spa_per_plt < 0 and n_spa_plotted == len(spatial_labels)):
                    plt.ylabel(self.feature_ylabel_map[response_features[o]], fontsize=8)
                    self._plot_xticks(temporal_labels, plt_range)
                    self._plot_yticks([y_min, y_max])
                    self._plot_xylim([0, temporal_labels.shape[0]], [y_min, y_max], plt_range)
                    self._plot_legend()
                    self._savefigs(partition, spatial_labels[s:s+1], response_features[o], options, plt_range, plt_dir)
                    y_min, y_max = sys.float_info.max, sys.float_info.min
            # Save all the data so we can reproduce the plots later
            if cache_plot_data:
                filename = "EventData_Partition[%s]_Subbasins[%s]_Response[%s].pkl" % (partition, ",".join(spatial_labels), ",".join(response_features))
                path = plt_dir + os.sep + filename
                util.to_cache(event_data, path)
                filename = "Evaluation_Partition[%s]_Subbasins[%s]_Response[%s]_Datatype[%s].pkl" % (partition, ",".join(spatial_labels), response_features[o], "Groundtruth")
                path = plt_dir + os.sep + filename
                util.to_cache(gt, path)
                filename = "Evaluation_Partition[%s]_Subbasins[%s]_Response[%s]_Datatype[%s].pkl" % (partition, ",".join(spatial_labels), response_features[o], "Prediction")
                path = plt_dir + os.sep + filename
                util.to_cache(prediction, path)
                prediction_interval_markers = np.arange(prediction.shape[0]+1) * n_temporal_out + n_temporal_in
                filename = "Evaluation_Partition[%s]_Datatype[%s]_Range[%.3f,%.3f].pkl" % (partition, "InferenceIntervalMarkers", plt_range[0], plt_range[1])
                path = plt_dir + os.sep + filename
                util.to_cache(prediction_interval_markers, path)


    def _plot_groundtruth(self, gt, dataset, n_spa, spa_label, n_spa_per_plt, color):
        label = self.dataset_legend_map[dataset]
        if n_spa > 1 and n_spa_per_plt > 1:
            label = ("Subbasin %s " % (str(spa_label))) + label
        plt.plot(gt, color=color, linestyle="-", label=label, linewidth=self.gt_line_width)


    def _plot_prediction(self, pred, n_tmp_in, n_tmp_out, n_spa, spa_label, n_spa_per_plt, color):
        label = "Prediction"
        if n_spa > 1 and n_spa_per_plt > 1:
            label = ("Subbasin %s " % (str(spa_label))) + label
        indices = np.arange(n_tmp_in, n_tmp_in + pred.shape[0])
        plt.plot(indices, pred, color=color, linestyle="-", label=label, linewidth=self.pred_line_width)


    def _plot_groundtruth_extremes(self, gt, means, stddevs, dataset, n_spa, spa_label, n_spa_per_plt, plt_range):
        interval_events_map = util.compute_events(gt, means, stddevs)
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
                plt.plot(events, color=color, linestyle="-", marker=marker, label=label, linewidth=2*lw, markersize=0.75*self.marker_size)


    def _plot_prediction_extremes(self, pred, means, stddevs, n_tmp_in, n_tmp_out, n_spa, spa_label, n_spa_per_plt, plt_range):
        interval_events_map = util.compute_events(pred, means, stddevs)
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
                plt.plot(indices, events, color=color, linestyle="-", marker=marker, label=label, linewidth=2*lw, markersize=0.75*self.marker_size)


    def _plot_zscore_confusion(self, preds, gts, means, stddevs):
        confusion = util.compute_zscore_confusion(preds, gts, means, stddevs, normalize=False)
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
        from confusion_matrix_pretty_print import pretty_plot_confusion_matrix
        """
        sns.heatmap(df, annot=True, fmt=".1%", cmap="Blues", cbar=False)
        """
        pretty_plot_confusion_matrix(df, cmap="Blues", pred_val_axis="col")
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


    def _plot_xticks(self, x_labels, plt_range):
        xtick_indices = np.arange(x_labels.shape[0])
        xtick_labels = x_labels[xtick_indices]
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


    def _savefigs(self, partition, spatial_labels, feature, options, plt_range, plt_dir):
        spatials = ",".join(map(str, spatial_labels))
        opts = list(options)
        if "confusion" in opts:
            opts.remove("confusion")
        filename = "Evaluation_Partition[%s]_Subbasins[%s]_Response[%s]_Options[%s]_Range[%.3f,%.3f].png" % (
            partition, spatials, feature, ",".join(opts), plt_range[0], plt_range[1])
        path = plt_dir + os.sep + filename
        plt.savefig(path, bbox_inches="tight", dpi=200)
        filename = "Evaluation_Partition[%s]_Subbasins[%s]_Response[%s]_Options[%s]_Range[%.3f,%.3f].pdf" % (
            partition, spatials, feature, ",".join(opts), plt_range[0], plt_range[1])
        path = plt_dir + os.sep + filename
#        plt.savefig(path, bbox_inches="tight", dpi=200)
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
    fname_path_map = {fname: "Data\\WabashRiverWatershedShapes\\"+fname+".shp" for fname in fnames}
    fname_item_map = {"riv1": "rivers", "subs1": "subbasins"}
    item_shapes_map = {fname_item_map[fname]: shapefile.Reader(path) for fname, path in fname_path_map.items()}
    path = "Plots\\Watershed_Components[%s]_Watershed[%s].png" % (",".join(item_shapes_map.keys()), watershed)
    plt.plot_watershed(item_shapes_map, subbasin_river_map, path, highlight=False, watershed=watershed, river_opts={"color_code": False, "name": False})


def plot_little():
    watershed = "LittleRiver"
    plt = Plotting()
    subbasin_river_map = {}
    fnames = ["basins", "gis_streams"]
#    fnames.remove("gis_streams")
    fname_path_map = {fname: "Data\\LittleRiverWatershedShapes\\"+fname+".shp" for fname in fnames}
    fname_item_map = {"gis_streams": "rivers", "basins": "subbasins"}
    item_shapes_map = {fname_item_map[fname]: shapefile.Reader(path) for fname, path in fname_path_map.items()}
    path = "Plots\\Watershed_Components[%s]_Watershed[%s].png" % (",".join(item_shapes_map.keys()), watershed)
    plt.plot_watershed(item_shapes_map, subbasin_river_map, path, highlight=False, watershed=watershed)


if len(sys.argv) > 1 and sys.argv[0] == "Plotting.py" and sys.argv[1] == "test":
    plot_wabash()
#    plot_little()
