import numpy as np
import scipy
import sklearn
from sklearn import cluster as sk_cluster

import Utility as util


class Clustering:

    def cluster(x, alg="KMeans", n_clusters=3, **kwargs):
        """
        Arguments
        ---------
        x : ndarray with shape=(N, D) for N samples on D variables
        alg : str
        kwargs : dict

        Returns
        -------
        cluster_index : ndarray with shape=(N,)

        """
        if alg == "KMeans":
            clusters = sk_cluster.KMeans(n_clusters, random_state=0, **kwargs).fit(x)
            cluster_index = clusters.labels_
        elif alg == "Agglomerative":
            clusters = sk_cluster.AgglomerativeClustering(n_clusters, **kwargs).fit(x)
            cluster_index = clusters.labels_
        elif alg == "DBSCAN":
            clusters = sk_cluster.DBSCAN(min_samples=5).fit(x)
            cluster_index = clusters.labels_
        else:
            raise NotImplementedError(alg)
        return cluster_index
        

class Probability:

    def compute_histograms(x, bins=10, lims=None):
        """
        Arguments
        ---------
        x : ndarray with shape=(N, D) for N samples on D variables
        bins : int or ndarray with shape(B,)
        lims : None or ndarray with shape=(2,) or shape=(D, 2)

        Returns
        -------
        histograms : ndarray with shape=(D, B) for D variables

        """
        N, D = x.shape
        n_bins = bins
        if not isinstance(bins, int): # bins defines bin edges
            n_bins = len(bins) - 1
        if lims is None:
            lims = np.stack((np.min(x, 0), np.max(x, 0)), -1)
        elif len(lims.shape) == 1:
            lims = np.tile(lims, (x.shape[-1], 1))
        elif not len(lims.shape) == 2 or not lims.shape == (D, 2):
            raise ValueError(lims.shape)
        histograms = np.zeros((D, n_bins))
        for i in range(D):
            histograms[i,:], _, _ = scipy.stats.binned_statistic(x[:,i], x[:,i], "count", bins, lims[i,:])
        return histograms

    def transform(x, rep="histogram", **kwargs):
        if rep == "histogram":
            bins = kwargs.get("bins", 12)
            lims = kwargs.get("lims", None)
            x = Probability.compute_histograms(x, bins, lims)
        else:
            raise NotImplementedError(rep)
        return x


class Decomposition:

    def reduce(x, dim=2, alg="TSNE", **kwargs):
        if alg == "PCA":
            x = sklearn.decomposition.PCA(dim, random_state=0).fit_transform(x)
        elif alg == "TSNE":
            x = sklearn.manifold.TSNE(dim, perplexity=min(30.0, x.shape[0]-1), random_state=0).fit_transform(x)
        else:
            raise NotImplementedError(alg)
        return x
