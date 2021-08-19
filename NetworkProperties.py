import networkx as nx
import numpy as np
from matplotlib.ticker import FormatStrFormatter


# Input: A graph 
# Output: find degrees and plot their distribution
def Degree_Distribution(G):
    degree = G.degree()
    degree = [ deg for (v,deg) in degree ]
    return degree


# Input: A graph 
# Find the sizes of all connected components and plot the distribution
def CC_Distribution(G):
    cc_sorted = sorted(nx.connected_components(G), key=len, reverse=True)
    # print statistics of the top 5 components (if exist)
    topcc = min(len(cc_sorted), 5)
    topcc = len(cc_sorted)
    for i in  range(topcc):
        cc = cc_sorted[i]
        cc_graph = G.subgraph(cc)
        n = cc_graph.number_of_nodes()
        m = cc_graph.number_of_edges()
        n_percent = (n/G.number_of_nodes()) * 100
        print("Largest component #", i+1)
        print("Number of vertices:", n, " (", n_percent, ")", "\nNumber of edges: ", m, "\n")
    return [len(c) for c in cc_sorted]


# Input: A graph 
# Find the local clustering coefficient of all vertices and plot distribution
def Clustering_Analysis(G):
    clust = nx.clustering(G)
    local_clust_coefficient = [ v for v in clust.values() ]
    avg_clust_coefficient = sum(local_clust_coefficient)/G.number_of_nodes()
    print("Average clustering coefficient: ", avg_clust_coefficient)
    #plot the distribution of clustering coefficient
    return local_clust_coefficient


# Input: A graph 
# Find shortest paths in the largest 5 componets and plot distribution
def ShortestPaths_Analysis(G):
    cc_sorted = sorted(nx.connected_components(G), key=len)
    cc = cc_sorted[-1]
    cc_graph = G.subgraph(cc)
    if(len(cc)>30000):
        print("This component is too large. Using ten single-source shortest paths.")
        cc = list(cc)
        cc_graph = G.subgraph(cc)
        shortest_path_lens = []
        vertex_selection = np.random.choice(G.number_of_nodes(), size=10, replace=False)
        for i in vertex_selection:
            length = nx.single_source_shortest_path_length(cc_graph, cc[i]) 
            shortest_path_lens += [ v for v in length.values() ]
    else:
        all_shortest_path_dict = dict(nx.all_pairs_shortest_path_length(cc_graph))
        shortest_path_lens = []
        for val1 in all_shortest_path_dict.values():
            for val in val1.values():
                shortest_path_lens.append(val)
    return shortest_path_lens


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
#plt.style.use('ggplot')
#plt.style.use('seaborn-ticks')
#plt.style.use('seaborn-notebook')
#plt.rcParams['lines.linewidth']=3
#plt.rcParams['xtick.labelsize']=12
#plt.rcParams['ytick.labelsize']=12
#plt.rcParams['axes.labelsize']=14

    
def plot_distribution (data, path="", xlabel='', ylabel='', title='', xlog = True, ylog= True, showLine=False, intAxis=False) :
    counts = {}
    for item in data :
        if item not in counts :
            counts [ item ] = 0
        counts [ item ] += 1
    counts = sorted ( counts.items () )
    fig = plt.figure ()
    ax = fig.add_subplot (111)
    ax.scatter ([ k for (k , v ) in counts ] , [ v for (k , v ) in counts ])
    if(len(counts)<20):  # for tiny graph
        showLine=True
    if showLine==True:
        ax.plot ([ k for (k , v ) in counts ] , [ v for (k , v ) in counts ])
    if xlog == True:
        ax.set_xscale ( 'log')
    if ylog == True:
        ax.set_yscale ( 'log')
    if intAxis == True:
        gca = fig.gca()
        gca.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2e"))
    ax.set_xlabel ( xlabel)
    ax.set_ylabel ( ylabel )
    plt.xticks(rotation=60)
    plt.title(title)
    if isinstance(path, str) and path != "":
        fig.savefig(path)
    else:
        fig.show()
    plt.close()

    
def plot_degree_bar (G) :
    degs = {}
    for n in G.nodes () :
        deg = G.degree ( n )
        if deg not in degs :
            degs [ deg ] = 0
        degs [ deg ] += 1
    items = sorted ( degs.items () )
    fig = plt.figure ()
    ax = fig.add_subplot (111)
    print(items)
    ax.bar([ k for (k , v ) in items ] , [ v for (k , v ) in items ])
    ax.set_xlabel ( 'Degree ($k$)')
    ax.set_ylabel ( 'Number of nodes with degree $k$ ($N_k$)')


def compute_graph_metrics(G, metrics, graph_name, approximate=True):
    mets = []
    if "n_nodes" in metrics:
        mets += [["Number of Nodes:", G.number_of_nodes()]]
        print(mets[-1])
    if "n_edges" in metrics:
        mets += [["Number of Edges:", G.number_of_edges()]]
        print(mets[-1])
    if "average_degree" in metrics:
        vertex_degree_pairs = G.degree()
        avg_degree = 0
        for vertex, degree in vertex_degree_pairs:
            avg_degree += degree
        avg_degree /= G.number_of_nodes()
        mets += [["Average Degree:", avg_degree]]
        print(mets[-1])
    if "largest_connected_component" in metrics:
        CC_sorted = sorted(nx.connected_components(G), key=len)
        largest_CC = len(CC_sorted[-1])
        mets += [["Largest Connected Component:", largest_CC]]
        print(mets[-1])
    if "average_local_clustering_coefficient" in metrics:
        if approximate:
            CC = list(CC_sorted[-1])
            CC_graph = G.subgraph(CC)
            LLCs = nx.clustering(CC_graph)
            avg_LLC = sum([LLC for LLC in LLCs.values()]) / CC_graph.number_of_nodes()
        else:
            LLCs = nx.clustering(G)
            avg_LLC = sum([LLC for LLC in LLCs.values()]) / G.number_of_nodes()
        mets += [["Average Local Clustering Coefficient:", avg_LLC]]
        print(mets[-1])
    if "average_shortest_path_length" in metrics:
        CC = list(CC_sorted[-1])
        CC_graph = G.subgraph(CC)
        n_samples = CC_graph.number_of_nodes()
        if len(CC) > 3000:
            n_samples = 10
        vertex_selection = np.random.choice(CC_graph.number_of_nodes(), size=n_samples, replace=False)
        SPLs = []
        for i in vertex_selection:
            path_lengths = nx.single_source_shortest_path_length(CC_graph, CC[i])
            SPLs += [length for length in path_lengths.values()]
        avg_SPL = sum(SPLs) / len(SPLs)
        mets += [["Average Shortest Path Length:", avg_SPL]]
        print(mets[-1])
        """
        CC = list(CC_sorted[-1])
        CC_graph = G.subgraph(CC)
        n_samples = CC_graph.number_of_nodes()
        if len(CC) > 3000:
            n_samples = 10
        vertex_selection = np.random.choice(CC_graph.number_of_nodes(), size=n_samples, replace=False)
        SPLs = []
        for i in vertex_selection:
            path_lengths = nx.single_source_shortest_path_length(CC_graph, CC[i])
            SPLs += [length for length in path_lengths.values()]
        """
    if "diameter" in metrics:
        diameter = max(SPLs)
        mets += [["Diameter:", diameter]]
        print(mets[-1])
    return mets
