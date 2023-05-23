import torch
import torch_geometric
import numpy as np
import sys
import inspect
from torch.utils.tensorboard import SummaryWriter as SW
from torch.utils.tensorboard import _pytorch_graph

import Utility as util


# Setup str -> function maps for all PyTorch and PYG functions/classes needed for model definition.
#   These maps allow users to call, for example, the LSTM module using layer_fn_map["LSTM"]().
#   This allows for higher-level model definitions that are agnostic of chosen layers, activations, etc.
#   For example, we can now define a cell-agnostic RNN model by passing a layer argument (e.g. layer="LSTM")
#       that specifies the specific layer to use. Thus, this model would define a general RNN architecture,
#       such as sequence-to-sequence, with the specific layer type as a hyper-parameter.

#   init activation -> function map
act_fn_map = {"identity": lambda x:x}
for name, fn in inspect.getmembers(torch.nn.functional, inspect.isfunction):
    if not name.startswith("_"):
        act_fn_map[name] = fn

#   init optimization -> class constructor function map
opt_fn_map = {}
for name, fn in inspect.getmembers(torch.optim, inspect.isclass):
    opt_fn_map[name] = fn

#   init scheduler -> function map
sched_fn_map = {}
for name, fn in inspect.getmembers(torch.optim.lr_scheduler, inspect.isclass):
    sched_fn_map[name] = fn

#   init initialization -> function map
init_fn_map = {}
for name, fn in inspect.getmembers(torch.nn.init, inspect.isfunction):
    if not name.startswith("_"):
        init_fn_map[name] = fn

#   init loss -> class constructor function map
loss_fn_map = {}
for name, fn in inspect.getmembers(torch.nn.modules.loss, inspect.isclass):
    if not name.startswith("_"):
        loss_fn_map[name] = fn

#   init layer -> class constructor function map
layer_fn_map = {}
for name, fn in inspect.getmembers(torch.nn.modules, inspect.isclass):
    if not name.startswith("_") and not "Loss" in name:
        layer_fn_map[name] = fn

#   init gcn layer -> class constructor function map
gcnlayer_fn_map = {}
for name, fn in inspect.getmembers(torch_geometric.nn.conv, inspect.isclass):
    if issubclass(fn, torch_geometric.nn.conv.MessagePassing) and not name in ["MessagePassing"]:
        gcnlayer_fn_map[name] = fn
for name, fn in inspect.getmembers(torch_geometric.nn.dense, inspect.isclass):
    if name.endswith("Conv"):
        gcnlayer_fn_map[name] = fn

#   init gcn layer -> supported feature list map
gcnlayer_supported_map = {
    "APPNP": ["SparseTensor", "edge_weight", "static"],
    "ARMAConv": ["SparseTensor", "edge_weight", "static", "lazy"],
    "CGConv": ["SparseTensor", "edge_attr", "bipartite", "static"],
    "ChebConv": ["edge_weight", "static", "lazy"],
    "DNAConv": ["SparseTensor", "edge_weight"],
    "FAConv": ["SparseTensor", "edge_weight", "static", "lazy"],
    "FlowConv": ["SparseTensor", "edge_weight", "static", "lazy"],
    "GCNConv": ["SparseTensor", "edge_weight", "static", "lazy"],
    "GCN2Conv": ["SparseTensor", "edge_weight", "static"],
    "GraphConv": ["SparseTensor", "edge_weight", "bipartite","static", "lazy"],
    "GatedGraphConv": ["SparseTensor", "edge_weight", "static"],
    "LEConv": ["SparseTensor", "edge_weight", "bipartite", "static", "lazy"],
    "LGConv": ["SparseTensor", "edge_weight", "static"],
    "ResGatedGraphConv": ["SparseTensor", "bipartite", "static", "lazy"],
    "SAGEConv": ["SparseTensor", "bipartite", "static", "lazy"],
    "SGConv": ["SparseTensor", "edge_weight", "static", "lazy"],
    "TAGConv": ["SparseTensor", "edge_weight", "static", "lazy"],
}

#   init gcn layer -> requirements dict map
gcnlayer_required_map = {
    "AGNNConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "APPNP": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["?", "|E|"], 
    },
    "ARMAConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["|E|"], 
    },
    "CGConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "ChebConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["|E|"], 
    },
    "ClusterGCNConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "DenseGCNConv": {
        "inputs": ["x", "adj"],
        "x.shape": ["?", "|V|", "F"],
        "adj.shape": ["?", "|V|", "|V|"],
    },
    "DenseGINConv": {
        "inputs": ["x", "adj"],
        "x.shape": ["?", "|V|", "F"],
        "adj.shape": ["?", "|V|", "|V|"],
    },
    "DenseGraphConv": {
        "inputs": ["x", "adj"],
        "x.shape": ["N", "|V|", "F"],
        "adj.shape": ["N", "|V|", "|V|"],
    },
    "DenseSAGEConv": {
        "inputs": ["x", "adj"],
        "x.shape": ["?", "|V|", "F"],
        "adj.shape": ["?", "|V|", "|V|"],
    },
    "DNAConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "DynamicEdgeConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "ECConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "EdgeConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "EGConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "FAConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "FastRGCNConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "FeaStConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "FlowConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "FiLMConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["?", "|E|"], 
    },
    "GatedGraphConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GATConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GATv2Conv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GCN2Conv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GCNConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["|E|"], 
    },
    "GENConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GeneralConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GINConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GINEConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GMMConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GraphConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["|E|"], 
    },
    "GravNetConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "HGTConv": {
        "inputs": ["x_dict", "edge_index_dict"],
        "x_dict.shape": ["|V|", "F"],
        "edge_index_dict.shape": [2, "|E|"],
    },
    "HypergraphConv": {
        "inputs": ["x", "hyperedge_index"],
        "x.shape": ["|V|", "F"],
        "hyperedge_index.shape": ["|V|", "|E|"],
    },
    "LEConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["|E|"], 
    },
    "LEConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["|E|"], 
    },
    "MFConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "NNConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "PANConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "PDNConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "PNAConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "PointConv": {
        "inputs": ["x", "pos", "edge_index"],
        "x.shape": ["|V|", "F"],
        "pos.shape": ["|V|", 3],
        "edge_index.shape": [2, "|E|"],
    },
    "PointNetConv": {
        "inputs": ["x", "pos", "edge_index"],
        "x.shape": ["|V|", "F"],
        "pos.shape": ["|V|", 3],
        "edge_index.shape": [2, "|E|"],
    },
    "PPFConv": {
        "inputs": ["x", "pos", "normal", "edge_index"],
        "x.shape": ["|V|", "F"],
        "pos.shape": ["|V|", 3],
        "normal.shape": ["|V|", 3],
        "edge_index.shape": [2, "|E|"],
    },
    "ResGatedGraphConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "RGCNConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "SAGEConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "SGConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["|E|"], 
    },
    "SignedConv": {
        "inputs": ["x", "pos_edge_index", "neg_edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "pos_edge_index.shape": [2, "|E^(+)|"],
        "neg_edge_index.shape": [2, "|E^(-)|"],
    },
    "SplineConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "SuperGATConv": {
        "inputs": ["x", "edge_index", "neg_edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "neg_edge_index.shape": [2, "|E^(-)|"],
    },
    "TAGConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["|E|"], 
    },
    "TransformerConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
}


# Override add_graph() from tensorboard.SummaryWriter to allow the passage of paramters to trace()
#   This is needed to pass strict=False so that models with dictionary outputs may be logged with add_graph()
class SummaryWriter(SW):

    def add_graph(self, model, input_to_model=None, verbose=False, trace_kwargs={}):
        self._get_file_writer().add_graph(graph(model, input_to_model, verbose, trace_kwargs))


# Override graph() from tensorboard._pytorch_graph to allow the passage of paramters to trace()
#   This is needed to pass strict=False so that models with dictionary outputs may be logged with add_graph()
def graph(model, args, verbose=False, trace_kwargs={}):
    with torch.onnx.select_model_mode_for_export(model, torch.onnx.TrainingMode.EVAL):
        try:
            trace = torch.jit.trace(model, args, **trace_kwargs)
            graph = trace.graph
            torch._C._jit_pass_inline(graph)
        except RuntimeError as e:
            print(e)
            print('Error occurs, No graph saved')
            raise e
        if verbose:
            print(graph)
    list_of_nodes = _pytorch_graph.parse(graph, trace, args)
    stepstats = _pytorch_graph.RunMetadata(
        step_stats=_pytorch_graph.StepStats(
            dev_stats=[_pytorch_graph.DeviceStepStats(device="/device:CPU:0")]
        )
    )
    graph_def = _pytorch_graph.GraphDef(
        node=list_of_nodes,
        versions=_pytorch_graph.VersionDef(producer=22)
    )
    return graph_def, stepstats


def l1_reg_grad(weight, factor, weight_target=0):
    return factor * torch.sign(weight - weight_target)


def l2_reg_grad(weight, factor, weight_target=0):
    return factor * (weight - weight_target)


def unravel_index(indices, shape):
    """Converts flat indices into unraveled coordinates in a target shape. This is a `torch` implementation of `numpy.unravel_index`.

    Arguments
    ---------
    indices : LongTensor with shape=(?, N)
    shape : tuple with shape=(D,)

    Returns
    -------
    coord : LongTensor with shape=(?, N, D)

    Source
    ------
    author : francois-rozet @ https://github.com/pytorch/pytorch/issues/35674#issuecomment-739492875

    """
    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices
    coord = torch.zeros(indices.size() + shape.size(), dtype=indices.dtype, device=indices.device)
    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = torch.div(indices, dim, rounding_mode="trunc")
    return coord.flip(-1)


def batch_sampler_collate(batch):
    return batch[0]


def bcast_safe_view(x, y, matched_dim=-1):
    """ Reshapes x to match dimensionality of y by adding dummy axes so that any broadcasted operation on x and y is deterministic

    Arguments
    ---------
    x : tensor with len(x.shape) <= len(y.shape)
        the input data to be reshaped
    y : tensor
        the target data that x will be reshaped to
    matched_dim : int or tuple of ints
        dimensions/axis in which x and y have one-to-one correpsondence. input x will be broadcasted to all other dimensions/axes of y 
    """
    if isinstance(matched_dim, int):
        matched_dim = [matched_dim]
    bcast_safe_shape = list(1 for _ in range(len(y.shape)))
    for dim in matched_dim:
        bcast_safe_shape[dim] = y.shape[dim]
    return x.view(bcast_safe_shape)


def align_dims(x, y, x_dim, y_dim):
    if len(x.shape) > len(y.shape):
        raise ValueError(
            "Input x must have fewer dimensions than y to be aligned. Received x.shape=%s and y.shape=%s" % (
                x.shape, y.shape
            )
        )
    elif len(x.shape) == len(y.shape):
        return x
    if x_dim < 0:
        x_dim = len(x.shape) + x_dim
    if y_dim < 0:
        y_dim = len(y.shape) + y_dim
    new_shape = [1 for _ in y.shape]
    start = y_dim - x_dim
    end = start + len(x.shape)
    new_shape[start:end] = x.shape
    return x.view(new_shape)


def align(inputs, dims):
    if not isinstance(inputs, (tuple, list)) or not all([isinstance(inp, torch.Tensor) for inp in inputs]):
        raise ValueError("Argumet inputs must be tuple or list of tensors")
    if len(inputs) < 2:
        return inputs
    if isinstance(dims, int):
        dims = tuple(dims for inp in inputs)
    elif not isinstance(dim, tuple):
        raise ValueError("Argument dim must be int or tuple of ints")
    input_dims = [inp.dim() for inp in inputs]
    idx = input_dims.index(max(input_dims))
    y = inputs[idx]
    y_dim = dims[idx]
    return [align_dims(inp, y, dim, y_dim) for inp, dim in zip(inputs, dims)]


def maybe_expand_then_cat(tensors, dim=0):
    debug = 0
    if debug:
        for tensor in tensors:
            print(tensor.shape)
    dims = [tensor.dim() for tensor in tensors]
    if debug:
        print(dims)
    idx = dims.index(max(dims))
    if debug:
        print(idx)
    shape = list(tensors[idx].shape)
    if debug:
        print(shape)
    shape[dim] = -1
    if debug:
        print(shape)
    tensors = align(tensors, dim)
    for i, tensor in enumerate(tensors):
        tensors[i] = tensor.expand(shape)
    return torch.cat(tensors, dim)


def squeeze_not(x, dim):
    debug = 0
    if isinstance(dim, int):
        dim = (dim,)
    elif not isinstance(dim, tuple):
        raise TypeError(dim)
    orig_dim = x.dim()
    dims = list(range(orig_dim))
    for i in range(len(dim)):
        _dim = dim[i]
        if _dim < 0:
            _dim = orig_dim + _dim
        dims.remove(_dim)
    if debug:
        print("squeeze_dims =", dims)
        print(x.size())
    for i, _dim in enumerate(dims):
        _dim -= (orig_dim - x.dim())
        x = torch.squeeze(x, _dim)
        if debug:
            print("_dim =", _dim, "x =", x.size())
    if debug:
        input()
    return x


#   init activation -> function map
act_fn_map = util.sort_dict(act_fn_map)

#   init optimization -> class constructor function map
opt_fn_map = util.sort_dict(opt_fn_map)

#   init scheduler -> function map
sched_fn_map = util.sort_dict(sched_fn_map)

#   init initialization -> function map
init_fn_map = util.sort_dict(init_fn_map)

#   init loss -> class constructor function map
loss_fn_map = util.sort_dict(loss_fn_map)

#   init layer -> class constructor function map
layer_fn_map = util.sort_dict(layer_fn_map)

#   init gcn layer -> class constructor function map
gcnlayer_fn_map = util.sort_dict(gcnlayer_fn_map)

#   init gcn layer -> supported feature list map
gcnlayer_supported_map = util.sort_dict(gcnlayer_supported_map)

#   init gcn layer -> requirements dict map
gcnlayer_required_map = util.sort_dict(gcnlayer_required_map)


class EarlyStopper:

    def __init__(self, patience, init_steps=0):
        self.patience = patience
        self.n_plateau_steps = init_steps
        self.min_loss = sys.float_info.max

    def step(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.n_plateau_steps = 0
        else:
            self.n_plateau_steps += 1
        return self.stop()

    def stop(self):
        return self.patience > 0 and self.n_plateau_steps >= self.patience


class StandardLoader(torch.utils.data.Dataset):

    __sampled__ = ["x", "y"]
    debug = 0

    def __init__(self, data, var):
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")
        if "__sampled__" in data:
            self.__sampled__ = data["__sampled__"]
        if not "x" in data:
            raise ValueError("Input data must contain field \"x\"")
        if not all(item in data for item in self.__sampled__):
            raise ValueError("Input data must contained sampled fields %s" % (str(self.__sampled__)))
        self.data = data
        self.mb = {}

    def __len__(self):
        return self.data["x"].shape[0]

    def __getitem__(self, key):
        if self.debug:
            print("key =", type(key), "= len(%d)" % (len(key)), "=")
            if self.debug > 1:
                print(key)
        if self.debug:
            for item, data in self.data.items():
                print(item, "=", type(data), "=", data.shape, "=")
                if self.debug > 1:
                    print(data)
        for item in self.__sampled__:
            self.mb[item] = self.data[item][key]
        if self.debug:
            for item, sample in self.mb.items():
                print(item, "=", type(sample), "=", sample.shape, "=")
                if self.debug > 1:
                    print(sample)
        self.mb["__index__"] = key
        if self.debug:
            sys.exit(1)
        return self.mb


class PartitionLoader(torch.utils.data.Dataset):

    __sampled__ = ["x", "y"]
    debug = 0

    def __init__(self, data, var):
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")
        if "__sampled__" in data:
            self.__sampled__ = data["__sampled__"]
        if not ("x" in data and "indices" in data):
            raise ValueError("Input data must contain fields \"x\" and \"indices\"")
        if not all(item in data for item in self.__sampled__):
            raise ValueError("Input data must contained sampled fields %s" % (str(self.__sampled__)))
        self.indices = data["indices"]
        self.data = data
        self.mb = {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        if self.debug:
            print("key =", type(key), "= len(%d)" % (len(key)), "=")
            if self.debug > 1:
                print(key)
        if self.debug:
            for item, data in self.data.items():
                print(item, "=", type(data), "=", data.shape, "=")
                if self.debug > 1:
                    print(data)
        for item in self.__sampled__:
            self.mb[item] = self.data[item][self.indices[key]]
        if self.debug:
            for item, sample in self.mb.items():
                print(item, "=", type(sample), "=", sample.shape, "=")
                if self.debug > 1:
                    print(sample)
        self.mb["__index__"] = self.indices[key]
        if self.debug:
            sys.exit(1)
        return self.mb


class SlidingWindowLoader(torch.utils.data.Dataset):

    __sampled__ = ["x", "y"]
    __inputs__ = ["x"]
    __outputs__ = ["y"]
    debug = 0

    def __init__(self, data, var):
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")
        if "__sampled__" in data:
            self.__sampled__ = data["__sampled__"]
        if "__inputs__" in data:
            self.__inputs__ = data["__inputs__"]
        if "__outputs__" in data:
            self.__outputs__ = data["__outputs__"]
        if not ("x" in data and "y" in data):
            raise ValueError("Arugment data must contain fields \"x\" and \"y\"")
        if not all(item in data for item in self.__sampled__):
            raise ValueError("Argument data must contained sampled fields %s" % (str(self.__sampled__)))
        if not all(item in data for item in self.__inputs__):
            raise ValueError("Arugment data must contained input fields %s" % (str(self.__inputs__)))
        if not all(item in data for item in self.__outputs__):
            raise ValueError("Arugment data must contained output fields %s" % (str(self.__outputs__)))
        if len(data["x"].shape) != len(data["y"].shape):
            raise ValueError(
                "Data x and y must have equal dimensionality, received x.shape=%s and y.shape=%s" % (
                    str(data["x"].shape),
                    str(data["y"].shape)
                )
            )
        if len(data["x"].shape) == 3: # Original spatiotemporal format
            self.n_temporal, self.n_spatial, self.n_feature = data["x"].shape
            self.in_indices, self.out_indices = util.input_output_window_indices(
                self.n_temporal,
                var.temporal_mapping[0],
                var.temporal_mapping[1],
                var.horizon
            )
        elif len(data["x"].shape) == 4: # Reduced spatiotemporal format
            self.n_channel, self.n_temporal, self.n_spatial, self.n_feature = data["x"].shape
            self.in_indices, self.out_indices = util.input_output_window_indices(
                self.n_temporal,
                var.temporal_mapping[0],
                var.temporal_mapping[1],
                var.horizon
            )
            if self.debug:
                print(self.in_indices.shape, self.out_indices.shape)
            self.in_indices = np.tile(
                self.in_indices,
                (self.n_channel, 1, 1)
            ) + np.arange(self.n_channel)[:,None,None] * self.in_indices.shape[0]
            self.out_indices = np.tile(
                self.out_indices,
                (self.n_channel, 1, 1)
            ) + np.arange(self.n_channel)[:,None,None] * self.out_indices.shape[0]
            if self.debug:
                print(self.in_indices.shape, self.out_indices.shape)
                print(self.in_indices)
                print(self.out_indices)
            data["x"] = torch.reshape(data["x"], (-1,) + data["x"].shape[2:])
            data["y"] = torch.reshape(data["y"], (-1,) + data["y"].shape[2:])
            if self.debug:
                print(data["x"].shape, data["y"].shape)
            self.in_indices = np.reshape(self.in_indices, (-1,) + self.in_indices.shape[2:])
            self.out_indices = np.reshape(self.out_indices, (-1,) + self.out_indices.shape[2:])
            if self.debug:
                print(self.in_indices.shape)
                print(self.in_indices)
                print(self.out_indices.shape)
                print(self.out_indices)
        else:
            raise NotImplementedError(
                "SlidingWindowLoader only supports 3D and 4D inputs, received x.shape=%s" % (
                    str(data["x"].shape)
                )
            )
        self.data = data
        self.mb = {}

    def __len__(self):
        return self.in_indices.shape[0]

    #  Description:
    #   Pull a single sample or batch of samples from x and y
    # Arguments:
    #   key - index or list of indices to pull sample from
    def __getitem__(self, index):
        if self.debug:
            print("index =", type(index), "= len(%d)" % (len(index)), "=")
            if self.debug > 1:
                print(index)
        if self.debug:
            for key, value in self.data.items():
                print(key, "=", type(value), "=", value.shape, "=")
                if self.debug > 1:
                    print(value)
        for key in self.__sampled__:
            value = self.data[key]
            indices = self.in_indices[index,:]
            if key in self.__outputs__:
                indices = self.out_indices[index,:]
            final_shape = indices.shape + value.shape[1:]
            self.mb[key] = torch.reshape(value[np.reshape(indices, (-1,))], final_shape)
        if self.debug:
            for key, value in self.mb.items():
                print(key, "=", type(value), "=", value.shape, "=")
                if self.debug > 1:
                    print(value)
        self.mb["__index__"] = index
        if self.debug:
            sys.exit(1)
        return self.mb
