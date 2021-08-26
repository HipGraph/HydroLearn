from torch import optim as torch_opt
from torch.nn import functional as torch_func
from torch import nn as torch_nn
from torch import manual_seed, save, load
from torch_geometric import nn as torchg_nn
from os.path import exists as path_exists
import Utility as util
from inspect import currentframe
from time import time


class Model(torch_nn.Module):

    dataxes_layout_map = {
        "sample": 0, 
        "temporal": 1, 
        "spatial": 2, 
        "feature": 3, 
    }
    config_name_partition_pairs = [
        ["dataset", "train"],
        ["dataset", "valid"],
        ["dataset", "test"],
        ["transformation_resolution", None],
        ["feature_transformation_map", None],
        ["temporal_reduction", None],
        ["spatial_selection", "train"],
        ["spatial_selection", "valid"],
        ["spatial_selection", "test"],
        ["temporal_selection", "train"],
        ["temporal_selection", "valid"],
        ["temporal_selection", "test"],
        ["predictor_features", None],
        ["response_features", None],
        ["n_temporal_in", None],
        ["n_temporal_out", None]
    ]
    act_func_map = {
        "relu": torch_func.relu, 
        "tanh": torch_func.tanh, 
        "sigmoid": torch_func.sigmoid, 
        "identity": lambda x:x,
    }
    opt_func_map = {
        "sgd": torch_opt.SGD, 
        "adam": torch_opt.Adam,
        "adadelta": torch_opt.Adadelta,
    }
    init_func_map = {
        "constant": torch_nn.init.constant_,
        "normal": torch_nn.init.normal_, 
        "xavier": torch_nn.init.xavier_normal_, 
        "kaiming": torch_nn.init.kaiming_normal_,
    }
    loss_func_map = {
        "mae": torch_nn.L1Loss, 
        "mse": torch_nn.MSELoss, 
        "smooth_mae": torch_nn.SmoothL1Loss,
    }
    layer_func_map = {
        "lstm": torch_nn.LSTM,
        "lstm_cell": torch_nn.LSTMCell,
        "linear": torch_nn.Linear, 
        "dropout": torch_nn.Dropout,
    }
    gnnlayer_func_map = {
        "sage": torchg_nn.SAGEConv,
        "gcn": torchg_nn.GCNConv,
        "graph": torchg_nn.GraphConv,
        "cluster": torchg_nn.ClusterGCNConv,
    }    

    def __init__(self):
        super(Model, self).__init__()

    def curate_config(self, var):
        config = []
        for [name, partition] in self.config_name_partition_pairs:
            config += ["%s = %s" % (var.get_key(name, partition), var.get(name, partition))]
        return "\n".join(config)

    def init_params(self, init, seed=-1):
        manual_seed((seed if seed > -1 else time.time))
        for name, param in self.named_parameters():
            if "bias" in name:
                self.init_func_map["constant"](param, 0.0)
            elif "weight" in name:
                self.init_func_map[init](param)

    def load(self, var, chkpt_path):
        chkpt_path += ".pth"
        if var.get("process_rank") == var.get("root_process_rank"):
            if not path_exists(chkpt_path):
                raise FileNotFoundError(chkpt_path)
            checkpoint = load(chkpt_path)
            self.load_state_dict(checkpoint["model_state_dict"])
            self.opt = self.opt_func_map[var.get("opt")](
                self.parameters(), 
                var.get("lr")
            )
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        if var.get("n_processes") > 1:
            for param_tensor in self.parameters():
                torch_dist.broadcast(param_tensor.data, src=var.get("root_process_rank"))
        return self
    
    def checkpoint(self, path):
        chkpt_dict = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.opt.state_dict()
        }
        save(chkpt_dict, path)

    def get_id(self, var):
        return str(util.hash_str_to_int(self.curate_config(var), 10))

    def data_layout(self):
        return self.dataxes_layout_map

    def name(self):
        return self.__class__.__name__

    def prepare(self, data, use_gpu, revert=False):
        return [self.prepare_model(use_gpu, revert)] + self.prepare_data(data, use_gpu, revert)
    
    def prepare_model(self, use_gpu, revert=False):
        if revert:
            device = util.get_device(False)
            self = util.to_device(self, device)
        else:
            device = util.get_device(use_gpu)
            self = util.to_device(self, device)
        return self

    def prepare_data(self, data, use_gpu, revert=False):
        func_name = self.__class__.__name__+"."+currentframe().f_code.co_name
        raise NotImplementedError("Implement %s() please!" % (func_name))

    def optimize(self):
        func_name = self.__class__.__name__+"."+currentframe().f_code.co_name
        raise NotImplementedError("Implement %s() please!" % (func_name))

    def predict(self):
        func_name = self.__class__.__name__+"."+currentframe().f_code.co_name
        raise NotImplementedError("Implement %s() please!" % (func_name))
