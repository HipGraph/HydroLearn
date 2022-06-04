import torch
import torch_geometric
import numpy as np
import os
import sys
import Utility as util
import time
import inspect
from torch.utils.tensorboard import SummaryWriter as SW
from torch.utils.tensorboard import _pytorch_graph
from Arguments import ArgumentBuilder


class StandardLoader(torch.utils.data.Dataset):

    def __init__(self, data, var):
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        if not ("X" in data and "Y" in data):
            raise ValueError("Data must at least contain \"X\" and \"Y\" fields.")
        self.data = data

    def __len__(self):
        return self.data["X"].shape[0]

    def __getitem__(self, key):
        return {"X": self.data["X"][key], "Y": self.data["Y"][key]}


class SlidingWindowLoader(torch.utils.data.Dataset):

    debug = 0
    
    def __init__(self, data, var):
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        if not ("X" in data and "Y" in data):
            raise ValueError("Data must contain \"X\" and \"Y\" fields.")
        self.X, self.Y = data["X"], data["Y"]
        if len(self.X.shape) != len(self.Y.shape):
            raise ValueError(
                "Data X and Y must have equal dimensionality, received X.shape=%s and Y.shape=%s" % (
                    str(self.X.shape), 
                    str(self.Y.shape)
                )
            )
        if len(self.X.shape) == 3: # Original spatiotemporal format
            self.n_temporal, self.n_spatial, self.n_feature = self.X.shape
            self.in_indices, self.out_indices = util.input_output_window_indices(
                self.n_temporal, 
                var.temporal_mapping[0], 
                var.temporal_mapping[1], 
            )
        elif len(self.X.shape) == 4: # Reduced spatiotemporal format
            self.n_channel, self.n_temporal, self.n_spatial, self.n_feature = self.X.shape
            self.in_indices, self.out_indices = util.input_output_window_indices(
                self.n_temporal, 
                var.temporal_mapping[0], 
                var.temporal_mapping[1], 
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
            self.X = torch.reshape(self.X, (-1,) + self.X.shape[2:])
            self.Y = torch.reshape(self.Y, (-1,) + self.Y.shape[2:])
            if self.debug:
                print(self.X.shape, self.Y.shape)
            self.in_indices = np.reshape(self.in_indices, (-1,) + self.in_indices.shape[2:])
            self.out_indices = np.reshape(self.out_indices, (-1,) + self.out_indices.shape[2:])
            if self.debug:
                print(self.in_indices.shape)
                print(self.in_indices)
                print(self.out_indices.shape)
                print(self.out_indices)
        else:
            raise NotImplementedError(
                "SlidingWindowLoader only supports 3D and 4D inputs, received X.shape=%s" % (
                    str(self.X.shape)
                )
            )

    def __len__(self):
        return self.in_indices.shape[0]

    #  Description:
    #   Pull a single sample or batch of samples from X and Y
    # Arguments:
    #   key - index or list of indices to pull sample from
    def __getitem__(self, key):
        X, Y = self.X, self.Y
        if self.debug:
            print("key =", key)
            print(X.shape, Y.shape)
        X, Y = X[self.in_indices[key],:,:], Y[self.out_indices[key],:,:]
        if self.debug:
            print(X.shape, Y.shape)
            sys.exit(1)
        return {"X": X, "Y": Y}


class Model(torch.nn.Module):

    # Setup str -> func maps for all PyTorch and PYG functions/classes needed for model definition.
    #   These maps allow users to call, for example, the LSTM module using layer_func_map["LSTM"]().
    #   This allows for higher-level model definitions that are agnostic of chosen layers, activations, etc.
    #   For example, we can now define a cell-agnostic RNN model by passing a layer argument (e.g. layer="LSTM")
    #       that specifies the specific layer to use. Thus, this model would define a general RNN architecture,
    #       such as sequence-to-sequence, with the specific layer type as a hyper-parameter.
    act_func_map = {"identity": lambda x:x}
    for name, func in inspect.getmembers(torch.nn.functional, inspect.isfunction):
        if not name.startswith("_"):
            act_func_map[name] = func
    opt_func_map = {}
    for name, func in inspect.getmembers(torch.optim, inspect.isclass):
        opt_func_map[name] = func
    init_func_map = {}
    for name, func in inspect.getmembers(torch.nn.init, inspect.isfunction):
        if not name.startswith("_"):
            init_func_map[name] = func
    loss_func_map = {}
    for name, func in inspect.getmembers(torch.nn.modules.loss, inspect.isclass):
        if not name.startswith("_"):
            loss_func_map[name] = func
    layer_func_map = {}
    for name, func in inspect.getmembers(torch.nn.modules, inspect.isclass):
        if not name.startswith("_") and not "Loss" in name:
            layer_func_map[name] = func
    gnnlayer_func_map = {}
    for name, func in inspect.getmembers(torch_geometric.nn.conv, inspect.isclass):
        gnnlayer_func_map[name] = func
    for name, func in inspect.getmembers(torch_geometric.nn.dense, inspect.isclass):
        gnnlayer_func_map[name] = func
    gnnlayer_supported_map = {
        "GCNConv": ["SparseTensor", "edge_weight", "static", "lazy"],
        "ChebConv": ["edge_weight", "static", "lazy"], 
        "SageConv": ["SparseTensor", "bipartite", "static", "lazy"], 
        "GraphConv": ["SparseTensor", "edge_weight", "bipartite","static", "lazy"], 
        "GatedGraphConv": ["SparseTensor", "edge_weight", "static"], 
        "ResGatedGraphConv": ["SparseTensor", "bipartite", "static", "lazy"], 
        "TAGConv": ["SparseTensor", "edge_weight", "static", "lazy"], 
        "ARMAConv": ["SparseTensor", "edge_weight", "static", "lazy"], 
        "SGConv": ["SparseTensor", "edge_weight", "static", "lazy"], 
        "APPNP": ["SparseTensor", "edge_weight", "static"], 
        "DNAConv": ["SparseTensor", "edge_weight"], 
        "LEConv": ["SparseTensor", "edge_weight", "bipartite", "static", "lazy"], 
        "GCN2Conv": ["SparseTensor", "edge_weight", "static"], 
        "FAConv": ["SparseTensor", "edge_weight", "static", "lazy"], 
        "LGConv": ["SparseTensor", "edge_weight", "static"], 
    }
    # Configuration variables (used to hash the model to a 10-digit ID)
    config_name_partition_pairs = [
        ["dataset", "train"],
        ["dataset", "valid"],
        ["dataset", "test"],
        ["transformation_resolution", None],
        ["feature_transformation_map", None],
        ["default_feature_transformation", None], 
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
    # Other variables
    LoaderDatasetClass = SlidingWindowLoader
    warnings = False
    debug = 0
    train_losses, valid_losses, test_losses = [-1], [-1], [-1]
    to_gpu_time, to_cpu_time = 0, 0

    def __init__(self):
        super(Model, self).__init__()

    # Purpose:
    #   The forward pass of this model. Required if using PyTorch for back-propagation
    # Preconditions:
    #   inputs={"X": torch.float, "n_temporal_out": int}
    #   X.shape=(n_samples, n_temporal_in, n_spatial, n_predictors)
    # Postconditions:
    #   a.shape=(n_samples, n_temporal_out, n_spatial, n_responses)
    def forward(self, inputs):
        func_name = "%s.%s" % (self.__class__.__name__, inspect.currentframe().f_code.co_name)
        raise NotImplementedError("Implement %s() please!" % (func_name))
        return a

    def optimize(self, train_dataset, valid_dataset, test_dataset, var):
        self.use_gpu = var.use_gpu
        self.all_to_gpu = var.use_gpu and var.gpu_data_mapping == "all"
        self.mbatches_to_gpu = var.use_gpu and var.gpu_data_mapping == "minibatch"
        # Reformat data to accepted shape, convert to tensors, put data + model onto device, then unpack
        train = self.pull_data(train_dataset, "train", var)
        valid = self.pull_data(valid_dataset, "valid", var)
        test = self.pull_data(test_dataset, "test", var)
        self, train = self.prepare(train, var.use_gpu)
        self, valid = self.prepare(valid, var.use_gpu)
        self, test = self.prepare(test, var.use_gpu)
        # Initialize loaders, loss, optimizer, parameters, etc
        train_iterable = self.LoaderDatasetClass(train, var)
        sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(
                train_iterable, 
                generator=torch.Generator().manual_seed(
                    var.batch_shuffle_seed if var.batch_shuffle_seed > -1 else time.time()
                )
            ), 
            var.mbatch_size, 
            True
        )
        train_loader = torch.utils.data.DataLoader(train_iterable, sampler=sampler)
        if not valid_dataset.is_empty():
            valid_iterable = self.LoaderDatasetClass(valid, var)
            sampler = torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(valid_iterable), 
                var.mbatch_size, 
                False
            )
            valid_loader = torch.utils.data.DataLoader(valid_iterable, sampler=sampler)
        if not test_dataset.is_empty():
            test_iterable = self.LoaderDatasetClass(test, var)
            sampler = torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(test_iterable), 
                var.mbatch_size, 
                False
            )
            test_loader = torch.utils.data.DataLoader(test_iterable, sampler=sampler)
        self.crit = self.criterion(var)
        self.opt = self.optimizer(var)
        self.init_params(var.initializer, var.initialization_seed)
        self.summary_writer = SummaryWriter(var.checkpoint_dir)
        # Commence optimization
        self.train_losses, self.valid_losses, self.test_losses = [], [], []
        n_plateau_epochs, min_valid_loss = 0, sys.float_info.max
        for epoch in range(var.n_epochs+1):
            print(35 * "+")
            self.train()
            epoch_loss, n_samples = 0, 0
            for mb_in in train_loader: # Training set pass
                mb_in = util.merge_dicts(mb_in, train, False)
#                print("SHAPE =", mb_in["X"].shape)
                mb_in["X"] = torch.squeeze(mb_in["X"], 0)
                mb_in["Y"] = torch.squeeze(mb_in["Y"], 0)
#                print("SHAPE =", mb_in["X"].shape)
                if self.mbatches_to_gpu:
                    start = time.time()
                    for key in mb_in.keys():
                        mb_in[key] = util.to_device(mb_in[key], util.get_device(True))
                    self.to_gpu_time += time.time() - start
                mb_out = self.forward(mb_in)
                if self.mbatches_to_gpu:
                    start = time.time()
                    for key in mb_in.keys():
                        mb_in[key] = util.to_device(mb_in[key], util.get_device(False))
                    self.to_cpu_time += time.time() - start
                mb_loss = self.loss(mb_in, mb_out)
                if epoch > 0:
                    self.step(mb_loss, var)
                epoch_loss += self.loss_to_numeric(mb_loss)
            epoch_loss /= len(train_loader)
            self.train_losses += [epoch_loss]
            print("Epoch %d : Train Loss = %.5f" % (epoch, epoch_loss))
            self.eval()
            if not valid_dataset.is_empty(): # Validation set pass
                with torch.set_grad_enabled(False):
                    epoch_loss, n_samples = 0, 0
                    for mb_in in valid_loader:
                        mb_in = util.merge_dicts(mb_in, valid, False)
                        mb_in["X"] = torch.squeeze(mb_in["X"], 0)
                        mb_in["Y"] = torch.squeeze(mb_in["Y"], 0)
                        if self.mbatches_to_gpu:
                            start = time.time()
                            for key in mb_in.keys():
                                mb_in[key] = util.to_device(mb_in[key], util.get_device(True))
                            self.to_gpu_time += time.time() - start
                        mb_out = self.forward(mb_in)
                        if self.mbatches_to_gpu:
                            start = time.time()
                            for key in mb_in.keys():
                                mb_in[key] = util.to_device(mb_in[key], util.get_device(False))
                            self.to_cpu_time += time.time() - start
                        mb_loss = self.loss(mb_in, mb_out)
                        epoch_loss += self.loss_to_numeric(mb_loss)
                    epoch_loss /= len(valid_loader)
                    print("Epoch %d : Valid Loss = %.5f" % (epoch, epoch_loss))
                    self.valid_losses += [epoch_loss]
                    if epoch_loss < min_valid_loss: # Check for improvement and update early stopping
                        min_valid_loss = epoch_loss
                        n_plateau_epochs = 0
                        path = os.sep.join([var.checkpoint_dir, "Best.pth"])
                        self.checkpoint(path)
                    else:
                        n_plateau_epochs += 1
                        if var.early_stop_epochs > 0 and n_plateau_epochs % var.early_stop_epochs == 0:
                            break
            if not test_dataset.is_empty(): # Testing set pass
                with torch.set_grad_enabled(False):
                    epoch_loss, n_samples = 0, 0
                    for mb_in in test_loader:
                        mb_in = util.merge_dicts(mb_in, test, False)
                        mb_in["X"] = torch.squeeze(mb_in["X"], 0)
                        mb_in["Y"] = torch.squeeze(mb_in["Y"], 0)
                        if self.mbatches_to_gpu:
                            start = time.time()
                            for key in mb_in.keys():
                                mb_in[key] = util.to_device(mb_in[key], util.get_device(True))
                            self.to_cpu_time += time.time() - start
                        mb_out = self.forward(mb_in)
                        if self.mbatches_to_gpu:
                            start = time.time()
                            for key in mb_in.keys():
                                mb_in[key] = util.to_device(mb_in[key], util.get_device(False))
                            self.to_cpu_time += time.time() - start
                        mb_loss = self.loss(mb_in, mb_out)
                        epoch_loss += self.loss_to_numeric(mb_loss)
                    epoch_loss /= len(test_loader)
                    print("Epoch %d :  Test Loss = %.5f" % (epoch, epoch_loss))
                    self.test_losses += [epoch_loss]
            print(35 * "+")
            self.update_optimizer(epoch, var)
            self.log_epoch_info(train, train_loader, valid, valid_loader, test, test_loader, epoch, var)
        self.summary_writer.close()
        # Save final model
        path = os.sep.join([var.checkpoint_dir, "Final.pth"])
        self.checkpoint(path)
        # Return data to original device (cpu), shape, and type (NumPy.ndarray)
        self, train = self.prepare(train, False, True)
        self, valid = self.prepare(valid, False, True)
        self, test = self.prepare(test, False, True)

    def predict(self, dataset, partition, var):
        self.use_gpu = var.use_gpu
        self.all_to_gpu = var.get("gpu_data_mapping") == "all"
        self.mbatches_to_gpu = var.get("gpu_data_mapping") == "minibatch"
        data = self.pull_data(dataset, partition, var)
        self, data = self.prepare(data, var.use_gpu)
        # Initialize loader
        data_iterable = self.LoaderDatasetClass(data, var)
        sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(data_iterable), 
            var.mbatch_size, 
            False
        )
        loader = torch.utils.data.DataLoader(data_iterable, sampler=sampler)
        # Commence prediction
        self.eval()
        with torch.set_grad_enabled(False):
            if var.method == "direct":
                Yhats = []
                for mb_in in loader:
                    mb_in = util.merge_dicts(mb_in, data, False)
                    mb_in["X"] = torch.squeeze(mb_in["X"], 0)
                    mb_in["Y"] = torch.squeeze(mb_in["Y"], 0)
                    Yhats.append(self.forward(mb_in)["Yhat"])
                if isinstance(Yhats[0], np.ndarray):
                    Yhat = np.concatenate(Yhats)
                elif isinstance(Yhats[0], torch.Tensor):
                    Yhat = torch.cat(Yhats)
                else:
                    raise ValueError("Cannot concatenate predicted elements of type %s" % (type(Yhats[0])))
        data["Yhat"] = Yhat
        self, data = self.prepare(data, False, True)
        return data["Yhat"]

    def criterion(self, var):
        return self.loss_func_map[var.loss]()

    def optimizer(self, var):
        opt = self.opt_func_map[var.optimizer](
            self.parameters(),
            lr=var.lr,
            weight_decay=var.regularization
        )
        return opt

    def update_optimizer(self, epoch, var):
        # Decay learning rate
        if var.lr_decay > 0:
            self.update_lr(var.lr / (1 + var.lr_decay * epoch))

    def update_lr(self, lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    # Takes a step of gradient descent
    def step(self, loss, var):
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), var.gradient_clip)
        self.opt.step()

    # Computes the loss
    def loss(self, mb_in, mb_out):
        return self.crit(mb_in["Y"], mb_out["Yhat"])

    def loss_to_numeric(self, loss):
        return loss.item()

    def log_epoch_info(self, train, train_loader, valid, valid_loader, test, test_loader, epoch, var):
        # Add loss scalars to tensorboard
        self.log_epoch_losses(train, train_loader, valid, valid_loader, test, test_loader, epoch, var)
        # Add weight histograms to tensorboard
        self.log_epoch_params(train, train_loader, valid, valid_loader, test, test_loader, epoch, var)
        # Add computational graph to tensorboard
        self.log_compgraph(train, train_loader, valid, valid_loader, test, test_loader, epoch, var)
    
    def log_epoch_losses(self, train, train_loader, valid, valid_loader, test, test_loader, epoch, var):
        self.summary_writer.add_scalar("Training Loss", self.train_losses[epoch], epoch)
        if len(self.valid_losses) >= epoch:
            self.summary_writer.add_scalar("Validation Loss", self.valid_losses[epoch], epoch)
        if len(self.valid_losses) >= epoch:
            self.summary_writer.add_scalar("Testing Loss", self.test_losses[epoch], epoch)

    def log_epoch_params(self, train, train_loader, valid, valid_loader, test, test_loader, epoch, var):
        for name, param in self.named_parameters():
            self.summary_writer.add_histogram(name, param, epoch)

    def log_compgraph(self, train, train_loader, valid, valid_loader, test, test_loader, epoch, var):
        if epoch == var.n_epochs:
            mb_in = util.merge_dicts(next(iter(train_loader)), train, False)
            mb_in["X"] = torch.squeeze(mb_in["X"], 0)
            mb_in["Y"] = torch.squeeze(mb_in["Y"], 0)
            _del = []
            for key, value in mb_in.items(): # Trace requires all inputs to be the same type (Tensor)
                if not isinstance(value, torch.Tensor):
                    try:
                        mb_in[key] = util.to_tensor(value, torch.float)
                    except:
                        _del.append(key)
            for key in _del:
                print("Deleting \"%s\" from the minibatch as it could not converted to tensor" % (key))
                del mb_in[key]
            self.summary_writer.add_graph(self, mb_in, trace_kwargs={"strict": False})

    # Defines the set of data to be pulled from ths dataset for optimization and prediction tasks
    def pull_data(self, dataset, partition, var):
        data = {}
        if not dataset.is_empty():
            data["X"] = dataset.spatiotemporal.transformed.reduced.get("predictor_features", partition)
            data["Y"] = dataset.spatiotemporal.transformed.reduced.get("response_features", partition)
            data["n_temporal_out"] = var.temporal_mapping[1]
        return data

    # Defines what must happen before optimization/prediction tasks can begin
    def prepare(self, data, use_gpu, revert=False):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.prepare_model(use_gpu, revert), self.prepare_data(data, use_gpu, revert)
    
    # Defines model preparation for optimization/prediction tasks
    def prepare_model(self, use_gpu, revert=False):
        if revert:
            self = util.to_device(self, util.get_device(False))
        else:
            self = util.to_device(self, util.get_device(use_gpu))
        return self

    # Defines data preparation for optimization/prediction tasks
    def prepare_data(self, data, use_gpu, revert=False):
        if revert: # Back to CPU
            for key, value in data.items():
                device = util.get_device(False)
                if isinstance(value, torch.Tensor): # PyTorch.Tensor: put on CPU and cast to NumPy.ndarray
                    value = util.to_device(value, device)
                    value = util.to_ndarray(value)
                else: # Everything else: do nothing
                    pass
                data[key] = value
        else: # To GPU or keep on CPU
            for key, value in data.items():
                device = util.get_device(use_gpu)
                if key in ["X", "Y", "Yhat", "E"]: # Which data may go to gpu if available
                    device = util.get_device(use_gpu and self.all_to_gpu)
                    _type = torch.float
                elif key in ["edge_indices"]: # Which data always goes to gpu if available
                    _type = torch.long
                elif "labels" in key:
                    continue
                elif isinstance(value, torch.Tensor):
                    _type = value.dtype
                elif isinstance(value, np.ndarray): # Numpy.ndarray: cast to PyTorch.Tensor and put on device
                    if self.warnings:
                        warn(
                            "Preparation for data \"%s\" undefined but found NumPy.ndarray." \
                            "Proceeding by casting to PyTorch.Tensor." % (key)
                        )
                    if issubclass(value.dtype.type, np.integer): # Integer data
                        _type = torch.long
                    elif issubclass(value.dtype.type, np.float): # Floating-point data
                        _type = torch.float
                    else:
                        raise ValueError(
                            "Do not know how to prepare \"%s\" which has data-type %s" % (
                                key, 
                                value.dtype.type
                            )
                        )
                elif isinstance(value, int):
                    _type = torch.long
                elif isinstance(vale, float):
                    _type = torch.float
                else:
                    raise ValueError(
                        "Do not know how to prepare \"%s\" which has data-type %s" % (
                            key, 
                            value.dtype.type
                        )
                    )
                value = util.to_tensor(value, [_type])
                value = util.to_device(value, device)
                data[key] = value
        return data

    def curate_config(self, var):
        names = [pair[0] for pair in self.config_name_partition_pairs]
        partitions = [pair[1] for pair in self.config_name_partition_pairs]
        config_var = var.checkout(names, partitions)
        arg_bldr = ArgumentBuilder()
        config = arg_bldr.view(config_var)
        return config

    def init_params(self, init, seed=-1):
        torch.manual_seed(seed if seed > -1 else time.time())
        if init is None:
            self.reset_parameters()
        else:
            for name, param in self.named_parameters():
                try:
                    if ".bias" in name:
                        self.init_func_map["constant_"](param, 0.0)
                    elif ".weight" in name:
                        self.init_func_map[init](param)
                except ValueError as err:
                    pass

    def reset_parameters(self):
        func_name = "%s.%s" % (self.__class__.__name__, inspect.currentframe().f_code.co_name)
        raise NotImplementedError("Implement %s() please!" % (func_name))

    def load(self, var, chkpt_path):
        chkpt_path += ".pth"
        if var.get("process_rank") == var.get("root_process_rank"):
            if not os.path.exists(chkpt_path):
                raise FileNotFoundError(chkpt_path)
            checkpoint = torch.load(chkpt_path)
            self.load_state_dict(checkpoint["model_state_dict"])
            self.opt = self.opt_func_map[var.get("optimizer")](
                self.parameters(), 
                var.get("lr")
            )
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
            self.train_losses = checkpoint["losses"]["train"]
            self.valid_losses = checkpoint["losses"]["valid"]
            self.test_losses = checkpoint["losses"]["test"]
        if var.get("n_processes") > 1:
            for param_tensor in self.parameters():
                torch_dist.broadcast(param_tensor.data, src=var.get("root_process_rank"))
        return self
    
    def checkpoint(self, path):
        chkpt_dict = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(), 
            "losses": {"train": self.train_losses, "valid": self.valid_losses, "test": self.test_losses}
        }
        torch.save(chkpt_dict, path)

    def n_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def get_id(self, var):
        return str(util.hash_str_to_int(self.curate_config(var), 10))

    def curate_info(self):
        info_names = ["No. Parameters"]
        info_values = [self.n_params()]
        report = []
        max_name_chars = max(len(name) for name in info_names)
        for name, value in zip(info_names, info_values):
            report += ["%-*s: %s" % (max_name_chars, name, str(value))]
        return "\n".join(report)

    def name(self):
        return self.__class__.__name__


class MultiHopGNN(Model):

    def __init__(
        self, 
        in_size, 
        out_size, 
        hidden_size=16, 
        n_hops=3, 
        layer_type="GCNConv", 
        hidden_activation="ReLU", 
        dropout=0.0, 
        use_edge_weights=True, 
    ):
        super(MultiHopGNN, self).__init__()
        # Setup
        #   Initialization function for each gnn layer
        self.gnnlayer_init_map = {}
        for gnnlayer in self.gnnlayer_func_map.keys():
            self.gnnlayer_init_map[gnnlayer] = self.default_init
            if hasattr(self, "%s_init" % (gnnlayer)):
                self.gnnlayer_init_map[gnnlayer] = getattr(self, "%s_init" % (gnnlayer))
        #   Forward function for each gnn layer
        self.gnnlayer_forward_map = {}
        for gnnlayer in self.gnnlayer_func_map.keys():
            self.gnnlayer_forward_map[gnnlayer] = self.default_forward
            if hasattr(self, "%s_forward" % (gnnlayer)):
                self.gnnlayer_forward_map[gnnlayer] = getattr(self, "%s_forward" % (gnnlayer))
        # Save all vars
        self.in_size, self.out_size, self.hidden_size = in_size, out_size, hidden_size
        self.n_hops, self.layer_type, self.hidden_activation = n_hops, layer_type, hidden_activation
        self.dropout = dropout
        self.use_edge_weights = use_edge_weights and "edge_weight" in self.gnnlayer_supported_map[layer_type]
        # Instantiate model layers
        self.name_layer_map = {}
        self.gnnlayer_init_map[layer_type](
            in_size, 
            out_size, 
            hidden_size, 
            n_hops, 
            layer_type, 
            hidden_activation, 
            dropout, 
        )
        self.gnn_forward = self.gnnlayer_forward_map[layer_type]

    def default_init(self, in_size, out_size, hidden_size, n_hops, layer_type, hidden_activation, dropout):
        for i in range(n_hops):
            _in_size, _out_size = hidden_size, hidden_size
            if i == 0:
                _in_size = in_size
            elif i == n_hops - 1:
                _out_size = out_size
            # GCN layer
            name, layer = "gnn_%d" % (i), self.gnnlayer_func_map[layer_type](_in_size, _out_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Activation layer
            name, layer = "act_%d" % (i), self.layer_func_map[hidden_activation]()
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Dropout layer
            name, layer = "drop_%d" % (i), self.layer_func_map["Dropout"](dropout)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    def ChebConv_init(self, in_size, out_size, hidden_size, n_hops, layer_type, hidden_activation, dropout):
        assert layer_type == "ChebConv"
        name, layer = "gnn_0", self.gnnlayer_func_map[layer_type](in_size, out_size, n_hops + 1)
        setattr(self, name, layer)
        self.name_layer_map[name] = layer

    def TAGConv_init(self, in_size, out_size, hidden_size, n_hops, layer_type, hidden_activation, dropout):
        assert layer_type == "TAGConv"
        name, layer = "gnn_0", self.gnnlayer_func_map[layer_type](in_size, out_size, n_hops)
        setattr(self, name, layer)
        self.name_layer_map[name] = layer

    def ARMAConv_init(self, in_size, out_size, hidden_size, n_hops, layer_type, hidden_activation, dropout):
        assert layer_type == "ARMAConv"
        name, layer = "gnn_0", self.gnnlayer_func_map[layer_type](
            in_size, 
            out_size, 
            num_layers=n_hops, 
            act=self.layer_func_map[hidden_activation], 
            dropout=dropout, 
        )
        setattr(self, name, layer)
        self.name_layer_map[name] = layer

    def SGConv_init(self, in_size, out_size, hidden_size, n_hops, layer_type, hidden_activation, dropout):
        assert layer_type == "SGConv"
        name, layer = "gnn_0", self.gnnlayer_func_map[layer_type](in_size, out_size, n_hops)
        setattr(self, name, layer)
        self.name_layer_map[name] = layer

    def APPNP_init(self, in_size, out_size, hidden_size, n_hops, layer_type, hidden_activation, dropout):
        assert layer_type == "APPNP"
        name, layer = "lin_0", self.layer_func_map["Linear"](in_size, out_size)
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        # Use teleport probability 0.1 found in "Model hyperparameters" section of 
        #   https://arxiv.org/pdf/1810.05997.pdf
        name, layer = "gnn_0", self.gnnlayer_func_map[layer_type](n_hops, 0.1, dropout=dropout)
        setattr(self, name, layer)
        self.name_layer_map[name] = layer

    def FAConv_init(self, in_size, out_size, hidden_size, n_hops, layer_type, hidden_activation, dropout):
        assert layer_type == "FAConv"
        name, layer = "lin_0", self.layer_func_map["Linear"](in_size, out_size)
        setattr(self, name, layer)
        self.name_layer_map[name] = layer
        for i in range(n_hops):
            # GCN layer
            if self.use_edge_weights: # Normalization not allowed by FAConv when providing edge weights
                name, layer = "gnn_%d" % (i), self.gnnlayer_func_map[layer_type](out_size, normalize=False)
            else:
                name, layer = "gnn_%d" % (i), self.gnnlayer_func_map[layer_type](out_size)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Activation layer
            name, layer = "act_%d" % (i), self.layer_func_map[hidden_activation]()
            setattr(self, name, layer)
            self.name_layer_map[name] = layer
            # Dropout layer
            name, layer = "drop_%d" % (i), self.layer_func_map["Dropout"](dropout)
            setattr(self, name, layer)
            self.name_layer_map[name] = layer

    # Preconditions:
    #   inputs={"X": torch.float, "edge_indices": torch.long, "n_temporal_out": int}
    #   X.shape[-2:]=(n_spatial, in_size)
    #   edge_indices.shape=(2, n_edges)
    # Postconditions:
    #   outputs={"Yhat": torch.float}
    #   Yhat.shape[-2:]=(n_spatial, out_size)
    def forward(self, inputs):
        X, edge_indices, edge_weights = inputs["X"], inputs["edge_indices"], inputs.pop("edge_weights", None)
        if self.use_edge_weights:
            edge_weights = None
        if self.debug:
            print(util.make_msg_block("MultiHopGNN Forward"))
        if self.debug:
            print("X =", X.shape)
            print(util.memory_of(X))
        if self.debug:
            print("Edge Indices =", edge_indices.shape)
            print(util.memory_of(edge_indices))
        if self.debug and not edge_weights is None:
            print("Edge Weights =", edge_weights.shape)
            print(util.memory_of(edge_weights))
        # GNN layer(s) forward
        a, i = X, 0
        for name, layer in self.name_layer_map.items():
            if name.startswith("gnn"):
                a = self.gnn_forward(layer, a, edge_indices, edge_weights)
                if self.debug:
                    print("GNN %d-Hop Encoding =" % (i+1), a.shape)
                    print(util.memory_of(a))
                i += 1
            else:
                a = layer(a)
        outputs = {"Yhat": a}
        return outputs

    def default_forward(self, layer, X, edge_indices, edge_weights=None):
        if edge_weights is None:
            a = layer(X, edge_indices)
        else:
            a = layer(X, edge_indices, edge_weight=edge_weights)
        return a

    def FAConv_forward(self, layer, X, edge_indices, edge_weights=None):
        return layer(X, X, edge_indices, edge_weight=edge_weights)

    def reset_parameters(self):
        for name, layer in self.name_layer_map.items():
            if name.startswith("gnn"):
                layer.reset_parameters()

    def __str__(self):
        _str = ""
        _str += "%s(\n" % (self.layer_type)
        _str += "    in_size=%d\n" % (self.in_size)
        _str += "    out_size=%d\n" % (self.out_size)
        _str += "    hidden_size=%d\n" % (self.hidden_size)
        _str += "    n_hops=%d\n" % (self.n_hops)
        _str += "    hidden_activation=%s\n" % (self.hidden_activation)
        _str += "    dropout=%.2f\n" % (self.dropout)
        _str += ")"
        return _str


class TemporalEncoder(Model):

    def __init__(self, in_size, out_size, n_layers=1, layer_type="LSTM", dropout=0.0):
        super(TemporalEncoder, self).__init__()
        # Instantiate model layers
        self.supported_layers = ["GRU", "LSTM", "RNN"]
        assert layer_type in self.supported_layers, "Layer \"%s\" not supported" % (layer_type)
        self.layer_outhandler_map = {}
        for layer in self.supported_layers:
            self.layer_outhandler_map[layer] = self.default_outhandler
            if hasattr(self, "%s_outhandler" % (layer)):
                self.layer_outhandler_map[layer] = getattr(self, "%s_outhandler" % (layer))
        self.tmp_enc = self.layer_func_map["%sCell" % (layer_type)](in_size, out_size)
        self.outhandler = self.layer_outhandler_map[layer_type]
        # Save all vars
        self.in_size, self.out_size = in_size, out_size
        self.layer_type = layer_type

    # Preconditions:
    #   inputs={"X": torch.float, "n_steps": int}
    #   X.shape=(n_samples, n_temporal, n_predictors)
    # Postconditions:
    #   a.shape=(n_samples, n_temporal_out, n_responses)
    def forward(self, inputs):
        X, temporal_dim = inputs["X"], inputs.pop("temporal_dim", -2)
        assert len(X.shape) == 3
        prev_state, n_steps = inputs.pop("prev_state", None), inputs.pop("n_steps", X.shape[temporal_dim])
        if self.debug:
            print(util.make_msg_block("TemporalEncoder Forward"))
        if self.debug:
            print("X =", X.shape)
            print(util.memory_of(X))
#            print("prev_state =", prev_state)
#            print("n_steps =", n_steps)
        autoregressive = False
        if n_steps != X.shape[temporal_dim]: # encode autoregressively
            assert X.shape[temporal_dim] == 1, "Encoding a sequence from %d to %d time-steps is ambiguous" % (
                X.shape[temporal_dim], n_steps
            )
            autoregressive = True
        a, A = torch.transpose(X, 0, temporal_dim)[0], []
        for i in range(n_steps):
            if not autoregressive:
                a = torch.transpose(X, 0, temporal_dim)[i]
            output = self.tmp_enc(a, prev_state)
            a, prev_state = self.outhandler(output)
            A.append(a)
        a = torch.stack(A, temporal_dim)
        outputs = {}
        outputs["Yhat"] = a
        outputs["final_state"] = prev_state
        return outputs

    def default_outhandler(self, output):
        hidden_state = output
        return hidden_state, hidden_state

    def LSTM_outhandler(self, output):
        hidden_state, cell_state = output
        return hidden_state, (hidden_state, cell_state)

    def reset_parameters(self):
        self.tmp_enc.reset_parameters()


class TemporalMapper(Model):

    def __init__(self, in_size, out_size, method="last", kwargs={}):
        super(TemporalMapper, self).__init__()
        # Setup
        self.supported_methods = ["last", "last_n", "attention"]
        assert method in self.supported_methods, "Temporal mapping method \"%s\" not supported" % (method)
        self.method_init_map = {}
        for _method in self.supported_methods:
            self.method_init_map[_method] = getattr(self, "%s_init" % (_method))
        self.method_mapper_map = {}
        for _method in self.supported_methods:
            self.method_mapper_map[_method] = getattr(self, "%s_mapper" % (_method))
        # Instantiate method
        self.method_init_map[method](in_size, out_size, kwargs)
        self.mapper = self.method_mapper_map[method]

    def last_init(self, in_size, out_size, kwargs={}):
        pass

    def last_n_init(self, in_size, out_size, kwargs={}):
        pass

    def attention_init(self, in_size, out_size, kwargs={}):
        attention_kwargs = util.merge_dicts(
            {"num_heads": 1, "dropout": 0.0, "kdim": in_size, "vdim": in_size}, 
            kwargs
        )
        self.attn = self.layer_func_map["MultiheadAttention"](out_size, **attention_kwargs)

    # Preconditions:
    #   inputs={"X": torch.float, "n_temporal_out": int}
    #   X.shape=(n_samples, n_temporal_in, n_predictors)
    # Postconditions:
    #   a.shape=(n_samples, n_temporal_out, n_responses)
    def forward(self, inputs):
        if self.debug:
            print(util.make_msg_block("TemporalMapper Forward"))
        a = self.mapper(inputs)
        return {"Yhat": a}

    def last_mapper(self, inputs):
        X, temporal_dim = inputs["X"], inputs.pop("temporal_dim", -2)
        return torch.transpose(torch.transpose(X, 0, temporal_dim)[-1:], 0, temporal_dim)

    def last_n_mapper(self, inputs):
        X, n_temporal_out, temporal_dim = inputs["X"], inputs["n_temporal_out"], inputs.pop("temporal_dim", -2)
        return torch.transpose(torch.transpose(X, 0, temporal_dim)[-n_temporal_out:], 0, temporal_dim)

    def attention_mapper(self, inputs):
        n_temporal_out, temporal_dim = inputs["n_temporal_out"], inputs.pop("temporal_dim", -2)
        if "Q" in inputs:
            Q = inputs["Q"]
        else:
            Q = torch.transpose(inputs["X"], 0, temporal_dim)[-n_temporal_out:]
        if "K" in inputs:
            K = inputs["K"]
        else:
            K = torch.transpose(inputs["X"], 0, temporal_dim)
        if "V" in inputs:
            V = inputs["V"]
        else:
            V = torch.transpose(inputs["X"], 0, temporal_dim)
        if self.debug:
            print("Q =", Q.shape)
            print("K =", K.shape)
            print("V =", V.shape)
        a, w = self.attn(Q, K, V)
        return torch.transpose(a, 0, temporal_dim)

    def reset_parameters(self):
        if hasattr(self, "attn"):
            self.attn._reset_parameters()


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
