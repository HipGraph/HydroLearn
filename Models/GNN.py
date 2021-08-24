import torch
import torch_geometric as torch_g
import torch.distributed as torch_dist
import NerscDistributed as nerscdist
import sys
import numpy as np
import os
from progressbar import ProgressBar
import Utility as util
from Models.Model import Model
from Container import Container


class GNN(Model):

    def __init__(self, n_predictors, n_responses, temporal_encoding_size=128, spatial_encoding_size=128, temporal_decoding_size=128, n_temporal_layers=1, n_spatial_layers=1, spatial_layer_type="gcn", use_bias=True, dropout=0.0, bidirectional=True):
        super(GNN, self).__init__()
        # Instantiate model layers
        self.tmp_enc = torch.nn.LSTM(
            n_predictors,
            temporal_encoding_size,
            n_temporal_layers,
            bias=use_bias,
            batch_first=True,
            dropout=0.0,
            bidirectional=bidirectional
        )
        self.tmp_enc_drop = torch.nn.Dropout(dropout)
        enc_size = (2 * temporal_encoding_size if bidirectional else temporal_encoding_size) * n_temporal_layers
        self.spa_enc_lyrs = []
        self.spa_enc_drop_lyrs = []
        for i in range(n_spatial_layers):
            n_in, n_out = spatial_encoding_size, spatial_encoding_size
            if i == 0:
                n_in = enc_size
            self.__dict__["_modules"]["spa_enc_%d" % (i)] = self.gnnlayer_func_map[spatial_layer_type](
                n_in, 
                n_out, 
                node_dim=1
            )
            self.__dict__["_modules"]["spa_enc_drop_%d" % (i)] = torch.nn.Dropout(dropout)
            self.spa_enc_lyrs += [self.__dict__["_modules"]["spa_enc_%d" % (i)]]
            self.spa_enc_drop_lyrs += [self.__dict__["_modules"]["spa_enc_drop_%d" % (i)]]
        self.tmp_dec = torch.nn.LSTMCell(spatial_encoding_size, temporal_decoding_size, bias=use_bias)
        self.tmp_dec_drop = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(temporal_decoding_size, n_responses, bias=use_bias)
        # Save all vars
        self.n_predictors = n_predictors
        self.n_responses = n_responses
        self.tmp_enc_size = temporal_encoding_size
        self.spa_enc_size = spatial_encoding_size
        self.tmp_dec_size = temporal_decoding_size
        if True:
            self.config_name_partition_pairs += [
                ["similarity_measure", None],
                ["construction_method", None],
                ["self_loops", None],
                ["diffuse", None],
                ["diffusion_method", None],
                ["diffusion_sparsification", None]
            ]
        else:
            self.config_name_partition_pairs += [
                ["construction_method", None],
                ["similarity_measure", None]
            ]
        self.config_name_partition_pairs += [
            ["temporal_encoding_size", None],
            ["spatial_encoding_size", None],
            ["temporal_decoding_size", None],
            ["n_temporal_layers", None],
            ["n_spatial_layers", None],
            ["spatial_layer_type", None],
            ["use_bias", None],
            ["dropout", None],
            ["bidirectional", None]
        ]

    # Preconditions:
    #   x.shape=(n_samples, n_spatial, n_temporal_in, n_predictors)
    #   edge_indices.shape=(2, n_edges)
    # Postconditions:
    #   a.shape=(n_samples, n_spatial, n_temporal_out, n_responses)
    def forward(self, x, edges, n_temporal_out):
        n_samples, n_spatial, n_temporal_in, n_predictors = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        # Temporal encoding layer forward
        a = x.reshape(-1, n_temporal_in, n_predictors)
        z, (h_n, c_n) = self.tmp_enc(a)
        z = h_n.reshape(h_n.shape[1], -1).reshape(n_samples, n_spatial, -1)
        a = self.act_func_map["relu"](z)
        # Temporal encoding dropout
        a = self.tmp_enc_drop(a)
        # Spatial encoding layer(s) forward
        for spa_enc, spa_enc_drop in zip(self.spa_enc_lyrs, self.spa_enc_drop_lyrs):
            # Spatially encode
            z = spa_enc(a, edges)
            a = self.act_func_map["relu"](z)
            # Spatially drop
            a = spa_enc_drop(a)
        # Temporal decoding layer forward
        a, H = a.reshape(-1, self.spa_enc_size), []
        for i in range(n_temporal_out):
            (h_n, c_n) = (self.tmp_dec(a) if i == 0 else self.tmp_dec(a, (h_i, c_i)))
            H, h_i, c_i = H + [h_n], h_n, c_n
        z = torch.stack(H, dim=1).reshape(n_samples, n_spatial, n_temporal_out, self.tmp_dec_size)
        a = self.act_func_map["relu"](z)
        # Temporal decoding dropout
        a = self.tmp_dec_drop(a)
        # Output layer forward
        z = self.fc(a)
        a = self.act_func_map["identity"](z)
        return a

    # Preconditions:
    #   train/valid/test = [*_X, *_Y, edges]
    #   *_X.shape=(n_samples, n_temporal_in, n_spatial, n_predictors)
    #   *_Y.shape=(n_samples, n_temporal_out, n_spatial, n_responses)
    def optimize(self, train, valid=None, test=None, axes=[0, 1, 2, 3], lr=0.001, lr_decay=0.01, n_epochs=100, early_stop_epochs=10, mbatch_size=256, reg=0.0, loss="mse", opt="sgd", init="xavier", init_seed=-1, batch_shuf_seed=-1, n_procs=1, proc_rank=0, chkpt_epochs=1, chkpt_dir="Checkpoints", use_gpu=True):
        sample_axis, temporal_axis, spatial_axis, feature_axis = axes[0], axes[1], axes[2], axes[3]
        # Reformat data to accepted shape, convert to tensors, put data + model onto device, and unpack
        device = util.get_device(use_gpu)
        self = util.to_device([self], device)[0]
        train[:2] = util.move_axes(
            train[:2], 
            [sample_axis, temporal_axis, spatial_axis, feature_axis], 
            [0, 2, 1, 3]
        )
        torch_types = [torch.float, torch.float, torch.long, torch.long]
        train = util.to_device(util.to_tensor(train, torch_types), device)
        train_X, train_Y, train_spatial_indices, edges = train[0], train[1], train[2], train[3]
        if valid is not None:
            valid[:2] = util.move_axes(
                valid[:2], 
                [sample_axis, temporal_axis, spatial_axis, feature_axis], 
                [0, 2, 1, 3]
            )
            valid = util.to_device(util.to_tensor(valid, torch_types), device)
            valid_X, valid_Y, valid_spatial_indices = valid[0], valid[1], valid[2]
        if test is not None:
            test[:2] = util.move_axes(
                test[:2], 
                [sample_axis, temporal_axis, spatial_axis, feature_axis], 
                [0, 2, 1, 3]
            )
            test = util.to_device(util.to_tensor(test, torch_types), device)
            test_X, test_Y, test_spatial_indices = test[0], test[1], test[2]
        # Initialize loss, optimizer, and parameters
        self.crit = self.loss_func_map[loss]()
        self.opt = self.opt_func_map[opt](self.parameters(), lr=lr, weight_decay=reg)
        self.init_params(init, init_seed)
        # Commence optimization
        self.train_losses, self.valid_losses, self.test_losses = [], [], []
        n_mbatches = train_X.shape[0] // mbatch_size
        if batch_shuf_seed > -1:
            np.random.seed(batch_shuf_seed)
        min_valid_loss = sys.float_info.max
        for epoch in range(n_epochs+1):
            selection = np.random.choice(train_X.shape[0], size=train_X.shape[0], replace=False)
            train_loss = 0
            self.train()
            # Step through minibatches
            for mbatch in range(n_mbatches):
                start = mbatch * mbatch_size
                end = (mbatch + 1) * mbatch_size
                mbatch_loss = self.crit(
                    self(train_X[selection[start:end]], edges, train_Y.shape[2])[:,train_spatial_indices,:,:],
                    train_Y[selection[start:end]][:,train_spatial_indices,:,:]
                )
                # Gradient descent step
                self.opt.zero_grad()
                mbatch_loss.backward()
                self.opt.step()
                train_loss += mbatch_loss.item()
            train_loss /= n_mbatches
            self.train_losses += [train_loss]
            # Save current model
            if chkpt_epochs > 0 and epoch % chkpt_epochs == 0:
                path = chkpt_dir + os.sep + "Epoch[%d].pth" % (epoch)
                self.checkpoint(path)
            self.eval()
            # Print losses for this epoch
            print("Epoch %d : Train Loss = %.3f" % (epoch, train_loss))
            if valid is not None:
                valid_loss = self.crit(
                    self(valid_X, edges, valid_Y.shape[2])[:,valid_spatial_indices,:,:], 
                    valid_Y[:,valid_spatial_indices,:,:]
                ).item()
                self.valid_losses += [valid_loss]
                # Save best model
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    n_plateau_epochs = 0
                    path = chkpt_dir + os.sep + "Best.pth"
                    self.checkpoint(path)
                print("Epoch %d : Validation Loss = %.3f" % (epoch, valid_loss))
                # Check for loss plateau
                if valid_loss >= min_valid_loss:
                    n_plateau_epochs += 1
                    if early_stop_epochs > 0 and n_plateau_epochs % early_stop_epochs == 0:
                        break
            if test is not None:
                test_loss = self.crit(
                    self(test_X, edges, test_Y.shape[2])[:,test_spatial_indices,:,:], 
                    test_Y[:,test_spatial_indices,:,:]
                ).item()
                self.test_losses += [test_loss]
                print("Epoch %d : Test Loss = %.3f" % (epoch, test_loss))
                print("############################################################")
            # Decay learning rate
            if lr_decay > 0:
                for param_group in self.opt.param_groups:
                    param_group['lr'] = lr / (1 + lr_decay * epoch)
        # Save final model
        path = chkpt_dir + os.sep + "Final.pth"
        self.checkpoint(path)
        # Return data to original device (cpu), shape, and type (NumPy.ndarray)
        device = util.get_device(False)
        self = util.to_device([self], device)[0]
        train = util.to_ndarray(util.to_device(train, device))
        train[:2] = util.move_axes(train[:2], [temporal_axis, spatial_axis], [spatial_axis, temporal_axis])
        if valid is not None:
            valid = util.to_ndarray(util.to_device(valid, device))
            valid[:2] = util.move_axes(valid[:2], [temporal_axis, spatial_axis], [spatial_axis, temporal_axis])
        if test is not None:
            test = util.to_ndarray(util.to_device(test, device))
            test[:2] = util.move_axes(test[:2], [temporal_axis, spatial_axis], [spatial_axis, temporal_axis])

    # Preconditions:
    #   X.shape=(n_samples, n_temporal_in, n_spatial, n_predictors)
    # Postconditions: 
    #   Yhat.shape=(n_samples, n_temporal_out, n_spatial, n_responses)
    def predict(self, data, axes=[0, 1, 2, 3], mbatch_size=256, method="direct", use_gpu=True):
        X, indices, edges, n_temporal_out = data[0], data[1], data[2], data[3]
        sample_axis, temporal_axis, spatial_axis, feature_axis = axes[0], axes[1], axes[2], axes[3]
        n_samples, n_temporal_in = X.shape[sample_axis], X.shape[temporal_axis]
        n_spatial, n_predictors = X.shape[spatial_axis], X.shape[feature_axis]
        n_responses = self.n_responses
        Yhat = np.zeros([n_samples, n_spatial, n_temporal_out, n_responses])
        X = util.move_axes(
            [X], 
            [sample_axis, temporal_axis, spatial_axis, feature_axis], 
            [0, 2, 1, 3]
        )[0]
        device = util.get_device(use_gpu)
        self = util.to_device([self], device)[0]
        data = util.to_device(util.to_tensor([X, Yhat, edges], [torch.float, torch.float, torch.long]), device)
        X, Yhat, edges = data[0], data[1], data[2]
        self.eval()
        if method == "direct":
            n_mbatches = n_samples // mbatch_size
            indices = np.linspace(0, n_samples, n_mbatches+1, dtype=np.int)
            try:
                pb = ProgressBar()
                for i in pb(range(len(indices)-1)):
                    start, end = indices[i], indices[i+1]
                    Yhat[start:end] = self(X[start:end], edges, n_temporal_out)
            except RuntimeError as err:
                if "CUDA out of memory" in str(err):
                    msg = "CUDA ran out of memory. Will attempt on CPU." 
                    print(util.make_msg_block(msg, "*"))
                else:
                    raise RuntimeError(err)
                device = util.get_device(False)
                self = util.to_device([self], device)[0]
                data = util.to_device(data, device)
                X, Yhat, edges = data[0], data[1], data[2]
                pb = ProgressBar()
                for i in pb(range(len(indices)-1)):
                    start, end = indices[i], indices[i+1]
                    Yhat[start:end] = self(X[start:end], edges, n_temporal_out)
        elif method == "auto-regressive":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        device = util.get_device(False)
        self = util.to_device([self], device)[0]
        data = util.to_ndarray(util.to_device([X, Yhat, edges], device))
        data[:2] = util.move_axes(data[:2], [temporal_axis, spatial_axis], [spatial_axis, temporal_axis])
        X, Yhat, edges = data[0], data[1], data[2]
        return Yhat

    def init_params(self, init, seed=-1):
        if seed > -1:
            torch.manual_seed(seed)
        for name, param in self.named_parameters():
            if "bias" in name:
                self.init_func_map["constant"](param, 0.0)
            elif "weight" in name:
                if "spa" in name:
                    self.init_func_map["kaiming"](param)
                elif "tmp" in name:
                    self.init_func_map["xavier"](param)
                else:
                    self.init_func_map[init](param)


def init(var):
    model = GNN(
        var.get("n_predictors"),
        var.get("n_responses"),
        var.get("temporal_encoding_size"),
        var.get("spatial_encoding_size"),
        var.get("temporal_decoding_size"),
        var.get("n_temporal_layers"),
        var.get("n_spatial_layers"),
        var.get("spatial_layer_type"),
        var.get("use_bias"),
        var.get("dropout"),
        var.get("bidirectional"),
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


class HyperparameterVariables(Container):

    def __init__(self):
        n = 128
        ratios = [1.0, 1.0, 1.0]
        self.set("temporal_encoding_size", int(ratios[0]*n))
        self.set("spatial_encoding_size", int(ratios[1]*n))
        self.set("temporal_decoding_size", int(ratios[2]*n))
        self.set("n_temporal_layers", 1)
        self.set("n_spatial_layers", 1)
        self.set("spatial_layer_type", "sage")
        self.set("use_bias", True)
        self.set("dropout", 0.0)
        self.set("bidirectional", False)


def test():
    torch.manual_seed(1)
    n_samples, n_temporal_in, n_temporal_out, n_spatial, n_predictors, n_responses = 10, 4, 2, 5, 3, 1
    X = np.random.normal(size=(n_samples, n_temporal_in, n_spatial, n_predictors))
    Y = np.ones((n_samples, n_temporal_out, n_spatial, n_responses))
    edges = np.array([[0, 1, 1, 2, 1, 3, 3, 4], [1, 0, 2, 1, 3, 1, 4, 3]])
    print(X.shape, Y.shape)
    model = GNN(n_predictors, n_responses, 14, 7, 16, n_spa_lyrs=3, bidirectional=False)
    model.train()
    indices = np.arange(n_spatial)
    model.optimize([X, Y, indices, edges], mbatch_size=2, n_epochs=20)
    print(X.shape, Y.shape)
    Yhat = model.predict([X, indices, edges, n_temporal_out], mbatch_size=2)
    print(X.shape, Y.shape, Yhat.shape)
    print(np.mean((Y - Yhat)**2))

if __name__ == "__main__":
    test()
