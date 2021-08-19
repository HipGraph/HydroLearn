import torch
import torch.distributed as dist
import numpy as np
from sys import float_info
from time import time
from progressbar import ProgressBar
import Utility as util
from Models.Model import Model


class LSTM(Model):

    def __init__(self, n_predictors, n_responses, encoding_size=128, decoding_size=128, n_layers=1, use_bias=True, dropout=0.0, bidirectional=True, output_activation="identity"):
        super(LSTM, self).__init__()
        # Instantiate model layers
        self.enc = self.layer_func_map["lstm"](
            n_predictors,
            encoding_size,
            n_layers,
            bias=use_bias,
            batch_first=True,
            dropout=0.0,
            bidirectional=bidirectional
        )
        self.enc_drop = self.layer_func_map["dropout"](dropout)
        hidden_size = (2 * encoding_size if bidirectional else encoding_size) * n_layers
        self.dec = self.layer_func_map["lstm_cell"](hidden_size, decoding_size, bias=use_bias)
        self.dec_drop = self.layer_func_map["dropout"](dropout)
        self.fc = self.layer_func_map["linear"](decoding_size, n_responses, bias=use_bias)
        # Save all vars
        self.n_predictors, self.n_responses = n_predictors, n_responses
        self.enc_size, self.dec_size = encoding_size, decoding_size
        self.out_act = output_activation
        self.config_name_partition_pairs += [
            ["encoding_size", None],
            ["decoding_size", None],
            ["n_layers", None],
            ["use_bias", None],
            ["dropout", None],
            ["bidirectional", None],
        ]

    # Preconditions:
    #   x.shape=(n_samples, n_spatial, n_temporal_in, n_predictors)
    # Postconditions:
    #   a.shape=(n_samples, n_spatial, n_temporal_out, n_responses)
    def forward(self, x, n_temporal_out):
        n_samples, n_spatial, n_temporal_in, n_predictors = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        # Encoding layer forward
        a = x.reshape(-1, n_temporal_in, n_predictors)
        z, (h_n, c_n) = self.enc(a)
        z = h_n.reshape(h_n.shape[1], -1)
        a = self.act_func_map["relu"](z)
        # Encoding dropout forward
        a = self.enc_drop(a)
        # Decoding layer forward
        H = []
        for i in range(n_temporal_out):
            (h_n, c_n) = (self.dec(a) if i == 0 else self.dec(a, (h_i, c_i)))
            H, h_i, c_i = H + [h_n], h_n, c_n
        z = torch.stack(H, dim=1).reshape(n_samples, n_spatial, n_temporal_out, self.dec_size)
        a = self.act_func_map["relu"](z)
        # Decoding dropout forward
        a = self.dec_drop(a)
        # Output layer forward
        z = self.fc(a)
        a = self.act_func_map[self.out_act](z)
        return a

    # Preconditions:
    #   train/valid/test = [spatiotemporal_X, spatiotemporal_Y]
    #   spatiotemporal_X.shape = (n_samples, n_temporal_in, n_spatial, n_predictors)
    #   spatiotemporal_Y.shape = (n_samples, n_temporal_out, n_spatial, n_responses)
    def optimize(self, train, valid=None, test=None, lr=0.001, lr_decay=0.01, n_epochs=100, early_stop_epochs=10, mbatch_size=256, reg=0.0, loss="mse", opt="sgd", init="xavier", init_seed=-1, batch_shuf_seed=-1, n_procs=1, proc_rank=0, chkpt_epochs=1, chkpt_dir="Checkpoints", use_gpu=True):
        # Reformat data to accepted shape, convert to tensors, put data + model onto device, then unpack
        self, train_X, train_Y = self.prepare(train, use_gpu)
        if valid is not None:
            self, valid_X, valid_Y = self.prepare(valid, use_gpu)
        if test is not None:
            self, test_X, test_Y = self.prepare(test, use_gpu)
        # Initialize loss, optimizer, and parameters
        self.crit = self.loss_func_map[loss]()
        self.opt = self.opt_func_map[opt](self.parameters(), lr=lr, weight_decay=reg)
        self.init_params(init, init_seed)
        # Commence optimization
        self.train_losses, self.valid_losses, self.test_losses = [], [], []
        n_mbatches = train_X.shape[0] // mbatch_size
        if batch_shuf_seed > -1:
            np.random.seed(batch_shuf_seed)
        min_valid_loss = float_info.max
        for epoch in range(n_epochs+1):
            selection = np.random.choice(train_X.shape[0], size=train_X.shape[0], replace=False)
            train_loss = 0
            self.train()
            # Step through minibatches
            for mbatch in range(n_mbatches):
                start = mbatch * mbatch_size
                end = (mbatch + 1) * mbatch_size
                mbatch_loss = self.crit(
                    self(train_X[selection[start:end]], train_Y.shape[2]),
                    train_Y[selection[start:end]]
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
                path = util.path([chkpt_dir, "Epoch[%d].pth" % (epoch)])
                self.checkpoint(path)
            self.eval()
            # Print losses for this epoch
            print("Epoch %d : Train Loss = %.3f" % (epoch, train_loss))
            if valid is not None:
                valid_loss = self.crit(self(valid_X, valid_Y.shape[2]), valid_Y).item()
                self.valid_losses += [valid_loss]
                # Save best model
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    n_plateau_epochs = 0
                    path = util.path([chkpt_dir, "Best.pth"])
                    self.checkpoint(path)
                    elapsed = time() - start
                print("Epoch %d : Validation Loss = %.3f" % (epoch, valid_loss))
                # Check for loss plateau
                if valid_loss >= min_valid_loss:
                    n_plateau_epochs += 1
                    if early_stop_epochs > 0 and n_plateau_epochs % early_stop_epochs == 0:
                        break
                else:
                    min_valid_loss = valid_loss
                    n_plateau_epochs = 0
            if test is not None:
                test_loss = self.crit(self(test_X, test_Y.shape[2]), test_Y).item()
                self.test_losses += [test_loss]
                print("Epoch %d : Test Loss = %.3f" % (epoch, test_loss))
                print("############################################################")
            # Decay learning rate
            if lr_decay > 0:
                for param_group in self.opt.param_groups:
                    param_group['lr'] = lr / (1 + lr_decay * epoch)
        # Save final model
        path = util.path([chkpt_dir, "Final.pth"])
        self.checkpoint(path)
        # Return data to original device (cpu), shape, and type (NumPy.ndarray)
        self, train_X, train_Y = self.prepare([train_X, train_Y], False, True)
        if valid is not None:
            self, valid_X, valid_Y = self.prepare([valid_X, valid_Y], False, True)
        if test is not None:
            self, test_X, test_Y = self.prepare([test_X, test_Y], False, True)

    # Preconditions:
    #   data = [spatiotemporal_X, n_temporal_out]
    #   spatiotemporal_X.shape = (n_samples, n_temporal_in, n_spatial, n_predictors)
    def predict(self, data, mbatch_size=256, method="direct", use_gpu=True):
        X, n_temporal_out = data[0], data[1]
        n_samples, n_temporal_in, n_spatial, n_predictors = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        n_responses = self.n_responses
        Yhat = np.zeros([n_samples, n_temporal_out, n_spatial, n_responses])
        self, X, Yhat = self.prepare([X, Yhat], use_gpu)
        self.eval()
        if method == "direct":
            n_mbatches = n_samples // mbatch_size
            indices = np.linspace(0, n_samples, n_mbatches+1, dtype=np.int)
            try:
                pb = ProgressBar()
                for i in pb(range(len(indices)-1)):
                    start, end = indices[i], indices[i+1]
                    Yhat[start:end] = self(X[start:end], n_temporal_out)
            except RuntimeError as err:
                if "CUDA out of memory" in str(err):
                    msg = "CUDA ran out of memory. Will attempt on CPU." 
                    print(util.make_msg_block(msg, "*"))
                else:
                    raise RuntimeError(err)
                device = util.get_device(False)
                self = util.to_device(self, device)
                X, Yhat = util.to_device([X, Yhat], device)
                pb = ProgressBar()
                for i in pb(range(len(indices)-1)):
                    start, end = indices[i], indices[i+1]
                    Yhat[start:end] = self(X[start:end], n_temporal_out)
        elif method == "auto-regressive":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        self, X, Yhat = self.prepare([X, Yhat], use_gpu, True)
        return Yhat

    # Precondition(s):
    #   data.type = list
    #   spatiotemporal_X.shape = (n_samples, n_temporal_in, n_spatial, n_predictors)
    #   spatiotemporal_Y.shape = (n_samples, n_temporal_out, n_spatial, n_responses)
    def prepare_data(self, data, use_gpu, revert=False):
        if revert:
            device = util.get_device(False)
            data = util.move_axes(util.to_ndarray(util.to_device(data, device)), [1, 2], [2, 1])
        else:
            device = util.get_device(use_gpu)
            data = util.to_device(
                util.to_tensor(
                    util.move_axes(data, [1, 2], [2, 1]), 
                    [torch.float, torch.float]
                ), 
                device
            )
        return data


def init(var):
    model = LSTM(
        var.get("n_predictors"),
        var.get("n_responses"),
        var.get("encoding_size"),
        var.get("decoding_size"),
        var.get("n_layers"),
        var.get("use_bias"),
        var.get("dropout"),
        var.get("bidirectional"),
        var.get("output_activation")
    )
    return model


def test():
    torch.manual_seed(1)
    n_samples, n_temporal_in, n_temporal_out, n_spatial, n_predictors, n_responses = 10, 4, 1, 5, 3, 1
    X = np.random.normal(size=(n_samples, n_temporal_in, n_spatial, n_predictors))
    Y = np.ones((n_samples, n_temporal_out, n_spatial, n_responses))
    print(X.shape, Y.shape)
    model = LSTM(n_predictors, n_responses, 14, bidirectional=False)
    model.train()
    model.optimize([X, Y], mbatch_size=5)
    print(X.shape, Y.shape)
    Yhat = model.predict([X, n_temporal_out], mbatch_size=2)
    print(X.shape, Y.shape)
    print(np.mean((Y - Yhat)**2))
    print(Y.shape)
    print(Y)
    print(Yhat.shape)
    print(Yhat)


if __name__ == "__main__":
    test()
