import os
import sys
import torch
import torch.distributed as dist
import Utility as util
from Models.Model import Model, TemporalEncoder, TemporalMapper
from Container import Container


class LSTM(Model):

    def __init__(
        self, 
        n_predictors, 
        n_responses, 
        encoding_size=16, 
        decoding_size=16, 
        n_layers=1, 
        dropout=0.0, 
        temporal_mapper="last", 
    ):
        super(LSTM, self).__init__()
        # Instantiate model layers
        if 0:
            self.enc = self.layer_func_map["LSTM"](
                n_predictors, 
                encoding_size, 
                n_layers, 
                batch_first=True, 
                dropout=0.0
            )
        else:
            self.enc = TemporalEncoder(n_predictors, encoding_size, n_layers, "LSTM", dropout)
        self.enc_drop = self.layer_func_map["Dropout"](dropout)
        self.tmp_map = TemporalMapper(encoding_size, encoding_size, temporal_mapper)
        self.dec = TemporalEncoder(encoding_size, decoding_size, n_layers, "LSTM", dropout)
        self.dec_drop = self.layer_func_map["Dropout"](dropout)
        self.out_proj = self.layer_func_map["Linear"](decoding_size, n_responses)
        # Save all vars
        self.n_predictors, self.n_responses = n_predictors, n_responses
        self.enc_size, self.dec_size = encoding_size, decoding_size
        self.config_name_partition_pairs += [
            ["encoding_size", None],
            ["decoding_size", None],
            ["n_layers", None],
            ["dropout", None],
            ["temporal_mapper", None], 
        ]

    # Preconditions:
    #   inputs={"X": torch.float, "n_temporal_out": int}
    #   X.shape=(n_samples, n_temporal_in, n_spatial, n_predictors)
    # Postconditions:
    #   a.shape=(n_samples, n_temporal_out, n_spatial, n_responses)
    def forward(self, inputs):
#        self.debug = 1
        self.tmp_map.debug = self.debug
        self.dec.debug = self.debug
        X, n_temporal_out = inputs["X"], inputs["n_temporal_out"]
        n_samples, n_temporal_in, n_spatial, n_predictors = X.shape
        if self.debug:
            print(util.make_msg_block("LSTM Forward"))
        if self.debug:
            print("X =", X.shape)
            print(util.memory_of(X))
        # Encoding layer forward
        a = torch.reshape(torch.transpose(X, 1, 2), (-1, n_temporal_in, n_predictors))
        if self.debug:
            print("X Reshaped =", a.shape)
            print(util.memory_of(a))
        if 0:
            a, (h_n, c_n) = self.enc(a)
            a = h_n
            a = self.enc_drop(a)
            if self.debug:
                print("Encoding =", a.shape)
                print(util.memory_of(a))
            a = torch.reshape(torch.transpose(a, 0, 1), (a.shape[1], -1))
            a = torch.unsqueeze(a, -2)
        else:
            self.enc.debug = self.debug
            a = self.enc({"X": a})["Yhat"]
            a = self.enc_drop(a)
            if self.debug:
                print("Encoding =", a.shape)
                print(util.memory_of(a))
        a = self.tmp_map({"X": a, "n_temporal_out": n_temporal_out, "temporal_dim": -2})["Yhat"]
        if self.debug:
            print("Encoding Remapped =", a.shape)
            print(util.memory_of(a))
        # Decoding layer forward
        a = self.dec({"X": a, "n_steps": n_temporal_out})["Yhat"]
        a = self.dec_drop(a)
        if self.debug:
            print("Decoding =", a.shape)
            print(util.memory_of(a))
        a = torch.reshape(a, (n_samples, n_spatial, n_temporal_out, self.dec_size))
        a = torch.transpose(a, 1, 2)
        if self.debug:
            print("Decoding Reshaped =", a.shape)
            print(util.memory_of(a))
        # Output layer forward
        z = self.out_proj(a)
        a = self.act_func_map["identity"](z)
        if self.debug:
            print("Output =", a.shape)
            print(util.memory_of(a))
        if self.debug:
            sys.exit(1)
        outputs = {"Yhat": a}
        return outputs

    def reset_parameters(self):
        self.enc.reset_parameters()
        self.tmp_map.reset_parameters()
        self.dec.reset_parameters()
        self.out_proj.reset_parameters()


def init(dataset, var):
    spatmp = dataset.get("spatiotemporal")
    hyp_var = var.get("models").get(model_name()).get("hyperparameters")
    model = LSTM(
        spatmp.get("misc").get("n_predictors"),
        spatmp.get("misc").get("n_responses"),
        hyp_var.get("encoding_size"),
        hyp_var.get("decoding_size"),
        hyp_var.get("n_layers"),
        hyp_var.get("dropout"),
        hyp_var.get("temporal_mapper"),
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


class HyperparameterVariables(Container):

    def __init__(self):
        n = 16
        ratios = [1.0, 1.0]
        self.set("encoding_size", int(ratios[0]*n))
        self.set("decoding_size", int(ratios[1]*n))
        self.set("n_layers", 1)
        self.set("dropout", 0.0)
        self.set("temporal_mapper", "last")


class TrainingVariables(Container):

    def __init__(self):
        self.set("initializer", "xavier_normal_")
        self.set("initializer", None)
