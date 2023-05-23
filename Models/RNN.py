import os
import sys
import torch

import Utility as util
from Container import Container
from Models.Model import Model, TemporalMapper
from Models.Model import RNNCell
from Models.Model import RNN_HyperparameterVariables
from Models.Model import TemporalMapper_HyperparameterVariables


class RNN(Model):

    def __init__(
        self,
        in_size,
        out_size,
        hidden_size=16,
        rnn_kwargs={},
        mapper_kwargs={}, 
    ):
        super(RNN, self).__init__()
        # Instantiate model layers
        self.enc = RNNCell(in_size, hidden_size, **rnn_kwargs)
        self.map = TemporalMapper(hidden_size, hidden_size, **mapper_kwargs)
        self.dec = RNNCell(hidden_size, hidden_size, **rnn_kwargs)
        self.out_proj = self.layer_fn_map["Linear"](hidden_size, out_size)
        self.out_proj_act = self.layer_fn_map["Identity"]()
        # Save all vars
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        # Pairs (name, partition) that identify this model
        self.id_pairs = [
            ["hidden_size", None],
            ["rnn_kwargs", None],
            ["mapper_kwargs", None],
        ]

    def forward(self, inputs):
#        self.debug = 1
        self.enc.debug = self.debug
        self.map.debug = self.debug
        self.dec.debug = self.debug
        x, n_temporal_out = inputs["x"], inputs["n_temporal_out"]
        n_samples, n_temporal_in, n_spatial, in_size = x.shape
        if self.debug:
            print(util.make_msg_block("RNN Forward"))
        if self.debug:
            print("x =", x.shape)
            print(util.memory_of(x))
        # Encoding layer forward
        a = torch.reshape(torch.transpose(x, 1, 2), (-1, n_temporal_in, in_size))
        if self.debug:
            print("x Reshaped =", a.shape)
            print(util.memory_of(a))
        self.enc.debug = self.debug
        a = self.enc(x=a)["yhat"]
        if self.debug:
            print("Encoding =", a.shape)
            print(util.memory_of(a))
        a = self.map(x=a, n_temporal_out=n_temporal_out, temporal_dim=-2)["yhat"]
        if self.debug:
            print("Encoding Remapped =", a.shape)
            print(util.memory_of(a))
        # Decoding layer forward
        a = self.dec(x=a, n_steps=n_temporal_out)["yhat"]
        if self.debug:
            print("Decoding =", a.shape)
            print(util.memory_of(a))
        a = torch.reshape(a, (n_samples, n_spatial, n_temporal_out, self.hidden_size))
        a = torch.transpose(a, 1, 2)
        if self.debug:
            print("Decoding Reshaped =", a.shape)
            print(util.memory_of(a))
        # Output layer forward
        z = self.out_proj(a)
        a = self.out_proj_act(z)
        if self.debug:
            print("Output =", a.shape)
            print(util.memory_of(a))
        if self.debug:
            sys.exit(1)
        outputs = {"yhat": a}
        return outputs

    def reset_parameters(self):
        self.enc.reset_parameters()
        self.map.reset_parameters()
        self.dec.reset_parameters()
        self.out_proj.reset_parameters()


def init(dataset, var):
    spatiotemporal = dataset.spatiotemporal
    hyp_var = var.models.get(model_name()).hyperparameters
    model = RNN(
        spatiotemporal.misc.n_predictor,
        spatiotemporal.misc.n_response,
        hyp_var.hidden_size,
        hyp_var.rnn_kwargs.to_dict(), 
        hyp_var.mapper_kwargs.to_dict(), 
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


class HyperparameterVariables(Container):

    def __init__(self):
        self.hidden_size = 16
        self.rnn_kwargs = RNN_HyperparameterVariables()
        self.rnn_kwargs.rnn_layer = "RNN"
        self.mapper_kwargs = TemporalMapper_HyperparameterVariables()


class TrainingVariables(Container):

    def __init__(self):
        self.initializer = None
