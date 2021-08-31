import os
import torch
import torch.distributed as dist
import numpy as np
from sys import float_info
from time import time
from progressbar import ProgressBar
import Utility as util
from Models.Model import Model
from Container import Container


# Your very own model class. HydroLearn will instantiate this model and then call optimize() and/or predict()
class YourModel(Model):

    def __init__(self):
        raise NotImplementedError()
        super(YourModel, self).__init__()
        # This list is used to generate a config file and 10-digit model ID.
        #   This resulting model ID is used for checkpointing and saving prediction results.
        self.config_name_partition_pairs += [ 
            # Add variables that would uniquely define your model. Hyperparameters are a good choice.
            #   Items of this list are lists of size two that include the var name and partition it comes from
            #   Examples: ["n_hidden", None] or ["n_spatial", "train"]
        ]

    # The forward pass of this model. This method only required if using PyTorch for back-propagation
    def forward(self, x):
        raise NotImplementedError()
        return a

    # Optimizes model parameters given a training dataset and a set of optimization arguments
    def optimize(self, train_dataset, valid_dataset=None, test_dataset=None, lr=0.001, lr_decay=0.01, n_epochs=100, early_stop_epochs=10, mbatch_size=256, reg=0.0, loss="mse", opt="sgd", init="xavier", init_seed=-1, batch_shuf_seed=-1, n_procs=1, proc_rank=0, chkpt_epochs=1, chkpt_dir="Checkpoints", use_gpu=True):
        raise NotImplementedError()

    # Predict values given a dataset as input.
    def predict(self, dataset, mbatch_size=256, method="direct", use_gpu=True):
        raise NotImplementedError()
        return Yhat


# Pull items from the given dataset and/or init variables for model initialization
def init(dataset, var):
    raise NotImplementedError()
    return model


# The name to be used for this model (important: will be used as a reference internally)
def model_name():
    return os.path.basename(__file__).replace(".py", "")


# Defines all hyperparameters for this model. These will be included in init variable set "var" above in init()
class HyperparameterVariables(Container):

    def __init__(self):
        raise NotImplementedError()
        

if __name__ == "__main__":
    pass
