import os
import inspect
from Models.Model import Model
from Container import Container


# Your very own model class. HydroLearn will instantiate this model and then call optimize() and/or predict()
class YourModel(Model):

    def __init__(self):
        func_name = inspect.currentframe().f_code.co_name
        raise NotImplementedError("Implement %s() please!" % (func_name))
        super(YourModel, self).__init__()
        # This list is used to generate a config file and 10-digit model ID.
        #   This resulting model ID is used for checkpointing and saving prediction results.
        #   Add variables that would uniquely define your model. Hyperparameters are a good choice.
        #       Items of this list are lists of size two that include the var name and partition it comes from
        #       Examples: ["n_hidden", None] or ["n_spatial", "train"]
        self.config_name_partition_pairs += []

    # Purpose:
    #   The forward pass of this model. Required if using PyTorch for back-propagation
    # Preconditions:
    #   inputs={"X": torch.float}
    #   X.shape=(n_samples, n_temporal_in, n_spatial, n_predictors)
    # Postconditions:
    #   a.shape=(n_samples, n_temporal_out, n_spatial, n_responses)
    def forward(self, inputs):
        func_name = inspect.currentframe().f_code.co_name
        raise NotImplementedError("Implement %s() please!" % (func_name))
        return a


# Pull items from the given dataset and/or init variables for model initialization
def init(dataset, var):
    func_name = currentframe().f_code.co_name
    raise NotImplementedError("Implement %s() please!" % (func_name))
    return model


# The name to be used for this model (important: will be used as a reference internally)
def model_name():
    return os.path.basename(__file__).replace(".py", "")


# Defines all hyperparameter variables for this model. These var are included in init() variable set "var" above
class HyperparameterVariables(Container):

    def __init__(self):
        func_name = self.__class__.__name__+"."+currentframe().f_code.co_name
        raise NotImplementedError("Implement %s() please!" % (func_name))


# Defines all training variables specific to this model. Final training vars sent to optimize() will
#   be the default vars (defined in Variables.py) merged with these vars
class TrainingVariables(Container):

    def __init__(self):
        func_name = self.__class__.__name__+"."+currentframe().f_code.co_name
        raise NotImplementedError("Implement %s() please!" % (func_name))
