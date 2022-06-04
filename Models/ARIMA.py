import numpy as np
import os
import torch
import Utility as util
from Models.Model import Model
from statsmodels.tsa.arima.model import ARIMA as arima
from Container import Container


# SOURCE: https://machinelearningmastery.com/save-arima-time-series-forecasting-model-python/
# monkey patch around bug in ARIMA class
def __getnewargs__(self):
    return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
arima.__getnewargs__ = __getnewargs__


class ARIMA(Model):

    def __init__(self, n_responses, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0), trend=None, enforce_stationarity=True, enforce_invertability=True, concentrate_scale=False, trend_offset=1):
        self.n_responses, self.order, self.seasonal_order = n_responses, order, seasonal_order
        self.trend, self.enforce_stationarity = trend, enforce_stationarity
        self.enforce_invertability, self.concentrate_scale = enforce_invertability, concentrate_scale
        self.trend_offset = trend_offset
        self.config_name_partition_pairs += [
            ["order", None],
            ["seasonal_order", None],
            ["trend", None],
            ["enforce_stationarity", None],
            ["enforce_invertability", None],
            ["concentrate_scale", None],
            ["trend_offset", None],
        ]

    # Preconditions:
    #   X.shape=(n_samples, n_temporal_in, n_spatial, n_predictors)
    # Postconditions:
    #   A.shape=(n_samples, n_temporal_out, n_spatial, n_responses)
    def forward(self, inputs):
        self.debug = 1
        X, n_temporal_out = inputs["X"], inputs["n_temporal_out"]
        n_samples, n_temporal_in, n_spatial, n_predictors = X.shape
        if self.debug:
            print(util.make_msg_block("ARIMA Forward"))
        if self.debug:
            print("X =", X.shape)
            print(util.memory_of(X))
        X, A = util.to_ndarray(X), np.zeros((n_samples, n_temporal_out, n_spatial, self.n_responses))
        for i in range(n_samples):
            for j in range(n_spatial):
                for k in range(self.n_responses):
                    self.model = arima(
                        X[i,:,j,n_predictors-self.n_responses+k], 
                        None,
                        self.order,
                        self.seasonal_order,
                        self.trend,
                        self.enforce_stationarity,
                        self.enforce_invertability,
                        self.concentrate_scale,
                        self.trend_offset
                    ).fit()
                    A[i,:,j,k] = self.model.forecast(n_temporal_out)
        if self.debug:
            print("Output =", A.shape)
            print(util.memory_of(A))
        if self.debug:
            sys.exit(1)
        outputs = {"Yhat": util.to_tensor(A, torch.float)}
        return outputs

    def optimize(self, train_dataset, valid_dataset, test_dataset, var):
        pass

    def prepare_model(self, use_gpu, revert=False):
        return self

    def load(self, var, chkpt_path):
        return self

    def n_params(self):
        return -1

    def train(self):
        pass

    def eval(self):
        pass


def init(dataset, var):
    spatmp = dataset.get("spatiotemporal")
    hyp_var = var.get("models").get(model_name()).get("hyperparameters")
    model = ARIMA(
        spatmp.get("misc").get("n_responses"), 
        hyp_var.get("order"), 
        hyp_var.get("seasonal_order"), 
        hyp_var.get("trend"), 
        hyp_var.get("enforce_stationarity"), 
        hyp_var.get("enforce_invertability"), 
        hyp_var.get("trend_offset")
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


class HyperparameterVariables(Container):

    def __init__(self):
        self.set("order", [0, 0, 0])
        self.set("seasonal_order", [0, 0, 0, 0])
        self.set("trend", None)
        self.set("enforce_stationarity", True)
        self.set("enforce_invertability", True)
        self.set("concentrate_scale", False)
        self.set("trend_offset", 1)


class TrainingVariables(Container):
    
    def __init__(self):
        self.set("use_gpu", False)
