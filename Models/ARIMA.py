import numpy as np
import sys
import os
import time
import hashlib
from progressbar import ProgressBar
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
    def forward(self, X, n_temporal_out, method="direct"):
        n_samples, n_temporal_in, n_spatial, n_predictors = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        A = np.zeros([n_samples, n_temporal_out, n_spatial, self.n_responses])
        if method == "direct":
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
        return A

    # Preconditions:
    #   train/valid/test = []
    def optimize(self, train, valid=None, test=None, axes=[0, 1, 2, 3], lr=0.001, lr_decay=0.01, n_epochs=100, early_stop_epochs=10, mbatch_size=256, reg=0.0, loss="mse", opt="sgd", init="xavier", init_seed=-1, batch_shuf_seed=-1, n_procs=1, proc_rank=0, chkpt_epochs=1, chkpt_dir="Checkpoints", use_gpu=True):
        self.train_losses, self.valid_losses, self.test_losses = [0], [0], [0]

    # Preconditions:
    #   data = [spatiotemporal_X, n_temporal_out]
    #   spatiotemporal_X.shape = (n_samples, n_temporal_in, n_spatial, n_predictors)
    def predict(self, data, mbatch_size=256, method="direct", use_gpu=True):
        X, n_temporal_out = data[0], data[1]
        n_samples, n_spatial, n_responses = X.shape[0], X.shape[2], self.n_responses
        Yhat = np.zeros([n_samples, n_temporal_out, n_spatial, n_responses])
        if method == "direct":
            n_mbatches = n_samples // mbatch_size
            indices = np.linspace(0, n_samples, n_mbatches+1, dtype=np.int)
            pb = ProgressBar()
            for i in pb(range(len(indices)-1)):
                start, end = indices[i], indices[i+1]
                Yhat[start:end] = self.forward(X[start:end], n_temporal_out)
        elif method == "auto-regressive":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        return Yhat

    def init_params(self, init, seed):
        pass

    def load(self, var, chkpt_path):
        return self


def init(dataset, var):
    spatmp = dataset.get("spatiotemporal")
    hyp_var = var.get("models").get(model_name()).get("hyperparameters")
    model = ARIMA(
        spatmp.get("mapping").get("n_responses"), 
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


def test():
    np.random.seed(0)
    n_samples, n_temporal_in, n_temporal_out, n_spatial, n_predictors, n_responses = 10, 8, 2, 5, 3, 1
    X = np.random.normal(size=(n_samples, n_temporal_in, n_spatial, n_predictors))
    Y = np.ones((n_samples, n_temporal_out, n_spatial, n_responses))
    print(X.shape, Y.shape)
    model = ARIMA(n_responses)
    model.optimize([X, Y], n_epochs=100, mbatch_size=5)
    print(X.shape, Y.shape)
    Yhat = model.predict([X, n_temporal_out], mbatch_size=3)
    print(X.shape, Y.shape)
    print(np.mean((Y - Yhat)**2))
    print(Y.shape)
    print(Y)
    print(Yhat.shape)
    print(Yhat)


if __name__ == "__main__":
    test()
