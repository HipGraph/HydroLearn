import os
import numpy as np
import sys
from progressbar import ProgressBar
import Utility as util
from Models.Model import Model
from Container import Container


class NaiveLastTimestep(Model):

    def __init__(self, n_responses):
        self.n_responses = n_responses

    # Preconditions:
    #   x.shape=(n_samples, n_temporal_in, n_spatial, n_predictors)
    # Postconditions:
    #   a.shape=(n_samples, n_temporal_out, n_spatial, n_responses)
    def forward(self, x, n_temporal_out):
        a = np.tile(x[:,-1:,:,-self.n_responses:], [1, n_temporal_out, 1, 1])
        return a

    # Preconditions:
    #   train/valid/test = [*_X, *_Y]
    #   *_X.shape=(n_samples, n_temporal_in, n_spatial, n_predictors)
    #   *_Y.shape=(n_samples, n_temporal_out, n_spatial, n_responses)
    def optimize(self, train, valid=None, test=None, axes=[0, 1, 2, 3], lr=0.001, lr_decay=0.01, n_epochs=100, early_stop_epochs=10, mbatch_size=256, reg=0.0, loss="mse", opt="sgd", init="xavier", init_seed=-1, batch_shuf_seed=-1, n_procs=1, proc_rank=0, chkpt_epochs=1, chkpt_dir="Checkpoints", use_gpu=True):
        self.train_losses, self.valid_losses, self.test_losses = [0], [0], [0]

    # Preconditions:
    #   data = [X, n_temporal_out]
    #   axes = [sample_axis, temporal_axis, spatial_axis, feature_axis]
    # Postconditions:
    #   Yhat.shape[?] == X.shape[?] for ? in {sample_axis, temporal_axis, spatial_axis}
    def predict(self, data, axes=[0, 1, 2, 3], mbatch_size=256, method="direct", use_gpu=True):
        X, n_temporal_out = data[0], data[1]
        sample_axis, temporal_axis, spatial_axis, feature_axis = axes[0], axes[1], axes[2], axes[3]
        n_samples, n_spatial, n_responses = X.shape[sample_axis], X.shape[spatial_axis], self.n_responses
        X = util.move_axes([X], axes, [0, 1, 2, 3])[0]
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
        data = util.move_axes([X, Yhat], [0, 1, 2, 3], axes)
        X, Yhat = data[0], data[1]
        return Yhat

    def init_params(self, init, seed):
        pass

    def load(self, var, chkpt_path):
        return self


def init(dataset, var):
    spatmp = dataset.get("spatiotemporal")
    model = NaiveLastTimestep(
        spatmp.get("mapping").get("n_responses")
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


class HyperparameterVariables(Container):

    def __init__(self):
        pass


def test():
    np.random.seed(0)
    n_samples, n_temporal_in, n_temporal_out, n_spatial, n_predictors, n_responses = 10, 4, 2, 6, 3, 1
    X = np.random.normal(size=(n_samples, n_temporal_in, n_spatial, n_predictors))
    Y = np.ones((n_samples, n_temporal_out, n_spatial, n_responses))
    print(X.shape, Y.shape)
    model = NaiveLastTimestep(n_responses)
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
