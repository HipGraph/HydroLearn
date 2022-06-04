import os
from Models.Model import Model
from Container import Container


class NaiveLastTimestep(Model):

    def __init__(self, n_responses):
        super(NaiveLastTimestep, self).__init__()
        self.n_responses = n_responses

    # Preconditions:
    #   inputs={"X": torch.float, "n_temporal_out": int}
    #   X.shape=(n_samples, n_temporal_in, n_spatial, n_predictors)
    # Postconditions:
    #   a.shape=(n_samples, n_temporal_out, n_spatial, n_responses)
    def forward(self, inputs):
        X, n_temporal_out = inputs["X"], inputs["n_temporal_out"]
        return {"Yhat": X[:,-1:,:,-self.n_responses:].repeat((1, n_temporal_out, 1, 1))}

    def optimize(self, train_dataset, valid_dataset, test_dataset, var):
        self.train_losses, self.valid_losses, self.test_losses = [0], [0], [0]

    def load(self, var, chkpt_path):
        return self


def init(dataset, var):
    spatmp = dataset.get("spatiotemporal")
    model = NaiveLastTimestep(
        spatmp.get("misc").get("n_responses")
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


class HyperparameterVariables(Container):

    def __init__(self):
        pass
