import sys
import os
import torch
import numpy as np
import tensorflow as tf
from time import time
from progressbar import ProgressBar
import Utility as util
from Models.Model import Model
from Models.GeoMAN.GeoMAN import GeoMAN
from Container import Container


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class SlidingWindowDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.req_names = [
            "local_inputs", 
            "global_inputs", 
            "external_inputs", 
            "local_attn_states", 
            "global_attn_states", 
            "labels"
        ]
        self.data = data
        if not isinstance(self.data, dict):
            raise ValueError("Data must be a dictionary")
        for name in self.req_names:
            assert name in self.data, "Name \"%s\" not found in data dictionary" % (name)

    def __len__(self):
        return self.data["local_inputs"].shape[0]

    def __getitem__(self, key):
        mb = {
            "local_inputs": self.data["local_inputs"][key], 
            "global_inputs": self.data["global_inputs"][key], 
            "external_inputs": self.data["external_inputs"][key], 
            "local_attn_states": self.data["local_attn_states"][key], 
            "global_attn_states": self.data["global_attn_states"][key], 
            "labels": self.data["labels"][key]
        }
        return mb


class GEOMAN(Model):

    def __init__(self, n_predictors, n_responses, n_exogenous, n_temporal_in, n_temporal_out, n_spatial, n_hidden_encoder=64, n_hidden_decoder=64, n_stacked_layers=2, s_attn_flag=2, dropout_rate=0.3):
        super(GEOMAN, self).__init__()
        # Hyperparameters
        self.hps = tf.contrib.training.HParams(
            # GPU parameters
            gpu_id="0",
            # Model parameters
            learning_rate=1e-3,
            lambda_l2_reg=1e-3,
            gc_rate=2.5,
            dropout_rate=dropout_rate,
            n_stacked_layers=n_stacked_layers,
            s_attn_flag=s_attn_flag,
            ext_flag=True,
            # Encoder parameters
            n_sensors=n_spatial,
            n_input_encoder=n_predictors,
            n_steps_encoder=n_temporal_in,
            n_hidden_encoder=n_hidden_encoder,
            # Decoder parameters
            n_input_decoder=1,
            n_external_input=n_exogenous,
            n_steps_decoder=n_temporal_out,
            n_hidden_decoder=n_hidden_decoder,
            n_output_decoder=n_responses
        )
        # Model construction
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        session = tf.Session(config=tf_config)
        tf.reset_default_graph()
        self.model = GeoMAN(self.hps)
        self.sess = tf.Session()
        self.model.init(self.sess)
        self.saver = tf.train.Saver(max_to_keep=10000)
        # Configuration names
        self.config_name_partition_pairs += [
            ["n_hidden_encoder", None],
            ["n_hidden_decoder", None],
            ["n_stacked_layers", None],
            ["s_attn_flag", None],
            ["dropout_rate", None]
        ]
        # Other variables
#        self.LoaderDatasetClass = GeoMANDataset

    # Preconditions:
    #   inputs={"local_inputs": ndarray, ...}
    def forward(self, inputs):
#        self.debug = 1
        # Reformat the data and convert to ndarrays
        inputs = self.format_inputs(inputs)
        for key in inputs.keys():
            inputs[key] = util.to_ndarray(inputs[key])
        # Unpack the data
        local_inputs, global_inputs = inputs["local_inputs"], inputs["global_inputs"]
        external_inputs, local_attn_states = inputs["external_inputs"], inputs["local_attn_states"]
        global_attn_states, labels = inputs["global_attn_states"], inputs["labels"]
        n_spatial = local_inputs.shape[2]
        # Setup for forward
        fetch_names = ["loss"]
        if self.training:
            fetch_names.append("train_op")
        else:
            fetch_names.append("preds")
        fetches = [self.model.phs[fetch_name] for fetch_name in fetch_names]
        outputs = {fetch_name: [] for fetch_name in fetch_names}
        # Forward for each spatial element
        for s in range(n_spatial):
            feeds = {
                self.model.phs["local_inputs"]: local_inputs[:,:,s],
                self.model.phs["global_inputs"]: global_inputs,
                self.model.phs["external_inputs"]: external_inputs[:,:,s],
                self.model.phs["local_attn_states"]: local_attn_states[:,s],
                self.model.phs["global_attn_states"]: global_attn_states,
                self.model.phs["labels"]: labels[:,:,s]
            }
            fetched = self.sess.run(fetches, feeds)
            if self.debug and False:
                print(fetched)
            for fetch_name, fetch in zip(fetch_names, fetched):
                outputs[fetch_name].append(fetch)
        # Process outputs
        outputs["loss"] = sum(outputs["loss"]) / n_spatial
        if "preds" in outputs:
            outputs["preds"] = util.to_tensor(np.swapaxes(np.stack(outputs["preds"], axis=2), 0, 1), torch.float)
            outputs["Yhat"] = outputs["preds"]
        if "train_op" in outputs:
            outputs["train_op"] = sum(outputs["train_op"]) / n_spatial
        if self.debug:
            print("Loss =", outputs["loss"])
            if "Yhat" in outputs:
                print("Yhat =", outputs["Yhat"].shape)
            sys.exit(1)
        return outputs

    def pull_data(self, dataset, partition, var):
        data = {"E": None}
        if not dataset.is_empty():
            data["X"] = dataset.spatiotemporal.transformed.reduced.get("predictor_features", partition)
            data["Y"] = dataset.spatiotemporal.transformed.reduced.get("response_features", partition)
            if not dataset.spatial.is_empty():
                data["E"] = dataset.spatial.transformed.get("numerical_features", partition)
            else:
                data["E"] = np.zeros((self.hps.n_sensors, self.hps.n_external_input))
#            data["n_temporal_out"] = var.temporal_mapping[1]
        return data

    def step(self, loss, var):
        pass

    def loss(self, mb_in, mb_out):
        return mb_out["loss"]

    def loss_to_numeric(self, loss):
        return loss

    def criterion(self, var):
        pass

    def optimizer(self, var):
        self.model.hps.learning_rate = var.lr
        self.model.hps.lambda_l2_reg = var.regularization
        self.model.hps.gc_rate = var.gc_rate

    def update_lr(self, lr):
        self.model.hps.learning_rate = lr

    def prepare_model(self, use_gpu, revert=False):
        return self
    
    # Preconditions:
    #   data = {"X: torch.float, "Y": torch.float, "E": torch.float}
    #   X.shape = (n_samples, n_temporal_in, n_spatial, n_predictors)
    #   Y.shape = (n_samples, n_temporal_out, n_spatial, n_responses)
    #   E.shape = (n_spatial, n_exogeneous)
    def format_inputs(self, data):
        # ==========================
        # === GeoMAN Data Format ===
        # ==========================
        #
        # Local Inputs: Windowed time-series of predictors from a single sensor
        #   shape=(n_samples, n_steps_encoder, n_input_encoder)
        #               ||
        #               \/
        #   shape=(n_samples, n_temporal_in, n_predictors)
        #
        # Global Inputs:
        #   shape=(n_samples, n_steps_encoder, n_sensors)
        #               ||
        #               \/
        #   shape=(n_samples, n_temporal_in, n_spatial)
        #
        # External Inputs: Inputs outside of the temporal domain (maybe area, lat, lon)
        #   shape=(n_samples, n_steps_decoder, n_external_input)
        #               ||
        #               \/
        #   shape=(n_samples, n_temporal_out, n_exogenous)
        #
        # Local Attention States:
        #   shape=(n_samples, n_input_encoder, n_steps_encoder)
        #               ||
        #               \/
        #   shape=(n_samples, n_predictors, n_temporal_in)
        #
        # Global Attention States:
        #   shape=(n_samples, n_sensors, n_input_encoder, n_steps_encoder)
        #               ||
        #               \/
        #   shape=(n_samples, n_spatial, n_predictors, n_temporal_in)
        #
        # Labels:
        #   shape=(n_samples, n_steps_decoder, n_output_decoder)
        #               ||
        #               \/
        #   shape=(n_samples, n_temporal_out, n_responses)
        #
        X, Y, E, n_temporal_out = data["X"], data.pop("Y", None), data.pop("E", None), self.hps.n_steps_decoder
        n_samples, n_temporal_in, n_spatial, n_predictors = X.shape
        n_responses, n_exogenous = self.hps.n_output_decoder, self.hps.n_external_input
        if not Y is None:
            assert Y.shape[1] == self.hps.n_steps_decoder, "Output time-steps of \"Y\" (%d) is inconsistent with output time-steps of GeoMAN (%d)" % (Y.shape[1], self.hps.n_steps_decoder)
            assert Y.shape[2] == self.hps.n_sensors, "Spatial element count of \"Y\" ($d) is inconsistent with spatial element count of GeoMAN (%d)" % (Y.shape[2], self.hps.n_sensors)
        if not E is None:
            assert E.shape[0] == self.hps.n_sensors, "Spatial element count of \"E\" (%d) is inconsistent with spatial element count of GeoMAN (%d)" % (E.shape[0], self.hps.n_sensors)
            assert E.shape[-1] == self.hps.n_external_input, "Feature count of \"E\" (%d) is inconsistent with external feature count of GeoMAN (%d)" % (E.shape[-1], self.hps.n_external_input)
        local_inputs = X
        if self.debug:
            print("Local Inputs =", local_inputs.shape)
        global_inputs = torch.mean(X, -1)
        if self.debug:
            print("Global Inputs =", global_inputs.shape)
        if E is None:
            external_inputs = torch.zeros((n_samples, n_temporal_out, n_spatial, n_exogenous))
        else:
#            external_inputs = torch.tile(E, (n_samples, n_temporal_out, n_spatial, 1))
            external_inputs = E.repeat((n_samples, n_temporal_out, n_spatial, 1))
        if self.debug:
            print("External Inputs =", external_inputs.shape)
        local_attn_states = util.move_axes(X, [0, 1, 2, 3], [0, 3, 1, 2])
        if self.debug:
            print("Local Attention States =", local_attn_states.shape)
        global_attn_states = util.move_axes(X, [0, 1, 2, 3], [0, 3, 1, 2])
        if self.debug:
            print("Global Attention States =", global_attn_states.shape)
        if Y is None:
            labels = torch.zeros((n_samples, n_temporal_out, n_spatial, n_responses))
        else:
            labels = Y
        if self.debug:
            print("Labels =", labels.shape)
        data = {
            "local_inputs": local_inputs, 
            "global_inputs": global_inputs, 
            "external_inputs": external_inputs, 
            "local_attn_states": local_attn_states, 
            "global_attn_states": global_attn_states, 
            "labels": labels
        }
        return data

    def log_compgraph(self, train, train_loader, valid, valid_loader, test, test_loader, epoch, var):
        pass

    # Notes:
    #   This function attempts to induce determinism for optimization but testing appears to show that GeoMAN uses non-deterministic tensorflow operations internally. 
    #   Non-deterministic operations may also come from cuDNN used in GPU training as described by Kilian Batzner here: 
    #       https://stackoverflow.com/questions/53396670/unable-to-reproduce-tensorflow-results-even-after-setting-random-seed
    def init_params(self, init, seed=-1):
        np.random.seed((seed if seed > -1 else time()))
        tf.random.set_random_seed((seed if seed > -1 else time()))
        self.model.init(self.sess)

    def load(self, var, path):
        chkpt_dir = os.sep.join(path.split(os.sep)[:-1])
        chkpt_name = path.split(os.sep)[-1]
        req_exts = [".index", ".meta", ".data-00000-of-00001"]
        req_paths = [path+req_ext for req_ext in req_exts] + [chkpt_dir+os.sep+"checkpoint"]
        if all([os.path.exists(req_path) for req_path in req_paths]):
            self.saver.restore(self.sess, path)
        else:
            raise FileNotFoundError(
                "Some checkpoint files were missing from \"%s\". Required checkpoint files include: \"%s\"" % (
                    path,
                    ", ".join([req_path.replace(chkpt_dir+os.sep, "") for req_path in req_paths])
                )
            )
        return self

    def checkpoint(self, path):
        path = path.replace(".pth", "")
        self.saver.save(self.sess, path)

    def n_params(self):
        return sum(np.prod(var.get_shape()) for var in tf.trainable_variables())


def init(dataset, var):
    spatmp = dataset.get("spatiotemporal")
    spa = dataset.get("spatial")
    hyp_var = var.get("models").get(model_name()).get("hyperparameters")
    n_exogenous = 0
    if not spa.is_empty():
        n_exogenous = spa.get("transformed").get("n_numerical_features")
    model = GEOMAN(
        spatmp.get("misc").get("n_predictors"),
        spatmp.get("misc").get("n_responses"),
        n_exogenous,
        spatmp.get("mapping").get("temporal_mapping")[0],
        spatmp.get("mapping").get("temporal_mapping")[1],
        spatmp.get("original").get("n_spatial", "train"),
        hyp_var.get("n_hidden_encoder"),
        hyp_var.get("n_hidden_decoder"),
        hyp_var.get("n_stacked_layers"),
        hyp_var.get("s_attn_flag"),
        hyp_var.get("dropout_rate")
    )
    return model


def model_name():
    return os.path.basename(__file__).replace(".py", "")


class HyperparameterVariables(Container):
    
    def __init__(self):
        self.set("n_hidden_encoder", 64)
        self.set("n_hidden_decoder", 64)
        self.set("n_stacked_layers", 2)
        self.set("s_attn_flag", 2)
        self.set("dropout_rate", 0.3)


class TrainingVariables(Container):
    
    def __init__(self):
        self.set("lr", 1e-3)
        self.set("regularization", 1e-3)
        self.set("gc_rate", 2.5)
        self.set("use_gpu", False)
