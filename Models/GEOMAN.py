import sys
import os
import torch
import torch.distributed as dist
import numpy as np
from time import time
import hashlib
from progressbar import ProgressBar
import Utility as util
from Models.Model import Model
from Models.GeoMAN.GeoMAN import GeoMAN
from Container import Container
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class GEOMAN(Model):

    def __init__(self, n_predictors, n_responses, n_spatial, n_temporal_in, n_temporal_out, n_exogenous, n_hidden_encoder=128, n_hidden_decoder=128, n_stacked_layers=1, s_attn_flag=2, dropout_rate=0.0):
        # Hyperparameters
        self.hps = tf.contrib.training.HParams(
            # GPU parameters
            gpu_id="0",
            # Model parameters
            learning_rate=0.001,
            lambda_l2_reg=0.0,
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
        # model construction
        self.name
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

    # Preconditions:
    def forward(self, data, indices, fetches):
        n_spatial = len(data[0])
        results = [[] for fetch in fetches]
        for s in range(n_spatial):
            feeds = {
                self.model.phs["local_inputs"]: data[0][s][indices],
                self.model.phs["global_inputs"]: data[1][indices],
                self.model.phs["external_inputs"]: data[2][s][indices],
                self.model.phs["local_attn_states"]: data[3][s][indices],
                self.model.phs["global_attn_states"]: data[4][indices],
                self.model.phs["labels"]: data[5][s][indices]
            }
            fetched = self.sess.run(fetches, feeds)
            for i in range(len(fetched)):
                results[i] += [fetched[i]]
        return results

    # Preconditions:
    #   train/valid/test = [spatiotemporal_X, spatiotemporal_Y, spatial]
    #   spatiotemporal_X.shape = (n_samples, n_temporal_in, n_spatial, n_predictors)
    #   spatiotemporal_Y.shape = (n_samples, n_temporal_out, n_spatial, n_responses)
    #   spatial.shape = (n_spatial, n_exogeneous)
    def optimize(self, train, valid=None, test=None, axes=[0, 1, 2, 3], lr=0.001, lr_decay=0.01, n_epochs=100, early_stop_epochs=10, mbatch_size=256, reg=0.0, loss="mse", opt="sgd", init="xavier", init_seed=-1, batch_shuf_seed=-1, n_procs=1, proc_rank=0, chkpt_epochs=1, chkpt_dir="Checkpoints", use_gpu=True):
        train = self.prepare_data(train)
        if not valid is None:
            valid = self.prepare_data(valid)
        if not test is None:
            test = self.prepare_data(test)
        # Initialize parameters
        self.init_params(init, init_seed)
        # Commence optimization
        self.train_losses, self.valid_losses, self.test_losses = [], [], []
        n_samples = train[0][0].shape[0]
        n_mbatches = n_samples // mbatch_size
        if batch_shuf_seed > -1:
            np.random.seed(batch_shuf_seed)
        min_valid_loss = sys.float_info.max
        for epoch in range(n_epochs+1):
            selection = np.random.choice(n_samples, size=n_samples, replace=False)
            train_loss = 0
            # Step through minibatches
            for mbatch in range(n_mbatches):
                start, end = mbatch * mbatch_size, (mbatch + 1) * mbatch_size
                results = self.forward(train, selection[start:end], [self.model.phs["train_op"]])
                train_loss += np.mean(results[0])
            train_loss /= n_mbatches
            self.train_losses += [train_loss]
            # Save current model
            if chkpt_epochs > 0 and epoch % chkpt_epochs == 0:
                path = util.path([chkpt_dir, "Epoch[%d]" % (epoch)])
                self.checkpoint(path)
            # Print losses for this epoch
            print("Epoch %d : Train Loss = %.3f" % (epoch, train_loss))
            if valid is not None:
                results = self.forward(valid, np.arange(valid[0][0].shape[0]), [self.model.phs["loss"]])
                valid_loss = np.mean(results[0])
                self.valid_losses += [valid_loss]
                # Save best model
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    n_plateau_epochs = 0
                    path = util.path([chkpt_dir, "Best"])
                    self.checkpoint(path)
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
                results = self.forward(test, np.arange(test[0][0].shape[0]), [self.model.phs["loss"]])
                test_loss = np.mean(results[0])
                self.test_losses += [test_loss]
                print("Epoch %d : Test Loss = %.3f" % (epoch, test_loss))
                print("############################################################")
            # Decay learning rate
            if lr_decay > 0:
                self.hps.learning_rate = lr / (1 + lr_decay * epoch)
        # Save final model
        path = util.path([chkpt_dir, "Final"])
        self.checkpoint(path)

    # Preconditions:
    #   data = [spatiotemporal_X, spatial]
    #   spatiotemporal_X.shape = (n_samples, n_temporal_in, n_spatial, n_predictors)
    #   spatial.shape = (n_spatial, n_exogenous)
    # Postconditions: 
    #   Yhat.shape = (n_samples, n_temporal_out, n_spatial, n_responses)
    def predict(self, data, mbatch_size=256, method="direct", use_gpu=True):
        X, E = data[0], data[1]
        n_samples, n_temporal_in, n_spatial, n_predictors  = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        n_temporal_out, n_responses = self.hps.n_steps_decoder, self.hps.n_output_decoder
        Yhat = np.zeros([n_samples, n_temporal_out, n_spatial, n_responses])
        data = self.prepare_data([X, None, E])
        if method == "direct":
            n_mbatches = n_samples // mbatch_size
            indices = np.linspace(0, n_samples, n_mbatches+1, dtype=np.int)
            pb = ProgressBar()
            for i in pb(range(len(indices)-1)):
                start, end = indices[i], indices[i+1]
                results = self.forward(data, np.arange(start, end), [self.model.phs["preds"]])
                Yhat[start:end] = util.move_axes([np.stack(results[0])], [0, 1, 2, 3], [2, 1, 0, 3])[0]
        elif method == "auto-regressive":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        return Yhat
    
    # Preconditions:
    #   data = [spatiotemporal_X, spatiotemporal_Y, spatial]
    #   spatiotemporal_X.shape = (n_samples, n_temporal_in, n_spatial, n_predictors)
    #   spatiotemporal_Y.shape = (n_samples, n_temporal_out, n_spatial, n_responses)
    #   spatial.shape = (n_spatial, n_exogeneous)
    def prepare_data(self, data):
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
        X, Y, E = data[0], data[1], data[2]
        n_samples, n_temporal_in, n_spatial, n_predictors = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        n_temporal_out, n_responses = self.hps.n_steps_decoder, self.hps.n_output_decoder 
        n_exogenous = self.hps.n_external_input
        if not E is None and E.shape[0] != n_spatial:
            raise ValueError()
        if not Y is None and Y.shape[2] != n_spatial:
            raise ValueError()
        local_inputs = [X[:,:,s,:] for s in range(n_spatial)]
        global_inputs = np.mean(X, axis=3)
        if E is None:
            external_inputs = np.zeros((n_samples, n_temporal_out, n_exogenous))
            external_inputs = [external_inputs for s in range(n_spatial)]
        else:
            external_inputs = [np.tile(E[s,:], (n_samples, n_temporal_out, 1)) for s in range(n_spatial)]
        local_attention_states = [
            util.move_axes(X[:,:,s,:], [0, 1, 2], [0, 2, 1]) for s in range(n_spatial)
        ]
        global_attention_states = util.move_axes(X, [0, 1, 2, 3], [0, 3, 1, 2])
        if Y is None:
            labels = np.zeros((n_samples, n_temporal_out, n_responses))
            labels = [labels for s in range(n_spatial)]
        else:
            labels = [Y[:,:,s,:] for s in range(n_spatial)]
        data = [
            local_inputs, 
            global_inputs, 
            external_inputs, 
            local_attention_states, 
            global_attention_states, 
            labels
        ]
        return data

    # Notes:
    #   This function attempts to induce determinism for optimization but testing appears to show that GeoMAN uses non-deterministic tensorflow operations internall. 
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
        self.saver.save(self.sess, path)


def init(dataset, var):
    spatmp = dataset.get("spatiotemporal")
    spa = dataset.get("spatial")
    hyp_var = var.get("models").get(model_name()).get("hyperparameters")
    model = GEOMAN(
        spatmp.get("mapping").get("n_predictors"),
        spatmp.get("mapping").get("n_responses"),
        spatmp.get("original").get("original_n_spatial", "train"),
        spatmp.get("mapping").get("n_temporal_in"),
        spatmp.get("mapping").get("n_temporal_out"),
        spat.get("original").get("original_n_exogenous"),
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
        n = 128
        ratios = [1.0, 1.0]
        self.set("n_hidden_encoder", int(ratios[0]*n))
        self.set("n_hidden_decoder", int(ratios[1]*n))
        self.set("n_stacked_layers", 1)
        self.set("s_attn_flag", 2)
        self.set("dropout_rate", 0.0)


def test():
    torch.manual_seed(1)
    n_samples, n_temporal_in, n_temporal_out, n_spatial, n_predictors, n_responses, n_exogenous = 10, 4, 2, 5, 3, 1, 2
    X = np.random.normal(size=(n_samples, n_temporal_in, n_spatial, n_predictors))
    Y = np.ones((n_samples, n_temporal_out, n_spatial, n_responses))
    E = np.ones((n_spatial, n_exogenous))
    print(X.shape, Y.shape, E.shape)
    model = GEOMAN(n_predictors, n_responses, n_spatial, n_temporal_in, n_temporal_out, n_exogenous)
    model.optimize([X, Y, E], mbatch_size=5, n_epochs=100)
    print(X.shape, Y.shape, E.shape)
    Yhat = model.predict([X, E], mbatch_size=3)
    print(X.shape, Y.shape, E.shape)
    print("MSE =", np.mean((Y - Yhat)**2))


if __name__ == "__main__":
    test()
