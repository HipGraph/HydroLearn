import numpy as np
import os
import itertools
from Experimentation.Experiment import Experiment


class ENVSOFT_Experiment(Experiment):

    def add_pre_subexp(self, args, subexp_id):
        args.set("n_temporal_in", 8)
        args.set("n_temporal_out", 1)
        args.set("temporal_reduction", ["avg", 7, 1])
        args.set("default_feature_transformation", ["min_max"])
        args.set("n_epochs", 100)
        args.set("optimizer", "Adadelta")
        args.set("lr", 0.1)
        if not "model" in args:
            args.set("model", "LSTM")
        if args.get("model") == "LSTM":
            args.set("encoding_size", 128)
            args.set("decoding_size", 128)
        if args.get("model") == "GEOMAN":
            args.set("n_hidden_encoder", 128)
            args.set("n_hidden_decoder", 128)
            args.set("n_stacked_layers", 1)
            args.set("dropout_rate", 0.0)
            args.set("regularization", 0.0)
        return args

    def checkpoint(self):
        return True

    def root_dir(self):
        return os.sep.join([__file__.replace(".py", ""), self.name()])


class Experiment1(ENVSOFT_Experiment):

    def name(self):
        return "SwatVsObserved"

    def subexp_ids(self):
        return np.arange(16)

    def add_subexp(self, args, subexp_id):
        # Add var for subexp
        model = "LSTM"
        partitions = ["train", "valid", "test"]
        datasets = ["wabashriver_swat", "wabashriver_swat", "wabashriver_observed"]
        spatial_labels = ["43", "169", "348", "529", "757"]
        temporal_selections = {}
        temporal_selections["wabashriver_swat"] = [
            ["interval", "1985-01-01", "1997-12-31"], 
            ["interval", "1998-01-01", "2005-12-31"], 
            ["interval", "2006-01-01", "2013-12-31"], 
        ]
        temporal_selections["wabashriver_observed"] = [
            ["interval", "1985-01-01", "1997-12-31"], 
            ["interval", "1998-01-01", "2005-12-31"], 
            ["interval", "2006-01-01", "2013-12-31"], 
        ]
        if subexp_id in range(1):
            model = "Identity"
            spatial_selection = ["literal"] + spatial_labels
            args.set("transform_features", False)
            args.set("partition_datasetasprediction_map", {"train": "train", "valid": "valid", "test": "train"})
        elif subexp_id in range(1, 6):
            spatial_selection = ["literal", spatial_labels[(subexp_id-1)%5]]
            datasets[0:2] = ["wabashriver_observed", "wabashriver_observed"]
        elif subexp_id in range(6, 11):
            spatial_selection = ["literal", spatial_labels[(subexp_id-6)%5]]
            datasets[0:2] = ["wabashriver_swat", "wabashriver_swat"]
        elif subexp_id in range(11, 16):
            spatial_selection = ["literal", spatial_labels[(subexp_id-11)%5]]
            temporal_selections["wabashriver_swat"][0] = ["interval", "1929-01-01", "1997-12-31"]
            datasets[0:2] = ["wabashriver_swat", "wabashriver_swat"]
        args.set("model", model)
        args.set("dataset", datasets, partitions, multi_value=True)
        args.set("spatial_selection", spatial_selection, partitions)
        for dataset in temporal_selections.keys():
            for temporal_selection, partition in zip(temporal_selections[dataset], partitions):
                args.set(
                    "temporal_selection", 
                    temporal_selection, 
                    partition, 
                    ["datasets", dataset, "spatiotemporal", "partitioning"]
                )
        # Add static var
        args.set("predictor_features", ["date", "tmin", "tmax", "PRECIPmm", "FLOW_OUTcms"])
        args.set("response_features", ["FLOW_OUTcms"])
        return args


class Experiment2(ENVSOFT_Experiment):

    def name(self):
        return "SwatTemporalInduction"

    def grid_argvalue_maps(self):
        grid_argvalue_maps = {
            "predictor_features": [
                ["date", "tmin", "tmax", "PRECIPmm", "FLOW_OUTcms", "SWmm"], 
                ["date", "tmin", "tmax", "PRECIPmm", "SWmm", "FLOW_OUTcms"]
            ], 
        }
        return grid_argvalue_maps

    def subexp_ids(self):
        return np.arange(1276)

    def add_subexp(self, args, subexp_id):
        # Add var for subexp
        partitions = ["train", "valid", "test"]
        datasets = ["wabashriver_swat", "wabashriver_swat", "wabashriver_swat"]
        args.set("spatial_selection", ["literal", str(subexp_id + 1)], partitions)
        # Add static var
        args.set("response_features", args.get("predictor_features")[-1:])
        args.set("dataset", datasets, partitions, multi_value=True)
        args.set("plot_model_fit", False)
        return args


class Experiment3(ENVSOFT_Experiment):

    def name(self):
        return "SwatSpatiotemporalInduction"

    def grid_argvalue_maps(self):
        grid_argvalue_maps = {
            "predictor_features": [
                ["date", "tmin", "tmax", "PRECIPmm", "FLOW_OUTcms", "SWmm"], 
                ["date", "tmin", "tmax", "PRECIPmm", "SWmm", "FLOW_OUTcms"]
            ], 
        }
        return grid_argvalue_maps

    def subexp_ids(self):
        return np.arange(20)

    def add_subexp(self, args, subexp_id):
        # Add var for subexp
        partitions = ["train", "valid", "test"]
        datasets = ["wabashriver_swat", "wabashriver_swat", "wabashriver_swat"]
        np.random.seed(1)
        n, r = 1276, 20
        labels = np.arange(n) + 1
        train_indices = sorted(np.random.choice(n, size=r, replace=False))
        train_labels = labels[train_indices]
        test_labels = sorted(np.delete(labels, train_indices)[np.random.choice(n-r, size=r, replace=False)])
        train_spatial_selection = ["literal", str(train_labels[subexp_id])]
        test_spatial_selection = ["literal"] + ",".join(map(str, test_labels))
        args.set("spatial_selection", train_spatial_selection, "train")
        args.set("spatial_selection", train_spatial_selection, "valid")
        args.set("spatial_selection", test_spatial_selection, "test")
        # Add static var
        args.set("response_features", args.get("predictor_features")[-1:])
        args.set("dataset", datasets, partitions, multi_value=True)
        return args


class Experiment4(ENVSOFT_Experiment):

    def name(self):
        return "SwatTransductionInductionAllModels"

    def grid_argvalue_maps(self):
        grid_argvalue_maps = {
            "model": ["NaiveLastTimestep", "ARIMA", "GEOMAN", "LSTM"], 
            "predictor_features": [
                ["date", "tmin", "tmax", "PRECIPmm", "FLOW_OUTcms", "SWmm"], 
                ["date", "tmin", "tmax", "PRECIPmm", "SWmm", "FLOW_OUTcms"]
            ], 
        }
        return grid_argvalue_maps

    def subexp_ids(self):
        return np.arange(6)

    def add_subexp(self, args, subexp_id):
        # Add var for subexp
        #   Transductive and inductive spatial selections are separate 
        #       (e.g. subexp_id 1 & 4 not combined into "literal,1,2") because GeoMAN requires spatial 
        #       element count be identical across partitions
        partitions = ["train", "valid", "test"]
        datasets = ["wabashriver_swat", "wabashriver_swat", "wabashriver_swat"]
        trans_spatial_labels = [1, 141, 1275]
        induc_spatial_labels = [2, 142, 1276]
        train_spatial_selection = ["literal", str(trans_spatial_labels[subexp_id % 3])]
        test_spatial_selection = train_spatial_selection
        if subexp_id in range(3, 6):
            test_spatial_selection = ["literal", str(induc_spatial_labels[subexp_id % 3])]
        args.set("spatial_selection", train_spatial_selection, ["train", "valid"])
        args.set("spatial_selection", test_spatial_selection, "test")
        # Add static var
        args.set("response_features", args.get("predictor_features")[-1:])
        args.set("dataset", datasets, partitions, multi_value=True)
        return args


class Experiment5(ENVSOFT_Experiment):

    def name(self):
        return "LittleRiverTemporalInduction"

    def subexp_ids(self):
        return np.arange(8)

    def add_subexp(self, args, subexp_id):
        # Add var for subexp
        partitions = ["train", "valid", "test"]
        datasets = ["littleriver_observed", "littleriver_observed", "littleriver_observed"]
        spatial_labels = ["B", "F", "I", "J", "K", "M", "N", "O"]
        spatial_selection = ["literal", spatial_labels[subexp_id]]
        args.set("spatial_selection", spatial_selection, partitions)
        # Add static var
        args.set("predictor_features", ["date", "tmin", "tmax", "PRECIPmm", "FLOW_OUTcms"])
        args.set("response_features", ["FLOW_OUTcms"])
        args.set("dataset", datasets, partitions, multi_value=True)
        return args


class Experiment6(ENVSOFT_Experiment):

    def name(self):
        return "SwatToLittleRiverSpatiotemporalInduction"

    def subexp_ids(self):
        return np.arange(16)

    def add_subexp(self, args, subexp_id):
        # Add var for subexp
        partitions = ["train", "valid", "test"]
        np.random.seed(0)
        wabash_spatial_labels = sorted(
            np.random.choice(1276, size=8, replace=False).astype(str)
        )
        little_spatial_labels = ["B", "F", "I", "J", "K", "M", "N", "O"]
        if subexp_id in range(8): # LittleRiver->LittleRiver temporal induction baseline tests
            datasets = ["littleriver_observed", "littleriver_observed", "littleriver_observed"]
            train_spatial_selection = ["literal", little_spatial_labels[subexp_id%8]]
            test_spatial_selection = train_spatial_selection
        else: # WabashRiver->LittleRiver spatiotemporal induction tests
            datasets = ["wabashriver_swat", "wabashriver_swat", "littleriver_observed"]
            train_spatial_selection = ["literal", wabash_spatial_labels[subexp_id%8]]
            test_spatial_selection = ["literal"] + little_spatial_labels 
        args.set(
            "spatial_selection", 
            train_spatial_selection, 
            partitions[:2], 
            ["datasets", datasets[0], "spatiotemporal", "partitioning"]
        )
        args.set(
            "spatial_selection", 
            test_spatial_selection, 
            partitions[-1], 
            ["datasets", datasets[-1], "spatiotemporal", "partitioning"]
        )
        # Add static var
        args.set("predictor_features", ["date", "tmin", "tmax", "PRECIPmm", "FLOW_OUTcms"])
        args.set("response_features", ["FLOW_OUTcms"])
        args.set("dataset", datasets, partitions, multi_value=True)
        return args


class Experiment7(ENVSOFT_Experiment):

    def name(self):
        return "LittleRiverToSwatSpatiotemporalInduction"

    def subexp_ids(self):
        return np.arange(16)

    def add_subexp(self, args, subexp_id):
        # Add var for subexp
        partitions = ["train", "valid", "test"]
        np.random.seed(0)
        wabash_spatial_labels = sorted(
            np.random.choice(1276, size=8, replace=False).astype(str)
        )
        little_spatial_labels = ["B", "F", "I", "J", "K", "M", "N", "O"]
        if subexp_id in range(8): # WabashRiver->WabashRiver temporal induction baseline tests
            datasets = ["wabashriver_swat", "wabashriver_swat", "wabashriver_swat"]
            train_spatial_selection = ["literal", wabash_spatial_labels[subexp_id%8]]
            test_spatial_selection = train_spatial_selection
        else: # LittleRiver->WabashRiver spatiotemporal induction tests
            datasets = ["littleriver_observed", "littleriver_observed", "wabashriver_swat"]
            train_spatial_selection = ["literal", little_spatial_labels[subexp_id%8]]
            test_spatial_selection = ["literal"] + wabash_spatial_labels 
        args.set(
            "spatial_selection", 
            train_spatial_selection, 
            partitions[:2], 
            ["datasets", datasets[0], "spatiotemporal", "partitioning"]
        )
        args.set(
            "spatial_selection", 
            test_spatial_selection, 
            partitions[-1], 
            ["datasets", datasets[-1], "spatiotemporal", "partitioning"]
        )
        # Add static var
        args.set("predictor_features", ["date", "tmin", "tmax", "PRECIPmm", "FLOW_OUTcms"])
        args.set("response_features", ["FLOW_OUTcms"])
        args.set("dataset", datasets, partitions, multi_value=True)
        return args


class Experiment8(ENVSOFT_Experiment):

    def name(self):
        return "PredictorRelevance"

    def grid_argvalue_maps(self):
        return {"response_features": [["SWmm"], ["FLOW_OUTcms"]]}

    def subexp_ids(self):
        return np.arange(len(self.get_features_set(["date", "tmin", "tmax", "PRECIPmm", "FLOW_OUTcms", "SWmm"])))

    def add_subexp(self, args, subexp_id):
        # Add var for subexp
        partitions = ["train", "valid", "test"]
        datasets = ["wabashriver_swat", "wabashriver_swat", "wabashriver_swat"]
        spatial_selection = ["literal", "1"]
        if args.get("response_features") == ["SWmm"]:
            predictor_features_set = self.get_features_set(
                ["date", "tmin", "tmax", "PRECIPmm", "FLOW_OUTcms", "SWmm"]
            )
        elif args.get("response_features") == ["FLOW_OUTcms"]:
            predictor_features_set = self.get_features_set(
                ["date", "tmin", "tmax", "PRECIPmm", "SWmm", "FLOW_OUTcms"]
            )
        else:
            raise ValueError()
        predictor_features = list(predictor_features_set[subexp_id])
        # Add static var
        args.set("spatial_selection", spatial_selection, partitions)
        args.set("predictor_features", predictor_features)
        args.set("dataset", datasets, partitions, multi_value=True)
        return args

    def get_features_set(self, features):
        n = len(features)
        features_set = []
        for k in range(1, n+1):
            features_set += list(itertools.combinations(features, k))
        return features_set


class Experiment9(ENVSOFT_Experiment):

    def name(self):
        return "SwatJointTransductInduction"

    def grid_argvalue_maps(self):
        grid_argvalue_maps = {
            "predictor_features": [
                ["date", "tmin", "tmax", "PRECIPmm", "FLOW_OUTcms", "SWmm"], 
                ["date", "tmin", "tmax", "PRECIPmm", "SWmm", "FLOW_OUTcms"]
            ], 
        }
        return grid_argvalue_maps

    def subexp_ids(self):
        return np.arange(1)

    def add_subexp(self, args, subexp_id):
        # Add var for subexp
        partitions = ["train", "valid", "test"]
        datasets = ["wabashriver_swat", "wabashriver_swat", "wabashriver_swat"]
        train_spatial_selection = ["literal", "1", "141", "1275"]
        test_spatial_selection = ["literal", "1", "141", "1275", "2", "142", "1276"]
        # Add static var
        args.set("spatial_selection", train_spatial_selection, partitions[:2])
        args.set("spatial_selection", test_spatial_selection, partitions[-1])
        args.set("response_features", args.get("predictor_features")[-1:])
        args.set("dataset", datasets, partitions, multi_value=True)
        return args


class Experiment10(ENVSOFT_Experiment):

    def name(self):
        return "SwatExtremeEventForecast"

    def grid_argvalue_maps(self):
        grid_argvalue_maps = {
            "predictor_features": [
                ["date", "tmin", "tmax", "PRECIPmm", "FLOW_OUTcms", "SWmm"], 
                ["date", "tmin", "tmax", "PRECIPmm", "SWmm", "FLOW_OUTcms"]
            ], 
        }
        return grid_argvalue_maps

    def subexp_ids(self):
        return np.arange(3)

    def add_subexp(self, args, subexp_id):
        # Add var for subexp
        partitions = ["train", "valid", "test"]
        datasets = ["wabashriver_swat", "wabashriver_swat", "wabashriver_swat"]
        trans_spatial_labels = ["1", "141", "1275"]
        induc_spatial_labels = ["2", "142", "1276"]
        train_spatial_selection = ["literal", trans_spatial_labels[subexp_id]]
        test_spatial_selection = ["literal", trans_spatial_labels[subexp_id], induc_spatial_labels[subexp_id]]
        args.set("spatial_selection", train_spatial_selection, partitions[:2])
        args.set("spatial_selection", test_spatial_selection, partitions[-1])
        # Add static var
        args.set("response_features", args.get("predictor_features")[-1:])
        args.set("dataset", datasets, partitions, multi_value=True)
        args.set(
            "options", 
            [
                "groundtruth", 
                "prediction", 
                "groundtruth_extremes", 
                "prediction_extremes", 
                "mean", 
                "stddev", 
                "confusion"
            ], 
            context=["plotting"]
        )
        return args


class Experiment11(ENVSOFT_Experiment):

    def name(self):
        return "TemporalReduction"

    def grid_argvalue_maps(self):
        grid_argvalue_maps = {
            "model": ["LSTM", "ARIMA", "GEOMAN"], 
            "predictor_features": [
                ["date", "tmin", "tmax", "PRECIPmm", "SWmm"], 
                ["date", "tmin", "tmax", "PRECIPmm", "FLOW_OUTcms"], 
            ], 
        }
        return grid_argvalue_maps

    def subexp_ids(self):
        self.temporal_reductions = self.get_temporal_reductions("avg", [1, 7, 14, 28])
        return np.arange(len(self.temporal_reductions))[:]

    def checkpoint(self):
#        return False
        return True

    def add_subexp(self, args, subexp_id):
        # Add var for subexp
        partitions = ["train", "valid", "test"]
        spatial_selection = ["literal", "529"]
        if args.get("predictor_features")[-1] == "SWmm": # Use SWAT (since only SWAT has soil water data)
            datasets = ["wabashriver_swat", "wabashriver_swat", "wabashriver_swat"] 
        else: # Use observed otherwise
            datasets = ["wabashriver_observed", "wabashriver_observed", "wabashriver_observed"] 
        args.set("temporal_reduction", self.temporal_reductions[subexp_id%len(self.temporal_reductions)])
        # Add static var
        if args.get("model") == "ARIMA":
            args.set("evaluated_partitions", ["test"])
        args.set("spatial_selection", spatial_selection, partitions)
        args.set("response_features", args.get("predictor_features")[-1:])
        args.set("dataset", datasets, partitions, multi_value=True)
        args.set("cache", True, None, ["evaluating"])
        return args

    def add_post_subexp(self, args, subexp_id):
        n_temporal_in, n_temporal_out = 224, 28
        temporal_reduction = args.get("temporal_reduction")
        """
        args.set("n_temporal_in", n_temporal_in // window_size)
        args.set("n_temporal_out", n_temporal_out // window_size)
        """
        window_size, window_stride = temporal_reduction[1:]
        args.set("temporal_resolution", [window_size, "days"])
        return args

    def get_temporal_reductions(self, reduce_op, window_sizes):
        temporal_reductions = []
        for w in window_sizes:
            for s in range(1, w+1):
                if w % s == 0:
                    temporal_reductions.append([reduce_op, w, s])
        return temporal_reductions


class Experiment12(ENVSOFT_Experiment):

    def name(self):
        return "LSTMvsARIMA"

    def grid_argvalue_maps(self):
        grid_argvalue_maps = {
            "model": ["LSTM", "ARIMA"], 
            "predictor_features": [
                ["date", "tmin", "tmax", "PRECIPmm", "SWmm"], 
                ["date", "tmin", "tmax", "PRECIPmm", "FLOW_OUTcms"], 
            ], 
        }
        return grid_argvalue_maps

    def subexp_ids(self):
        self.spatial_labels = ["43", "169", "348", "529", "757"]
        return np.arange(len(self.spatial_labels))[3:4]

    def checkpoint(self):
#        return False
        return True

    def add_subexp(self, args, subexp_id):
        # Add var for subexp
        partitions = ["train", "valid", "test"]
        spatial_selection = ["literal", self.spatial_labels[subexp_id]]
        if args.get("predictor_features")[-1] == "SWmm": # Use SWAT (since only SWAT has soil water data)
            datasets = ["wabashriver_swat", "wabashriver_swat", "wabashriver_swat"] 
        else: # Use observed otherwise
            datasets = ["wabashriver_observed", "wabashriver_observed", "wabashriver_observed"] 
        # Add static var
        if args.get("model") == "ARIMA":
            args.set("evaluated_partitions", ["test"])
        args.set("spatial_selection", spatial_selection, partitions)
        args.set("response_features", args.get("predictor_features")[-1:])
        args.set("dataset", datasets, partitions, multi_value=True)
        args.set("cache", True, None, ["evaluating"])
        return args
