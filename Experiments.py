import numpy as np
import os
import sys
import Driver as driver
from Arguments import ArgumentBuilder


def main():
    if len(sys.argv) < 2:
        raise ValueError("Must give experiments to run (e.g. \"123\")")
    expIDs, mode, n_processes = [x for x in sys.argv[1]], sys.argv[2], 1
    expID_exp_map = {}
    try:
        for i in range(100):
            expID_exp_map[str(i)] = globals()["Experiment%s" % (i)]()
    except:
        pass
    mode_args = ArgumentBuilder().add_mode_args(mode)
    for expID in expIDs:
        exp = expID_exp_map[expID]
        for model in exp.models[:]:
            for predictor_features, response_features in zip(exp.predictor_features, exp.response_features):
                for subexpID in exp.subexpIDs[:]:
                    args = ArgumentBuilder().copy(mode_args)
                    args = add_base_args(
                        args, 
                        model, 
                        exp.data_sources, 
                        predictor_features, 
                        response_features
                    )
                    args = exp.add_args(args, subexpID)
                    print(args.to_string())
                    if experiment_completed(args) and False:
                        print("Experiment Already Completed. Moving on...")
                        continue
                    driver.invoke(args, n_processes)
                    cache_experiment(args)


def experiment_completed(args):
    path = args.get_argvalue("evaluation_dir") + os.sep + "CompletedExperiments.txt"
    if not os.path.exists(path):
        return False
    with open(path, "r") as f:
        completed_experiments = f.read()
    return args.to_string() in completed_experiments


def cache_experiment(args):
    path = args.get_argvalue("evaluation_dir") + os.sep + "CompletedExperiments.txt"
    with open(path, "a") as f:
        f.write(args.to_string())


class Experiment0:

    name = "SwatVsObserved"
    models = ["LSTM"]
    predictor_features = ["date,tmin,tmax,PRECIPmm,FLOW_OUTcms"]
    response_features = ["FLOW_OUTcms,"]
    subexpIDs = np.arange(15) + 1
    data_sources = ["historical", "historical", "historical"]

    def __init__(self): pass

    def add_args(self, args, subexpID): # train and test on all subbasins one at a time
        spatial_labels = [43, 169, 348, 529, 757]
        spatial_selection = "literal,%d" % (spatial_labels[(subexpID-1)%5])
        temporal_selections = [
            "interval,1985-01-01,1997-12-31",
            "interval,1998-01-01,2005-12-31",
            "interval,2006-01-01,2013-12-31",
        ]
        if subexpID / 5 <= 1:
            data_sources = ["observed", "observed", "observed"]
        elif subexpID / 5 <= 2:
            data_sources = ["historical", "historical", "observed"]
        elif subexpID / 5 <= 3:
            temporal_selections = [
                "interval,1929-01-01,1997-12-31",
                "interval,1998-01-01,2005-12-31",
                "interval,2006-01-01,2013-12-31",
            ]
            data_sources = ["historical", "historical", "observed"]
        args.add("spatial_selection", spatial_selection, "train")
        args.add("spatial_selection", spatial_selection, "valid")
        args.add("spatial_selection", spatial_selection, "test")
        args.add("temporal_selection", temporal_selections[0], "train", [data_sources[0]])
        args.add("temporal_selection", temporal_selections[1], "valid", [data_sources[1]])
        args.add("temporal_selection", temporal_selections[2], "test", [data_sources[2]])
        args.replace("data_source", data_sources[0], "train")
        args.replace("data_source", data_sources[1], "valid")
        args.replace("data_source", data_sources[2], "test")
        args.replace("evaluation_dir", "Evaluations"+os.sep+self.name)
        args.replace("checkpoint_dir", "Checkpoints"+os.sep+self.name)
        return args


class Experiment1:

    name = "SwatTemporalInduction"
    models = ["LSTM"]
    predictor_features = ["date,tmin,tmax,PRECIPmm,FLOW_OUTcms,SWmm", "date,tmin,tmax,PRECIPmm,SWmm,FLOW_OUTcms"]
    response_features = ["SWmm,", "FLOW_OUTcms,"]
    subexpIDs = np.arange(1276) + 1
    data_sources = ["historical", "historical", "historical"]

    def __init__(self): pass

    def add_args(self, args, subexpID): # train and test on all subbasins one at a time
        spatial_selection = "literal,%d" % (subexpID)
        args.add("spatial_selection", spatial_selection, "train")
        args.add("spatial_selection", spatial_selection, "valid")
        args.add("spatial_selection", spatial_selection, "test")
#        args.add("plot_model_fit", False)
        if "SWmm" in args.get_argvalue("response_features"):
            args.replace("evaluation_dir", "Evaluations"+os.sep+self.name+os.sep+"SW")
            args.replace("checkpoint_dir", "Checkpoints"+os.sep+self.name+os.sep+"SW")
        elif "FLOW_OUTcms" in args.get_argvalue("response_features"):
            args.replace("evaluation_dir", "Evaluations"+os.sep+self.name+os.sep+"SF")
            args.replace("checkpoint_dir", "Checkpoints"+os.sep+self.name+os.sep+"SF")
        else:
            raise ValueError()
        return args


class Experiment2:

    name = "SwatSpatiotemporalInduction"
    models = ["LSTM"]
    predictor_features = ["date,tmin,tmax,PRECIPmm,FLOW_OUTcms,SWmm", "date,tmin,tmax,PRECIPmm,SWmm,FLOW_OUTcms"]
    response_features = ["SWmm,", "FLOW_OUTcms,"]
    data_sources = ["historical", "historical", "historical"]
    subexpIDs = np.arange(20) + 1

    def __init__(self): pass

    def add_args(self, args, subexpID):
        np.random.seed(1)
        n, r = 1276, 20
        labels = np.arange(n) + 1
        train_indices = sorted(np.random.choice(n, size=r, replace=False))
        train_labels = labels[train_indices]
        test_labels = sorted(np.delete(labels, train_indices)[np.random.choice(n-r, size=r, replace=False)])
        train_spatial_selection = "literal,%s" % (train_labels[subexpID-1])
        test_spatial_selection = "literal,%s" % (",".join(map(str, test_labels)))
        args.add("spatial_selection", train_spatial_selection, "train")
        args.add("spatial_selection", train_spatial_selection, "valid")
        args.add("spatial_selection", test_spatial_selection, "test")
#        args.add("plot_model_fit", False)
        if "SWmm" in args.get_argvalue("response_features"):
            args.replace("evaluation_dir", "Evaluations"+os.sep+self.name+os.sep+"SW")
            args.replace("checkpoint_dir", "Checkpoints"+os.sep+self.name+os.sep+"SW")
        elif "FLOW_OUTcms" in args.get_argvalue("response_features"):
            args.replace("evaluation_dir", "Evaluations"+os.sep+self.name+os.sep+"SF")
            args.replace("checkpoint_dir", "Checkpoints"+os.sep+self.name+os.sep+"SF")
        else:
            raise ValueError()
        return args


class Experiment3:

    name = "SwatTransductionInductionAllModels"
    models = ["NaiveLastTimestep", "ARIMA", "GEOMAN", "LSTM"]
#    models = models[-1:]
    predictor_features = ["date,tmin,tmax,PRECIPmm,FLOW_OUTcms,SWmm", "date,tmin,tmax,PRECIPmm,SWmm,FLOW_OUTcms"]
    response_features = ["SWmm,", "FLOW_OUTcms,"]
#    response_features = response_features[:1]
    data_sources = ["historical", "historical", "historical"]
    subexpIDs = np.arange(6) + 1

    def __init__(self): pass

    def add_args(self, args, subexpID):
        # Transductive and inductive spatial selections are separate (e.g. subexpID 1 & 4 not combined into "literal,1,2") because GeoMAN requires spatial element count be identical across partitions
        if subexpID == 1:
            train_spatial_selection = "literal,%d" % (1)
            test_spatial_selection = "literal,%d" % (1)
        elif subexpID == 2:
            train_spatial_selection = "literal,%d" % (141)
            test_spatial_selection = "literal,%d" % (141)
        elif subexpID == 3:
            train_spatial_selection = "literal,%d" % (1275)
            test_spatial_selection = "literal,%d" % (1275)
        elif subexpID == 4:
            train_spatial_selection = "literal,%d" % (1)
            test_spatial_selection = "literal,%d" % (2)
        elif subexpID == 5:
            train_spatial_selection = "literal,%d" % (141)
            test_spatial_selection = "literal,%d" % (142)
        elif subexpID == 6:
            train_spatial_selection = "literal,%d" % (1275)
            test_spatial_selection = "literal,%d" % (1276)
        args.add("spatial_selection", train_spatial_selection, "train")
        args.add("spatial_selection", train_spatial_selection, "valid")
        args.add("spatial_selection", test_spatial_selection, "test")
        args.replace("evaluation_dir", "Evaluations"+os.sep+self.name)
        args.replace("checkpoint_dir", "Checkpoints"+os.sep+self.name)
        return args


class Experiment4:

    name = "LittleRiverTemporalInduction"
    models = ["LSTM"]
    predictor_features = ["date,tmin,tmax,PRECIPmm,FLOW_OUTcms"]
    response_features = ["FLOW_OUTcms,"]
    data_sources = ["littleriver", "littleriver", "littleriver"]
    subexpIDs = np.arange(8) + 1

    def __init__(self): pass

    def add_args(self, args, subexpID):
        spatial_labels = "B,F,I,J,K,M,N,O".split(",")
        spatial_selection = "literal,%s" % (spatial_labels[subexpID-1])
        args.add("spatial_selection", spatial_selection, "train")
        args.add("spatial_selection", spatial_selection, "valid")
        args.add("spatial_selection", spatial_selection, "test")
        args.replace("evaluation_dir", "Evaluations"+os.sep+self.name)
        args.replace("checkpoint_dir", "Checkpoints"+os.sep+self.name)
        return args


class Experiment5:

    name = "SwatToLittleRiverSpatiotemporalInduction"
    models = ["LSTM"]
    predictor_features = ["date,tmin,tmax,PRECIPmm,FLOW_OUTcms"]
    response_features = ["FLOW_OUTcms,"]
    data_sources = ["historical", "historical", "littleriver"]
    subexpIDs = np.arange(8) + 1

    def __init__(self): pass

    def add_args(self, args, subexpID):
        np.random.seed(0)
        train_spatial_labels = list(map(str, sorted(np.random.choice(1276, size=8, replace=False))))
        train_spatial_selection = "literal,%s" % (train_spatial_labels[subexpID-1])
        test_spatial_selection = "literal,B,F,I,J,K,M,N,O"
        args.add("spatial_selection", train_spatial_selection, "train", [self.data_sources[0]])
        args.add("spatial_selection", train_spatial_selection, "valid", [self.data_sources[1]])
        args.add("spatial_selection", test_spatial_selection, "test", [self.data_sources[2]])
        args.replace("evaluation_dir", "Evaluations"+os.sep+self.name)
        args.replace("checkpoint_dir", "Checkpoints"+os.sep+self.name)
        return args


class Experiment6:

    name = "LittleRiverToSwatSpatiotemporalInduction"
    models = ["LSTM"]
    predictor_features = ["date,tmin,tmax,PRECIPmm,FLOW_OUTcms"]
    response_features = ["FLOW_OUTcms,"]
    data_sources = ["littleriver", "littleriver", "historical"]
    subexpIDs = np.arange(16) + 1

    def __init__(self): pass

    def add_args(self, args, subexpID):
        np.random.seed(0)
        test_spatial_labels = sorted(np.random.choice(1276, size=8, replace=False))
        if subexpID <= 8:
            self.data_sources = ["historical", "historical", "historical"]
            train_spatial_labels = test_spatial_labels
            train_spatial_selection = "literal,%d" % (train_spatial_labels[subexpID-1])
            test_spatial_selection = "literal,%d" % (test_spatial_labels[subexpID-1])
        else:
            self.data_sources = ["littleriver", "littleriver", "historical"]
            train_spatial_labels = "B,F,I,J,K,M,N,O".split(",")
            train_spatial_selection = "literal,%s" % (train_spatial_labels[subexpID-8-1])
            test_spatial_selection = "literal," + ",".join(map(str, test_spatial_labels))
        args.replace("data_source", self.data_sources[0], "train")
        args.replace("data_source", self.data_sources[1], "valid")
        args.replace("data_source", self.data_sources[2], "test")
        args.add("spatial_selection", train_spatial_selection, "train", [self.data_sources[0]])
        args.add("spatial_selection", train_spatial_selection, "valid", [self.data_sources[1]])
        args.add("spatial_selection", test_spatial_selection, "test", [self.data_sources[2]])
        args.replace("evaluation_dir", "Evaluations"+os.sep+self.name)
        args.replace("checkpoint_dir", "Checkpoints"+os.sep+self.name)
        return args


class Experiment7:

    name = "PredictorRelevance"
    models = ["LSTM"]
    predictor_features = ["date,PRECIPmm,FLOW_OUTcms"]
    response_features = ["FLOW_OUTcms,"]
    data_sources = ["historical", "historical", "historical"]
    subexpIDs = np.arange(2*36) + 1

    def __init__(self): pass

    def add_args(self, args, subexpID):
        spatial_labels = "1,141,1275".split(",")
        spatial_selection = "literal,%s" % (spatial_labels[0])
        predictor_feature_set = [
            [
                "date,",
                "tmin,",
                "tmax,",
                "PRECIPmm,",
                "FLOW_OUTcms,",
                "SWmm,",
                "date,SWmm",
                "tmin,SWmm",
                "tmax,SWmm",
                "PRECIPmm,SWmm",
                "FLOW_OUTcms,SWmm",
                "date,tmin,SWmm",
                "date,tmax,SWmm",
                "date,PRECIPmm,SWmm",
                "date,FLOW_OUTcms,SWmm",
                "tmin,tmax,SWmm",
                "tmin,PRECIPmm,SWmm",
                "tmin,FLOW_OUTcms,SWmm",
                "tmax,PRECIPmm,SWmm",
                "tmax,FLOW_OUTcms,SWmm",
                "PRECIPmm,FLOW_OUTcms,SWmm",
                "date,tmin,tmax,SWmm",
                "date,tmin,PRECIPmm,SWmm",
                "date,tmin,FLOW_OUTcms,SWmm",
                "date,tmax,PRECIPmm,SWmm",
                "date,tmax,FLOW_OUTcms,SWmm",
                "date,PRECIPmm,FLOW_OUTcms,SWmm",
                "tmin,tmax,PRECIPmm,SWmm",
                "tmin,tmax,FLOW_OUTcms,SWmm",
                "tmin,PRECIPmm,FLOW_OUTcms,SWmm",
                "tmax,PRECIPmm,FLOW_OUTcms,SWmm",
                "date,tmin,tmax,PRECIPmm,SWmm",
                "date,tmin,tmax,FLOW_OUTcms,SWmm",
                "date,tmin,PRECIPmm,FLOW_OUTcms,SWmm",
                "date,tmax,PRECIPmm,FLOW_OUTcms,SWmm",
                "tmin,tmax,PRECIPmm,FLOW_OUTcms,SWmm",
                "date,tmin,tmax,PRECIPmm,FLOW_OUTcms,SWmm"
            ],
            [
                "date,",
                "tmin,",
                "tmax,",
                "PRECIPmm,",
                "SWmm,",
                "FLOW_OUTcms,",
                "date,FLOW_OUTcms",
                "tmin,FLOW_OUTcms",
                "tmax,FLOW_OUTcms",
                "PRECIPmm,FLOW_OUTcms",
                "SWmm,FLOW_OUTcms",
                "date,tmin,FLOW_OUTcms",
                "date,tmax,FLOW_OUTcms",
                "date,PRECIPmm,FLOW_OUTcms",
                "date,SWmm,FLOW_OUTcms",
                "tmin,tmax,FLOW_OUTcms",
                "tmin,PRECIPmm,FLOW_OUTcms",
                "tmin,SWmm,FLOW_OUTcms",
                "tmax,PRECIPmm,FLOW_OUTcms",
                "tmax,SWmm,FLOW_OUTcms",
                "PRECIPmm,SWmm,FLOW_OUTcms",
                "date,tmin,tmax,FLOW_OUTcms",
                "date,tmin,PRECIPmm,FLOW_OUTcms",
                "date,tmin,SWmm,FLOW_OUTcms",
                "date,tmax,PRECIPmm,FLOW_OUTcms",
                "date,tmax,SWmm,FLOW_OUTcms",
                "date,PRECIPmm,SWmm,FLOW_OUTcms",
                "tmin,tmax,PRECIPmm,FLOW_OUTcms",
                "tmin,tmax,SWmm,FLOW_OUTcms",
                "tmin,PRECIPmm,SWmm,FLOW_OUTcms",
                "tmax,PRECIPmm,SWmm,FLOW_OUTcms",
                "date,tmin,tmax,PRECIPmm,FLOW_OUTcms",
                "date,tmin,tmax,SWmm,FLOW_OUTcms",
                "date,tmin,PRECIPmm,SWmm,FLOW_OUTcms",
                "date,tmax,PRECIPmm,SWmm,FLOW_OUTcms",
                "tmin,tmax,PRECIPmm,SWmm,FLOW_OUTcms",
                "date,tmin,tmax,PRECIPmm,SWmm,FLOW_OUTcms"
            ]
        ]
        response_feature_set = ["SWmm,", "FLOW_OUTcms,"]
        # Get values for this sub-experiment
        n = len(predictor_feature_set[0])
        predictor_features = predictor_feature_set[(subexpID-1)//n][(subexpID-1)%n]
        response_features = response_feature_set[(subexpID-1)//n]
        # Update argument values
        args.add("spatial_selection", spatial_selection, "train")
        args.add("spatial_selection", spatial_selection, "valid")
        args.add("spatial_selection", spatial_selection, "test")
        args.replace("predictor_features", predictor_features)
        args.replace("response_features", response_features)
        args.replace("evaluation_dir", "Evaluations"+os.sep+self.name)
        args.replace("checkpoint_dir", "Checkpoints"+os.sep+self.name)
        return args


class Experiment8:

    name = "SwatJointTransductInduction"
    models = ["LSTM"]
    predictor_features = ["date,tmin,tmax,PRECIPmm,FLOW_OUTcms,SWmm", "date,tmin,tmax,PRECIPmm,SWmm,FLOW_OUTcms"]
    response_features = ["SWmm,", "FLOW_OUTcms,"]
    data_sources = ["historical", "historical", "historical"]
    subexpIDs = np.arange(1) + 1

    def __init__(self): pass

    def add_args(self, args, subexpID):
        train_spatial_selection = "literal,1,141,1275"
        test_spatial_selection = "literal,1,141,1275,2,142,1276"
        args.add("spatial_selection", train_spatial_selection, "train")
        args.add("spatial_selection", train_spatial_selection, "valid")
        args.add("spatial_selection", test_spatial_selection, "test")
        args.replace("evaluation_dir", "Evaluations"+os.sep+self.name)
        args.replace("checkpoint_dir", "Checkpoints"+os.sep+self.name)
        return args


class Experiment9:

    name = "SwatExtremeEventForecast"
    models = ["LSTM"]
    predictor_features = ["date,tmin,tmax,PRECIPmm,FLOW_OUTcms,SWmm", "date,tmin,tmax,PRECIPmm,SWmm,FLOW_OUTcms"]
    response_features = ["SWmm,", "FLOW_OUTcms,"]
    data_sources = ["historical", "historical", "historical"]
    subexpIDs = np.arange(3) + 1

    def __init__(self): pass

    def add_args(self, args, subexpID):
        spatial_labels = [1, 141, 1275]
        train_spatial_selection = "literal,%s" % (str(spatial_labels[subexpID-1]))
        test_spatial_selection = "literal,1,141,1275,2,142,1276"
        if subexpID == 1:
            train_spatial_selection = "literal,%d" % (1)
            test_spatial_selection = "literal,%d,%d" % (1, 2)
        elif subexpID == 2:
            train_spatial_selection = "literal,%d" % (141)
            test_spatial_selection = "literal,%d,%d" % (141, 142)
        elif subexpID == 3:
            train_spatial_selection = "literal,%d" % (1275)
            test_spatial_selection = "literal,%d,%d" % (1275, 1276)
        args.add("spatial_selection", train_spatial_selection, "train")
        args.add("spatial_selection", train_spatial_selection, "valid")
        args.add("spatial_selection", test_spatial_selection, "test")
        args.add(
            "options", 
            "groundtruth,prediction,groundtruth_extremes,prediction_extremes,mean,stddev,confusion",
            context=["plotting"]
        )
        args.replace("evaluation_dir", "Evaluations"+os.sep+self.name)
        args.replace("checkpoint_dir", "Checkpoints"+os.sep+self.name)
        return args
        

def add_base_args(args, model, data_sources, predictor_features, response_features):
    args.add("model", model)
    args.add("data_source", data_sources[0], "train")
    args.add("data_source", data_sources[1], "valid")
    args.add("data_source", data_sources[2], "test")
    args.add("predictor_features", predictor_features)
    args.add("response_features", response_features)
    args.add("evaluation_dir", "Evaluations")
    args.add("checkpoint_dir", "Checkpoints")
    return args


main()
