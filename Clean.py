import sys
import os

import Utility as util
from Container import Container
from Arguments import ArgumentParser
from Variables import Variables


def clean_dataset_cache(args):
    dataset = input("Dataset? : ")
    var = Variables()
    dataset_var = var.datasets.get(dataset)
    print(dataset_var)
    cache_dir = dataset_var.spatiotemporal.structure.cache_dir
    check = input(
        "Remove cache directory \"%s\" for dataset \"%s\"? : " % (cache_dir, dataset)
    )
    if check.lower() in ["y", "yes"]:
        check = input("Are you sure? : ")
    else:
        return
    if check.lower() in ["y", "yes"]:
        os.system("rm -r %s" % (cache_dir))
    else:
        return


def clean_experiment_cache(args):
    exp_module = input("Experiment Module? : ")
    module_dir = os.sep.join(["Experimentation", exp_module])
    if exp_module == "":
        paths = util.get_paths("Experimentation", ".*", files=False)
        print("No experiment module given. Here are your options:")
        module_dir = util.get_choice(paths)
    exp = input("Experiment? : ")
    cache_dir = os.sep.join(["Experimentation", exp_module, exp])
    if exp == "":
        paths = util.get_paths(module_dir, ".*", files=False)
        print("No experiment given. Here are your options:")
        cache_dir = util.get_choice(paths)
    #
    check = input(
        "Remove cache directory \"%s\"? : " % (cache_dir)
    )
    if check.lower() in ["y", "yes"]:
        check = input("Are you sure? : ")
    else:
        return
    if check.lower() in ["y", "yes"]:
        os.system("rm -r %s" % (cache_dir))
    else:
        return


if __name__ == "__main__":
    _args = Container().set(
        [
            "mode", 
        ], 
        [
            "dataset_cache", 
        ], 
        multi_value=1
    )
    args = ArgumentParser().parse_arguments(sys.argv[1:])
    args = _args.merge(args)
    print(args)
    globals()["clean_%s"%(args.mode)](args)
