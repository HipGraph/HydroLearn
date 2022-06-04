from sklearn.model_selection import ParameterGrid
from Container import Container
from Driver import Job
import Utility as util
import numpy as np
import os


class Experiment:
    
    def __init__(self):
        self._grid_argvalue_maps = self.grid_argvalue_maps()
        self._grid_argstruct_map = self.grid_argstruct_map()
        self._subexp_ids = self.subexp_ids()
        self.jobs = []
        grid = ParameterGrid(self._grid_argvalue_maps)
        grid_args = Container()
        for arg_value_set in grid:
            for key, value in arg_value_set.items(): # Set args from parameter grid sample
                argstruct = self._grid_argstruct_map[key]
                name, partition, path = key, None, None
                if not argstruct is None:
                    if "partition" in argstruct:
                        partition = value[argstruct.index("partition")]
                    if "path" in argstruct:
                        path = value[argstruct.index("path")]
                    value = value[argstruct.index("value")]
                grid_args.set(name, value, partition, path)
            for subexp_id in self._subexp_ids: # Set subexp args and add to set of all experiments
                args = Container().copy(grid_args)
                args = self.add_pre_subexp(args, subexp_id)
                args = self.add_subexp(args, subexp_id)
                args = self.add_post_subexp(args, subexp_id)
                args = self.add_paths(args)
                job = Job(args, self.checkpoint())
                job.root_dir = self.root_dir
                job.checkpoint_dir = self.checkpoint_dir
                job.checkpoint_path = self.checkpoint_path
                self.jobs += [job]

    def name(self):
        return "Experiment"

    # Property: grid_argvalue_maps
    #   Purpose: Specifies set of arguments and value sets used to define a grid of experiment settings
    #   Example:
    #       grid_argvalue_maps = {
    #           "arg_name 1": ["arg_value 1", "arg_value 2", "arg_value 3"], 
    #           "arg_name 2": ["arg_value 1", "arg_value 2", "arg_value 3"], 
    #       }
    #   Produces the grid:
    #       {"arg_name 1": "arg_value 1", "arg_name 2": "arg_value 1"}
    #       ...
    #       {"arg_name 1": "arg_value 1", "arg_name 2": "arg_value 1"}
    def grid_argvalue_maps(self):
        return {}

    def grid_argstruct_map(self):
        return {key: None for key in self.grid_argvalue_maps().keys()}

    def subexp_ids(self):
        return np.arange(1)

    def add_pre_subexp(self, args, subexp_id):
        return args

    def add_subexp(self, args, subexp_id):
        return args

    def add_post_subexp(self, args, subexp_id):
        return args

    def add_paths(self, args):
        args.set("evaluation_dir", os.sep.join([self.root_dir(), "Evaluations"]))
        args.set("checkpoint_dir", os.sep.join([self.root_dir(), "Checkpoints"]))
        return args
    
    def root_dir(self):
        return os.sep.join([__file__.replace(".py", ""), self.name()])

    def checkpoint_dir(self):
        return os.sep.join([self.root_dir(), "Cache"])

    def checkpoint_path(self):
        return os.sep.join([self.checkpoint_dir(), "CompletedExperiments.txt"])

    def checkpoint(self):
        return True

    def __len__(self):
        return len(self.jobs)

    def __str__(self):
        rep = []
        i = 1
        for exp in self:
            rep += [util.make_msg_block("Experiment %d" % (i)), str(exp)]
            i += 1
        return "\n".join(rep)

    def __iter__(self):
        self.key = 0
        return self

    def __next__(self):
        if self.key < len(self):
            exp = self[self.key]
            self.key += 1
        else:
            raise StopIteration
        return exp

    def __getitem__(self, key):
        return self.jobs[key]
