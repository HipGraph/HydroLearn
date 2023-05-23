import itertools
import numpy as np
import os
from Container import Container
from Driver import Job
import Utility as util


class Experiment:
    
    def __init__(self):
        self._subexp_hooks = []
        self._subexp_prehooks = []
        self.init_prehook()
        self._grid_argvalue_map = self.grid_argvalue_map()
        self._grid_argstruct_map = self.grid_argstruct_map()
        self._subexp_ids = self.subexp_ids()
        self.jobs = []
        grid = ParameterGrid(self._grid_argvalue_map)
        grid_args = Container()
        for argvalue_set in grid:
            for key, value in argvalue_set.items(): # Set args from parameter grid sample
                argstruct = self._grid_argstruct_map.get(key, None)
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
                for subexp_prehook in self._subexp_prehooks: # Call all pre-hooks
                    args = subexp_prehook(args, subexp_id)
                args = self.add_subexp(args, subexp_id)
                for subexp_hook in self._subexp_hooks: # Call all hooks
                    args = subexp_hook(args, subexp_id)
                args = self.add_post_subexp(args, subexp_id)
                args = self.add_paths(args)
                job = Job(args, self.checkpoint())
                job.root_dir = self.root_dir
                job.checkpoint_dir = self.checkpoint_dir
                job.checkpoint_path = self.checkpoint_path
                self.jobs.append(job)
        self.init_hook()

    def init_prehook(self):
        pass

    def init_hook(self):
        pass

    def name(self):
        return "Experiment"

    # Property: grid_argvalue_map
    #   Purpose: Specifies set of arguments and value sets used to define a grid of experiment settings
    #   Example:
    #       grid_argvalue_map = {
    #           "argname 1": ["argvalue 1", "argvalue 2", "argvalue 3"], 
    #           "argname 2": ["argvalue 1", "argvalue 2", "argvalue 3"], 
    #       }
    #   Produces the grid:
    #       {"argname 1": "argvalue 1", "argname 2": "argvalue 1"}
    #       ...
    #       {"argname 1": "argvalue 1", "argname 2": "argvalue 1"}
    def grid_argvalue_map(self):
        return {}

    def grid_argstruct_map(self):
        return {key: None for key in self.grid_argvalue_map().keys()}

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

    def register_subexp_prehook(self, fn, idx):
        if idx == -1:
            idx = len(self._subexp_hooks)
        elif idx < -1:
            idx += 1
        self._subexp_prehooks.insert(idx, fn)

    def register_subexp_hook(self, fn, idx):
        if idx == -1:
            idx = len(self._subexp_hooks)
        elif idx < -1:
            idx += 1
        self._subexp_hooks.insert(idx, fn)
    
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
        

class ParameterSearch(Experiment):

    def add_paths(self, args):
        args.set("evaluation_dir", os.sep.join([self.root_dir(), "Evaluations"]))
        args.set("checkpoint_dir", os.sep.join([self.root_dir(), "Checkpoints"]))
        if args.has("dataset", "train"):
            dataset = args.get("dataset", "train")
            args.set("evaluation_dir", os.sep.join([self.root_dir(), dataset, "Evaluations"]))
            args.set("checkpoint_dir", os.sep.join([self.root_dir(), dataset, "Checkpoints"]))
        return args


class ParameterGrid:

    def __init__(self, grid):
        param_names = np.array(list(grid.keys()))
        param_value_sets = np.array([tuple(values) for values in grid.values()], dtype=object)
        shape = tuple(len(values) for values in param_value_sets)
        grid = list(itertools.product(*param_value_sets))
        self.param_names = param_names
        self.param_value_sets = param_value_sets
        self.shape = shape
        self.grid = grid

    def __getitem__(self, key):
        if not (isinstance(key, int) or isinstance(key, slice)):
            raise ValueError("Input key must be int or slice")
        indices = np.unravel_index(key, self.shape)
        return util.to_dict(self.param_names, [self.param_value_sets[i][j] for i, j in enumerate(indices)])

    def __len__(self):
        return len(self.grid)

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= len(self):
            raise StopIteration
        value = self[self._idx]
        self._idx += 1
        return value

    def __str__(self):
        tab_space = 4 * " "
        lines = ["%s" % (__class__.__name__)]
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            value_str = str(value)
            if "\n" in value_str:
                value_str = "\n%s%s" % (2*tab_space, value_str.replace("\n", "\n%s" % (2*tab_space)))
            lines.append("%s%s = %s" % (tab_space, key, value_str))
        lines.append(")")
        return "\n".join(lines)
