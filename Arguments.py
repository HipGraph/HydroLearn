import sys
import os
import re
import json
import warnings
import numpy as np

from Container import Container


arg_flag = "--"
partition_sep = Container.partition_sep
path_sep = Container.path_sep


class ArgumentParser:

    def __init__(self): pass

    def is_argname(self, string):
        return arg_flag in string

    def get_argname_elements(self, argname):
        rem_argname = argname.replace(arg_flag, "")
        path = (rem_argname.split(path_sep)[:-1] if path_sep in rem_argname else None)
        rem_argname = rem_argname.split(path_sep)[-1]
        partition = (rem_argname.split(partition_sep)[0] if partition_sep in rem_argname else None)
        rem_argname = rem_argname.split(partition_sep)[-1]
        name = rem_argname
        return name, partition, path

    def parse_arguments(self, argv, con=None):
        if con is None: con = Container()
        if not isinstance(argv, list) or len(argv) % 2 != 0:
            raise ValueError(
                "Arguments must be a list (sys.argv) and contain an even number of elements, received %s" % (
                    str(argv)
                )
            )
        for i in range(0, len(argv), 2):
            argname, argvalue = argv[i:i+2]
            if not self.is_argname(argname):
                raise ValueError(
                    "Expecting argument name to contain \"%s\", but received \"%s\"." % (
                        arg_flag, 
                        argname
                    )
                )
            name, partition, path = self.get_argname_elements(argname)
            try:
                value = json.loads(argvalue)
            except:
                err_msg = [
                    "Json failed to load arg-value \"%s\"." % (argvalue), 
                    "Json requires the following syntax:", 
                    "\tbools: use only true or false (case sensitive).", 
                    "\tstrings: use \\\"hello\\\" (double quotes must be escaped).", 
                    "\tlist of strings: use [\\\"hello\\\",\\\"world\\\"] for example.", 
                    "\tdictionary of strings: use {\\\"hello\\\":\\\"world\\\"} for example.", 
                ]
                raise ValueError("\n".join(err_msg))
            con.set(name, value, partition, path)
        return con


class NumPyEncoder(json.JSONEncoder):

    def default(self, value):
        if isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        return super(NpEncoder, self).default(value)


class ArgumentBuilder:

    def __init__(self): pass

    def build_argvalue(self, value):
        return json.dumps(value, cls=NumPyEncoder)

    def build_argname(self, name, partition=None, path=None):
        partition_str = ("" if partition is None else "%s%s" % (partition, partition_sep))
        path_str = ("" if path is None else path_sep.join(path) + path_sep)
        return "%s%s%s%s" % (arg_flag, path_str, partition_str, name)

    def build_arg(self, name, value, partition=None, path=None):
        return [self.build_argname(name, partition, path), self.build_argvalue(value)]

    def build(self, con):
        args = []
        for path, nam_val_par in con.walk():
            name, value, partition = nam_val_par
            if not isinstance(value, Container) and name not in con.reserved_names:
                args += self.build_arg(name, value, partition, path)
        return args

    def view(self, con):
        args = self.build(con)
        return "\n".join([" = ".join(args[i:i+2]) for i in range(0, len(args), 2)])
