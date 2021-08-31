from Container import Container
from os import sep as os_sep
import re


list_sep = ","
arg_flag = "--"
partition_sep = Container.partition_sep
context_sep = Container.context_sep
bool_strings = [str(True).lower(), str(False).lower(), str(True), str(False)]


class ArgumentParser(Container):

    def __init__(self, argv):
        self.copy(self.parse_arguments(argv))

    def is_argname(self, string):
        return arg_flag in string

    def get_argname_elements(self, argname):
        rem_argname = argname.replace(arg_flag, "")
        context = (rem_argname.split(context_sep)[:-1] if context_sep in rem_argname else None)
        rem_argname = rem_argname.split(context_sep)[-1]
        partition = (rem_argname.split(partition_sep)[0] if partition_sep in rem_argname else None)
        rem_argname = rem_argname.split(partition_sep)[-1]
        name = rem_argname
        return name, partition, context

    def parse_arguments(self, argv):
        if not isinstance(argv, list):
            raise ValueError()
        container = Container()
        for i in range(0, len(argv), 2):
            if self.is_argname(argv[i]):
                name, partition, context = self.get_argname_elements(argv[i])
                argvalue = argv[i+1]
            else:
                raise ValueError()
            value = self.string_to_value(argvalue)
            container.set(name, value, partition, context)
        return container

    def string_to_value(self, string):
        if self.is_int(string):
            value = self.string_to_int(string)
        elif self.is_float(string):
            value = self.string_to_float(string)
        elif self.is_bool(string):
            value = self.string_to_bool(string)
        elif self.is_list(string):
            value = self.string_to_list(string)
        else:#elif is_string(string):
            value = self.string_to_string(string)
#        else:
#            raise ValueError("The type of argument value %s was not recognized and cannot be parsed" % (string))
        return value

    def string_to_int(self, string):
        return int(string)

    def string_to_float(self, string):
        return float(string)

    def string_to_bool(self, string):
        return "t" in string.lower()

    def string_to_list(self, string):
        sep = self.get_list_order(string) * list_sep
        if string.endswith(sep): # list was a single element: trim the trailing empty string
            string = string[:-len(sep)]
        return [self.string_to_value(str_i) for str_i in string.split(sep)]

    def string_to_string(self, string):
        return string.replace("\"", "").replace("'", "")

    def is_int(self, string):
        return string.isdigit() or (string[0] is "-" and string[1:].isdigit())

    def is_float(self, string):
        return "." in string and self.is_int(string.replace(".", ""))

    def is_bool(self, string):
        return string in bool_strings

    def is_list(self, string):
        return list_sep in string

    def get_list_order(self, string):
        return len(max(re.compile("%s*" % (list_sep)).findall(string)))

    def is_string(self, string):
        return (string[0] == "\"" and string[-1] == "\"") or (string[0] == "'" and string[-1] == "'")


class ArgumentBuilder(Container):

    def __init__(self):
        self.set("args", [])

    def build_argvalue(self, value):
        return str(value)

    def build_argname(self, name, partition=None, context=None):
        partition_str = ("" if partition is None else "%s%s" % (partition, partition_sep))
        context_str = ("" if context is None else context_sep.join(context) + context_sep)
        return "%s%s%s%s" % (arg_flag, context_str, partition_str, name)

    def build_arg(self, name, value, partition=None, context=None):
        return [self.build_argname(name, partition, context), self.build_argvalue(value)]

    def add(self, name, value, partition=None, context=None, prepend=False):
        if prepend:
            self.set("args", self.build_arg(name, value, partition, context) + self.get("args"))
        else:
            self.set("args", self.get("args") + self.build_arg(name, value, partition, context))
        return self

    def replace(self, name, value, partition=None, context=None):
        try:
            i = self.get("args").index(self.build_argname(name, partition, context))
            self.get("args")[i+1] = self.build_argvalue(value)
        except ValueError as e:
            if "is not in list" not in e:
                raise ValueError(e)
        return self

    def get_argvalue(self, name, partition=None, context=None):
        try:
            i = self.get("args").index(self.build_argname(name, partition, context))
            return self.get("args")[i+1]
        except ValueError as e:
            if "is not in list" not in e:
                raise ValueError(e)

    def add_mode_args(self, mode, prepend=False):
        train_mode, eval_mode = 1, 2
        if str(eval_mode) in mode:
            checkpoint_filename = input("Checkpoint Filename: ")
            eval_start = input("Evaluation Start: ")
            eval_end = input("Evaluation End: ")
            eval_range = "%s,%s" % (eval_start, eval_end)
            print()
        if prepend:
            if str(eval_mode) in mode:
                self.add("evaluation_range", eval_range, None, ["evaluating"])
                self.add("evaluated_checkpoint", checkpoint_filename, None, ["evaluating"])
                self.add("evaluate", "True", None, ["evaluating"], prepend)
            if str(train_mode) in mode:
                self.add("train", "True", None, ["training"], prepend)
        else:
            if str(train_mode) in mode:
                self.add("train", "True", None, ["training"], prepend)
            if str(eval_mode) in mode:
                self.add("evaluate", "True", None, ["evaluating"], prepend)
                self.add("evaluated_checkpoint", checkpoint_filename, None, ["evaluating"])
                self.add("evaluation_range", eval_range, None, ["evaluating"])
        return self

    def add_invocation_args(self, program, prepend=False):
        program_arg = self.build_arg("python", "%s.py" % (program))
        program_arg[0] = program_arg[0].replace(arg_flag, "")
        if prepend:
            self.add("n_processes", 1, None, ["distributed"], prepend)
            self.add("process_rank", 0, None, ["distributed"], prepend)
            self.set("args", program_arg + self.get("args"))
        else:
            self.set("args", self.get("args") + program_arg)
            self.add("process_rank", 0, None, ["distributed"], prepend)
            self.add("n_processes", 1, None, ["distributed"], prepend)
        return self

    def to_string(self):
        args = self.get("args")
        string = "\n"
        for i in range(0, len(args), 2):
            string += "%s %s\n" % (args[i], args[i+1])
        return string
