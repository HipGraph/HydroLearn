import torch
import sys
import numpy as np
import Utility as util


class Container:

    partition_sep = "__"
    path_sep = ":"
    reserved_names = ["partitions"]
    debug = 0

    # Public alias method for set()
    def s(self, name, value, partition=None, path=None):
        return self.set(name, value, partition, path)

    # Public set method: implements all macro set operations
    def set(self, name, value, partition=None, path=None, multi_value=False):
        # set multiple values
        if multi_value and isinstance(value, list):
            # different name and partition for each value
            if isinstance(name, list) and isinstance(partition, list):
                if len(name) != len(value) or len(name) != len(partition):
                    raise ValueError("Name, value and partition lists must be equal length")
                for _name, _value, _partition in zip(name, value, partition):
                    self.__set(_name, _value, _partition, path)
            # different name but same or no partition for each value
            elif isinstance(name, list):
                if len(name) != len(value):
                    raise ValueError("Name and value lists must be equal length")
                for _name, _value in zip(name, value):
                    self.__set(_name, _value, partition, path)
            # same name but different partition for each value
            elif isinstance(partition, list):
                if len(partition) != len(value):
                    raise ValueError("Partition and value lists must be equal length")
                for _value, _partition in zip(value, partition):
                    self.__set(name, _value, _partition, path)
            else:
                raise NotImplementedError()
        # set single value for a single var under a single partition
        elif not isinstance(name, list) and not isinstance(partition, list):
            self.__set(name, value, partition, path)
        # set single value for multiple vars under multiple partitions
        elif isinstance(name, list) and isinstance(partition, list):
            if len(name) != len(partition):
                raise ValueError("Name and partition lists must be equal length")
            for _name, _partition in zip(name, partition):
                self.__set(_name, value, _partition, path) 
        # set single value for multiple vars under one or no partition(s)
        elif isinstance(name, list):
            for _name in name:
                self.__set(_name, value, partition, path)
        # set single value for single var under multiple partitions
        elif isinstance(partition, list):
            for _partition in partition:
                self.__set(name, value, _partition, path)
        else:
            raise NotImplementedError()
        return self

    # Private set method: implements the mapping of a variable's name to its value
    #   Sets a var called by "name", under partition "partition", at descendant container "path" to the value "value"
    def __set(self, name, value, partition=None, path=None):
        self.validate_types(name, value, partition, path)
        assert not self.is_reserved(name), "Attempting to set value for reserved name \"%s\"" % (name)
        con = self.get_container(path, True)
        if partition == "*":
            for _name, _value, _partition in self.get_name_value_partitions(False):
                if name == _name:
                    con.__set(_name, value, _partition)
            return self
        key = self.create_key(name, partition)
        con.update_partitions(partition)
        previous = None
        if con.key_exists(key):
            previous = con.get(name, partition)
        con.__dict__[key] = value
        return previous

    # Public alias method for get()
    def g(self, name, partition=None, path=None, recurse=False, must_exist=True):
        return self.get(name, partition, path, recurse, must_exist)

    # Public get method: implements all macro get operations
    def get(self, name, partition=None, path=None, recurse=False, must_exist=True):
        # get single value of a single var under a single partition
        if not isinstance(name, list) and not isinstance(partition, list):
            return self.__get(name, partition, path, recurse, must_exist)
        # get values of multiple vars under multiple partitions
        elif isinstance(name, list) and isinstance(partition, list):
            if len(name) != len(partition):
                raise ValueError("Name and partition lists must be equal length")

            return [self.__get(_name, _partition, path, recurse, must_exist) for _name, _partition in zip(name, partition)]
        # get values of multiple vars under one or no partition(s)
        elif isinstance(name, list):
            return [self.__get(_name, partition, path, recurse, must_exist) for _name in name]
        # get values of single var under multiple partitions
        elif isinstance(partition, list):
            return [self.__get(name, _partition, path, recurse, must_exist) for _partition in partition]
        else:
            raise NotImplementedError()

    # Private get method: implements retrieval of a variable's value
    #   Gets the value of var called by "name", under partition "partition", at descendant container "path"
    def __get(self, name, partition=None, path=None, recurse=False, must_exist=True):
        self.validate_types(name, None, partition, path)
        con = self.get_container(path)
        if partition == "*":
            value_partition_pairs = []
            for _name, _value, _partition in con.get_name_value_partitions(False):
                if name == _name:
                    value_partition_pairs += [[_value, _partition]]
            return value_partition_pairs
        key = self.create_key(name, partition)
        if not con.key_exists(key):
            val = None
            if recurse:
                for key, value in self.get_key_values():
                    if isinstance(value, Container):
                        val = value.get(name, partition, recurse=True, must_exist=False)
                    if not val is None:
                        break
            if val is None and must_exist:
                raise ValueError("Key \"%s\" does not exist in this Container" % (key))
            else:
                return val
        return con.__dict__[key]

    # Public alias method for rem()
    def r(self, name, partition=None, path=None, recurse=False, must_exist=True):
        return self.rem(name, partition, path, recurse, must_exist)

    # Public remove method: implements all macro remove operations
    def rem(self, name, partition=None, path=None, recurse=False, must_exist=True):
        # remove a single var under a single partition
        if not isinstance(name, list) and not isinstance(partition, list):
            return self.__rem(name, partition, path, recurse, must_exist)
        # remove multiple vars under multiple partitions
        elif isinstance(name, list) and isinstance(partition, list):
            if len(name) != len(partition):
                raise ValueError("Name and partition lists must be equal length")

            return [self.__rem(_name, _partition, path, recurse, must_exist) for _name, _partition in zip(name, partition)]
        # remove multiple vars under one or no partition(s)
        elif isinstance(name, list):
            return [self.__rem(_name, partition, path, recurse, must_exist) for _name in name]
        # remove single var under multiple partitions
        elif isinstance(partition, list):
            return [self.__rem(name, _partition, path, recurse, must_exist) for _partition in partition]
        else:
            raise NotImplementedError()

    # Private remove method: implements deletion of a variable
    #   Deletes a var called by "name", under partition "partition", at descendant container "path"
    def __rem(self, name, partition=None, path=None, recurse=False, must_exist=True):
        self.validate_types(name, None, partition, path)
        con = self.get_container(path)
        if partition == "*":
            partitions = []
            for _name, _value, _partition in con.get_name_value_partitions(False):
                if name == _name:
                    partitions += [_partition]
            for _partition in partitions:
                con.__rem(name, _partition, path, recurse, must_exist)
            return self
        key = self.create_key(name, partition)
        if not con.key_exists(key):
            if recurse:
                for key, value in con.get_key_values():
                    if isinstance(value, Container):
                        value.__rem(name, partition, recurse, must_exist)
            if must_exist:
                raise ValueError("Key \"%s\" does not exist in this Container" % (key))
        else:
            del con.__dict__[key]
        return self

    # Public copy method: implements all macro copy operations
    def copy(self, con):
        bad_type = True
        if isinstance(con, list):
            bad_type = any([not isinstance(_con, Container) for _con in con])
        elif isinstance(con, Container) or con is None:
            bad_type = False
        if bad_type:
            raise ValueError("Item for copying must be a Container, list of Containers, or None")
        if isinstance(con, list):
            for _con in con:
                self.__copy(_con)
        elif isinstance(con, Container):
            self.__copy(con)
        return self

    # Private copy method: copies all variables of the given container into this one
    def __copy(self, con):
        for name, value, partition in con.get_name_value_partitions():
            if isinstance(value, Container): # copy a container
                self.__set(name, Container().__copy(value), partition)
            elif not self.is_reserved(name): # copy a single var
                self.__set(name, value, partition)
        return self

    # Checkout returns a new container populated with the variable(s) called by name
    def checkout(self, name, partition=None, recurse=True, must_exist=False):
        if isinstance(name, str):
            name = [name]
        if isinstance(partition, str):
            partition = [partition]
        elif partition is None:
            partition = [None for i in range(len(name))]
        con = Container()
        for _name, _partition in zip(name, partition):
            value = self.__get(_name, _partition, recurse=recurse, must_exist=must_exist)
            if not value is None:
                con.__set(_name, value, _partition)
        return con

    # Merge combines all vars of this and the given container 
    def merge(self, con, recurse_surface_var=True):
        if recurse_surface_var:
            self.merge_surface_var(con)
        self.merge_containers(con)
        return self

    # Merges all surface variables: those not in a descendant container
    #   recurse: propagate all surface variables through descendant containers
    def merge_surface_var(self, con, recurse=True):
        for name, value, partition in self.get_name_value_partitions():
            if not isinstance(value, Container):
                if con.var_exists(name, partition) and not self.is_reserved(name): # merge required
                    self.__set(name, con.get(name, partition), partition)
            elif recurse: # value is a container and surface vars are being propagated down
                value.merge_surface_var(con)
        return self

    # Merges all non-surface variables: those in descendant containers
    def merge_containers(self, con, in_recursion=False):
        for name, value, partition in con.get_name_value_partitions():
            if name == "partitions": # will be populated automatically
                continue
            if self.var_exists(name, partition): # in the given container: must be merged
                my_value = self.__get(name, partition)
                if isinstance(value, Container) and isinstance(my_value, Container): # merge two containers
                    my_value.merge_containers(value, in_recursion=True)
                elif in_recursion: # only merge vars when in a descendant container
                    self.__set(name, value, partition)
        return self

    def walk(self):
        paths = {}
        leaf_paths = self._walk()
        for path in leaf_paths:
            for i in range(len(path), 0, -1):
                paths[tuple(path[:i])] = None # Cannot hash lists since they are mutable
        paths = [None] + [list(path) for path in paths.keys()] # [None] denotes the root path
        path_var_pairs= []
        for path in paths:
            con = self.get_container(path)
            for name, value, partition in con.get_name_value_partitions(False):
                path_var_pairs += [[path, [name, value, partition]]]
        return path_var_pairs

    def _walk(self):
        paths = []
        for name, value, partition in self.get_name_value_partitions(False):
            if self.is_container(name, partition):
                child_paths = value._walk() # Returns list of string lists (lists of strings)
                if len(child_paths) == 0: # Reached leaf container
                    paths += [[name]]
                else: # Not a leaf container: prepend "name" to all child_paths
                    paths += [[name] + path for path in child_paths]
        return paths

    def find(self, name, value, condition="==", partition=None, path=None, recurse=True):
        con = Container()
        _con = self.get_container(path)
        for _name, _value, _partition in _con.get_name_value_partitions(False):
            if _con.is_container(_name, _partition):
                if _value.__find(name, value, condition, partition, recurse):
                    con.set(_name, _value, _partition)
            else:
                con.set(_name, _value, _partition)
        return con

    def __find(self, name, value, condition="==", partition=None, recurse=True, in_recursion=False): 
        checks = []
        for _name, _value, _partition in self.get_name_value_partitions(False):
            if self.is_container(_name, _partition) and recurse:
                checks.append(_value.__find(name, value, condition, partition, recurse, True))
            elif _name == name and _partition == partition:
                if condition == "==":
                    checks.append(_value == value)
                elif condition == "!=":
                    checks.append(_value != value)
                elif condition == ">=":
                    checks.append(_value >= value)
                elif condition == "<=":
                    checks.append(_value <= value)
                elif condition == ">":
                    checks.append(_value > value)
                elif condition == "<":
                    checks.append(_value < value)
                else:
                    raise NotImplementedError("Condition \"%s\" not implemented for find()" % (condition))
        return any(checks)

    def get_name_value_partitions(self, sort=True, order=""):
        name_value_partitions = []
        for key, value in self.get_key_values():
            partition = self.get_partition_from_key(key)
            name = self.get_name_from_key(key)
            name_value_partitions += [[name, value, partition]]
        if sort:
            name_value_partitions.sort(key = lambda x: x[0])
        if order == "basic_first":
            first, last = [], []
            for name_value_partition in name_value_partitions:
                if isinstance(name_value_partition[1], Container):
                    last += [name_value_partition]
                else:
                    first += [name_value_partition]
            name_value_partitions = first + last
        return name_value_partitions

    # Validate the types are correct for of all inputs
    def validate_types(self, name, value, partition, path):
        self.validate_type(name, "name")
        self.validate_type(value, "value")
        self.validate_type(partition, "partition")
        self.validate_type(path, "path")

    # Validate the type is correct for the given input
    def validate_type(self, item, category):
        category_types_map = {
            "name": [util.Types.is_string], 
            "value": [util.Types.is_anything], 
            "partition": [util.Types.is_none, util.Types.is_string], 
            "path": [util.Types.is_none, util.Types.is_list_of_strings], 
        }
        bad_type = not any(type_func(item) for type_func in category_types_map[category])
        if bad_type:
            raise ValueError("%s has incorrect type: %s" % (category.capitalize(), item))

    # Validate the type is correct for the given input
    def old_validate_type(self, item, category, multi_value=False):
        category_types_map = {
            "name": [util.Types.is_string, util.Types.is_list_of_strings], 
            "value": [util.Types.is_list, util.Types.is_anything], 
            "partition": [util.Types.is_none, util.Types.is_string, util.Types.is_list_of_strings], 
            "path": [util.Types.is_none, util.Types.is_list_of_strings], 
        }
        if multi_value:
            category_types_map["value"] = [util.Types.is_list]
        bad_type = not any(type_func(item) for type_func in category_types_map[category])
        if bad_type:
            raise ValueError("%s has incorrect type: %s" % (category.capitalize(), item))

    # Updates the reserved variable partitions
    def update_partitions(self, partition):
        if partition is None:
            return
        if not "partitions" in self:
            self.__dict__["partitions"] = set()
        self.__get("partitions").add(partition)

    # Create the key that will map this variable to its value
    def create_key(self, name, partition):
        key = ""
        if self.partition_sep in name:
            raise ValueError("Name cannot contain the partition-name separator \"%s\"" % (self.partition_sep))
        if not partition is None:
            key += partition + self.partition_sep
        key += name
        return key

    # Get the container located at the given path
    #   create: add a new container or containers to establish the path if it doesn't exist
    def get_container(self, path, create=False):
        if path is None:
            return self
        con = self
        for _path in path:
            if not con.path_exists(_path):
                if create:
                    con.__set(_path, Container())
                else:
                    raise ValueError("Path \"%s\" does not exist in this Container" % (
                        self.path_sep.join(path))
                    )
            con = con.__get(_path)
        return con

    def get_keys(self):
        return self.__dict__.keys()

    def get_key_values(self):
        return self.__dict__.items()

    def get_partition(self, name, value):
        for key, _value in self.get_key_values():
            _partition = self.get_partition_from_key(key)
            _name = self.get_name_from_key(key)
            if name == _name and value == _value:
                return _partition

    def get_partition_from_key(self, key):
        partition = None
        if self.partition_sep in key:
            partition = self.partition_sep.join(key.split(self.partition_sep)[:-1])
        return partition

    def get_name_from_key(self, key):
        return key.split(self.partition_sep)[-1]

    def size(self):
        return len(self.get_keys())

    def get_memory_of(self):
        size = 0
        for key, value in self.get_key_values():
            if isinstance(value, Container):
                size += value.get_memory_of()
            else:
                size += sys.getsizeof(value)
        return size

    def key_exists(self, key):
        return key in self.__dict__

    def var_exists(self, name, partition=None, path=None, recurse=False):
        con = self.get_container(path)
        return con.key_exists(self.create_key(name, partition))

    def path_exists(self, path):
        if path is None:
            return False
        return self.key_exists(path) and self.is_container(path)

    def is_var(self, name, partition=None, path=None):
        return not self.is_container(name, partition, path)

    def is_container(self, name, partition=None, path=None):
        return isinstance(self.__get(name, partition, path), Container)

    def is_reserved(self, name):
        return name in self.reserved_names

    def is_empty(self):
        return self.size() == 0

    def to_string(self, recurse=True, sort=True, extent=[110, 1], in_recursion=False):
        def cut_x(line, x_extent):
            if x_extent < 0:
                return line
            j = line.rfind(" = ") + 3
            cut_idx = max(j, x_extent)
            return line[:cut_idx] + " ..."
        def cut_y(var_string, y_extent):
            if y_extent < 0:
                return var_string
            var_string_lines = var_string.split("\n")
            return "\n".join(var_string_lines[:y_extent])
        expand = True
        indent = 4 * " "
        lines = []
        max_key_len = -1
        for key, value in self.get_key_values():
            if len(key) > max_key_len:
                max_key_len = len(key)
        left_just = max_key_len
        for name, value, partition in self.get_name_value_partitions(sort=sort, order="basic_first"):
            key = self.create_key(name, partition)
            value_string = str(value)
            if isinstance(value, Container):
                left_just = 0
                key = "-> " + key
                var_string = "Container @ size(%s)" % (util.format_memory(value.get_memory_of()))
                if recurse:
                    var_string += " = \n%s%s" % (
                        indent, 
                        value.to_string(recurse, sort, extent, True).replace("\n", "\n%s" % (indent))
                    )
                var_string = "%-*s = %s" % (left_just, key, var_string)
                lines += var_string.split("\n")
            else:
                if isinstance(value, np.ndarray):
                    var_string = "NumPy.ndarray @ shape(" + ", ".join(map(str, value.shape)) + ")"
                    if expand:
                        var_string += " = %s" % (value_string)
                elif isinstance(value, torch.Tensor):
                    var_string = "PyTorch.Tensor @ shape(" + ", ".join(map(str, value.shape)) + ")"
                    if expand:
                        var_string += " = %s" % (value_string)
                elif isinstance(value, list):
                    var_string = "List @ len(%d)" % len(value)
                    if expand:
                        var_string += " = %s" % (value_string)
                elif isinstance(value, dict):
                    var_string = "Dictionary @ len(%d)" % len(value)
                    if expand:
                        var_string += " = %s" % (value_string)
                elif isinstance(value, set):
                    var_string = "Set @ len(%d)" % len(value)
                    if expand:
                        var_string += " = %s" % (value_string)
                elif isinstance(value, str):
                    var_string = "String @ len(%d)" % len(value)
                    if expand:
                        var_string += " = \"%s\"" % (value_string)
                else:
                    var_string = value_string
                if var_string.count("\n") > extent[1]:
                    var_string = cut_y(var_string, extent[1])
                lines += ["%-*s = %s" % (left_just, key, var_string)]
        if not in_recursion: # Only cut lines of final result
            for i in range(len(lines)):
                if len(lines[i]) > extent[0]: # Needs to be cut short
                    lines[i] = cut_x(lines[i], extent[0])
        return "\n".join(lines)

    def __str__(self):
        return self.to_string()

    def __contains__(self, obj):
        return self.key_exists(obj)

    def __getitem__(self, key):
        return self.get(key)
