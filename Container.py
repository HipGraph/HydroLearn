from os import sep as os_sep
from numpy import ndarray
from numpy import set_printoptions
from torch import Tensor
from sys import getsizeof
import Utility as util


set_printoptions(precision=3, suppress=True, linewidth=200)


class Container:


    partition_sep = os_sep
    context_sep = ":"


    def __init__(self):
        pass


    def g(self, name, partition=None, context=None, recurse=False, must_exist=True):
        return self.get(name, partition, context, recurse, must_exist)


    def get(self, name, partition=None, context=None, recurse=False, must_exist=True):
        if isinstance(name, str) and (isinstance(partition, str) or partition is None):
            return self._get(name, partition, context, recurse, must_exist)
        elif isinstance(name, list):
            return [self._get(_name, partition, context, recurse, must_exist) for _name in name]
        elif isinstance(partition, list):
            return [self._get(name, _partition, context, recurse, must_exist) for _partition in partition]
        else:
            ValueError("Name must be type string/list, partition must be type string/list or None")

    def _get(self, name, partition=None, context=None, recurse=False, must_exist=True):
        if partition == "*":
            values = []
            for _name, _value1, _partition in self.get_name_value_partitions(False):
                if name == name1:
                    print(_name, _value, _partition)
                    values += [_value]
            return values
        key = self.get_key(name, partition)
        container = self.get_container(context)
        if not container.key_exists(key):
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
        return container.__dict__[key]

    def s(self, name, value, partition=None, context=None):
        return self.set(name, value, partition, context)

    def set(self, name, value, partition=None, context=None):
        self.validate_type(name, "name")
        self.validate_type(value, "value")
        self.validate_type(partition, "partition")
        self.validate_type(context, "context")
        if isinstance(name, list) and isinstance(value, list):
            name_len = len(name)
            if not all(len(item) == name_len for item in [name, value]):
                raise ValueError("Name and value lists must be equal length")
            if isinstance(partition, list):
                partition += [None for i in range(name_len-len(partition))]
            else:
                partition = [partition for i in range(name_len)]
            for _name, _value, _partition in zip(name, value, partition):
                self._set(_name, _value, _partition, context)
        elif isinstance(name, list):
            raise NotImplementedError()
        else:
            self._set(name, value, partition, context)
        return self

    def validate_type(self, item, category):
        category_types_map = {
            "name": [util.is_string, util.is_list_of_strings], 
            "value": [util.is_anything], 
            "partition": [util.is_string, util.is_list_of_strings, util.is_none], 
            "context": [util.is_list_of_strings, util.is_none], 
        }
        correct = any(type_func(item) for type_func in category_types_map[category])
        if not correct:
            raise ValueError("%s has incorrect type: %s" % (category.capitalize(), item))

    def _set(self, name, value, partition=None, context=None):
#        self = self.get_context_leaf(context)
#        print("Set: ", name, value, partition, context)
        if not isinstance(name, str):
            raise ValueError("Name must be type str()")
        if self.partition_sep in name:
            raise ValueError("Name must not contain \"%s\"", self.partition_sep)
        if not partition is None and not isinstance(partition, str):
            raise ValueError("Partition names must be None or type str()")
        if isinstance(context, list) and len(context) > 0 and not isinstance(context[0], str):
            raise ValueError("Context names must be a list() of one or more type str() items")
        if not partition is None:
            self.update_partitions(partition)
        key = self.get_key(name, partition)
        container = self.get_container(context, True)
        previous = None
        if container.key_exists(key):
            previous = container.get(name, partition)
        container.__dict__[key] = value
        return previous


    def copy(self, container):
        if isinstance(container, list) and len(container) > 0 and isinstance(container[0], Container):
            for _container in container:
                self.copy(_container)
        elif isinstance(container, Container):
            self._copy(container)
        elif container is None:
            pass
        else:
            print(container.to_string())
            raise ValueError("Item for copying must be a Container, list of Containers, or None")
        return self


    def _copy(self, container):
        for name, value, partition in container.get_name_value_partitions():
            self.set(name, value, partition)

    def checkout(self, name):
        self.validate_type(name, "name")
        if isinstance(name, str):
            name = [name]
        container = Container()
        for _name in name:
            value = self.get(_name, recurse=True, must_exist=False)
            if not value is None:
                container.set(_name, value)
        return container


    def merge(self, container, recurse_noncontextual=True):
        if recurse_noncontextual:
            self.merge_noncontextual(container)
        self.merge_contextual(container)


    # Merge non-container variables recursively
    def merge_noncontextual(self, container, recurse=True):
        for name, value, partition in container.get_name_value_partitions():
            if self.var_exists(name, partition) and not isinstance(value, Container):
                    self.set(name, value, partition)
        if recurse:
            for my_name, my_value, my_partition in self.get_name_value_partitions():
                if isinstance(my_value, Container):
                    my_value.merge_noncontextual(container)
        return self


    def merge_contextual(self, container):
        for name, value, partition in container.get_name_value_partitions():
            if name == "partitions":
                continue
            if self.var_exists(name, partition): # merge two common variables
                my_value = self.get(name, partition)
                if isinstance(value, Container) and isinstance(my_value, Container): # merge two containers
                    my_value.merge_contextual(value)
                else:
                    self.set(name, value, partition)
        return self


    def get_key(self, name, partition):
        key = ""
        if not partition is None:
            key += partition + self.partition_sep
        key += name
        return key


    def get_container(self, context, create=False):
        if context is None:
            return self
        container = self
        for _context in context:
            if not container.context_exists(_context) and create:
                container.set(_context, Container())
            elif not create:
                raise ValueError("Context \"%s\" does not exist in this Container" % (context_sep.join(context)))
            container = container.get(_context)
        return container


    def key_exists(self, key):
        return key in self.__dict__


    def var_exists(self, name, partition=None, context=None, recurse=False):
        return self.key_exists(self.get_key(name, partition))


    def context_exists(self, context):
        if context is None:
            return False
        return self.key_exists(context) and isinstance(self.get(context), Container)


    def update_partitions(self, partition):
        if not self.key_exists("partitions"):
            self.set("partitions", set())
        self.get("partitions").add(partition)


    def get_keys(self):
        return list(self.__dict__.keys())


    def get_key_values(self):
        return self.__dict__.items()


    def get_name_value_partitions(self, sort=True):
        name_value_partitions = []
        for key, value in self.get_key_values():
            partition = self.get_partition_from_key(key)
            name = self.get_name_from_key(key)
            name_value_partitions.append([name, value, partition])
        if sort:
            name_value_partitions.sort(key = lambda x: x[0])
            return name_value_partitions
        return name_value_partitions


    def get_partition(self, target_name, target_value):
        for key, value in self.get_key_values():
            partition = self.get_partition_from_key(key)
            name = self.get_name_from_key(key)
            if target_value is value and target_name is name:
                return partition


    def get_partition_from_key(self, key):
        partition = None
        if self.partition_sep in key:
            partition = self.partition_sep.join(key.split(self.partition_sep)[:-1])
        return partition


    def get_name_from_key(self, key):
        return key.split(self.partition_sep)[-1]


    def get_memory_of(self):
        size = 0
        for key, value in self.get_key_values():
            if isinstance(value, Container):
                size += value.get_memory_of()
            else:
                size += getsizeof(value)
        return size

    def size(self):
        return len(self.get_key_values())


    def to_string(self, expand=False, recurse=False, sort=True):
        indent = "   "
        strings = []
        max_key_len = -1
        for key, value in self.get_key_values():
            if len(key) > max_key_len:
                max_key_len = len(key)
        left_just = max_key_len
        for name, value, partition in self.get_name_value_partitions(sort=sort):
            key = self.get_key(name, partition)
            if isinstance(value, ndarray):
                element_string = "NumPy.ndarray->shape(" + ", ".join(map(str, value.shape)) + ")"
                if expand:
                    element_string += " = \n%s%s" % (indent, str(value))
            elif isinstance(value, Tensor):
                element_string = "PyTorch.Tensor->shape(" + ", ".join(map(str, value.shape)) + ")"
                if expand:
                    element_string += " = \n%s%s" % (indent, str(value))
            elif isinstance(value, Container):
                element_string = "Container->size(%s)" % (util.format_memory(value.get_memory_of()))
                if expand:
                    element_string += " = \n%s%s" % (
                        indent, 
                        value.to_string(expand).replace("\n", "\n%s" % (indent))
                    )
            elif isinstance(value, str):
                element_string = "\"%s\"" % (value)
            elif isinstance(value, list):
                element_string = "List->len(%d)" % len(value)
                if expand:
                    element_string += " = \n%s%s" % (indent, str(value))
            elif isinstance(value, dict):
                element_string = "Dictionary->len(%d)" % len(value)
                if expand:
                    element_string += " = \n%s%s" % (indent, str(value))
            else:
                element_string = str(value)
            strings.append(
                "%-*s = %s" %
                (
                    left_just,
                    key,
                    element_string
                )
            )
        return "\n".join(strings)
