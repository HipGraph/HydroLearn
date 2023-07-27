import numpy as np

import Utility as util


class DataSelection:

    def interval_from_selection(self, selection):
        mode = selection[0]
        if mode == "interval":
            interval = selection[1:]
        elif mode == "range":
            interval = selection[1:3]
        elif mode == "literal":
            interval = selection[[1, -1]]
        elif mode == "random":
            interval = [str(selection[1]), str(selection[1])]
        elif mode == "random-split":
            interval = [str(i) for i in selection[1:3]]
        elif mode == "ordered-split":
            interval = [str(i) for i in selection[1:3]]
        elif mode == "all":
            interval = ["all", "all"]
        else:
            raise NotImplementedError(selection)
        return interval

    def indices_from_selection(self, labels, selection, **kwargs):
        mode = selection[0]
        if mode[0] == "~":
            selection = [mode[1:]] + selection[1:]
            indices = self.indices_from_selection(labels, selection)
            indices = np.delete(np.arange(len(labels)), indices)
        else:
            try:
                if mode == "interval":
                    start_idx = np.where(labels == str(selection[1]))[0][0]
                    end_idx = np.where(labels == str(selection[2]))[0][0]
                    indices = np.arange(start_idx, end_idx+1)
                elif mode == "range":
                    start, end, step = selection[1:]
                    if start < 0:
                        start += len(labels) 
                    if end < 0:
                        end += len(labels) 
                    indices = np.arange(start, end, step)
                elif mode == "random":
                    k = selection[1]
                    if isinstance(k, float):
                        k = round(k * len(labels))
                    rng = np.random.default_rng()
                    if len(selection) > 2:
                        rng = np.random.default_rng(selection[2])
                    indices = rng.choice(len(labels), k, replace=False)
                elif mode == "random-split":
                    start, end = selection[1:3]
                    if isinstance(start, float):
                        start = int(start * len(labels))
                    elif not isinstance(start, int):
                        raise ValueError()
                    if isinstance(end, float):
                        end = int(end * len(labels))
                    elif not isinstance(end, int):
                        raise ValueError()
                    rng = np.random.default_rng()
                    if len(selection) > 3:
                        rng = np.random.default_rng(selection[3])
                    indices = rng.permutation(len(labels))[start:end]
                elif mode == "ordered-split":
                    start, end = selection[1:3]
                    if isinstance(start, float):
                        start = int(start * len(labels))
                    elif not isinstance(start, int):
                        raise ValueError(selection)
                    if isinstance(end, float):
                        end = int(end * len(labels))
                    elif not isinstance(end, int):
                        raise ValueError(selection)
                    indices = np.arange(len(labels))[start:end]
                elif mode == "literal":
                    indices = np.array(
                        util.get_dict_values(util.to_key_index_dict(labels), selection[1:], **kwargs)
                    )
                elif mode == "all":
                    indices = np.arange(0, len(labels))
                else:
                    raise NotImplementedError("Selection mode \"%s\" not implemented" % (mode))
            except IndexError as err:
                print("Function indices_from_selection() failed to locate an element from the set of labels and selection criteria below:")
                print("Labels @", "len(%d)" % (len(labels)), "=")
                print(labels)
                print("Selection @", "len(%d)" % (len(selection)), "=")
                print(selection)
                raise IndexError(err)
        return indices

    # Description:
    #   Filter a single axis of "data" by applying the given filter index set "indices"
    # Arguments:
    #   data - the data to be filtered
    #   axis - the target axis/dimension of data for filtering
    #   indices - the array of locations at which to pull values from data along the given axis
    # Requirements:
    #   1. axis - integer
    #   2. indices - int or ndarray
    def __filter_axis(self, data, axis, indices):
        if not isinstance(axis, int):
            raise ValueError("Axis must be an integer, received %s" % (type(axis)))
        if not isinstance(indices, (int, np.int32, np.int64, np.ndarray)):
            raise ValueError("Filter index set must be int or NumPy.ndarray, received %s" % (type(indices)))
        if isinstance(indices, np.ndarray) and len(indices) == 0:
            new_shape = list(data.shape)
            new_shape[axis] = 0
            return np.empty(new_shape)
        return np.take(data, indices, axis)

    # Description:
    #   Broadcasting wrapper for __filter_axis() to handle multiple "axis" and "indices" arguments
    # Arguments:
    #   data - the data to be filtered
    #   axis - the axis to filter
    #   indices - the array, or arrays, of locations at which to pull values from data along the given axis
    # Requirements:
    #   1. axis - integer or list of integers
    #   2. indices - ndarray or list of ndarrays
    def filter_axis(self, data, axis, indices):
        """

        Arguments
        ---------
        data : ndarray or tuple/list/dict of ndarray
        axis : int or tuple/list/dict of int
        indices : ndarray or tuple/list/dict of ndarray

        Returns
        -------
        data : ndarray or tuple/list/dict of ndarray
            the data with given axis or axes filtered by the given indices

        """
        if isinstance(data, tuple):
            return tuple(self.filter_axis(_, axis, indices) for _ in data)
        elif isinstance(data, list):
            return tuple(self.filter_axis(_, axis, indices) for _ in data)
        elif isinstance(data, dict):
            return {key: self.filter_axis(value, axis, indices) for key, value in data.items()}
        # filter just one axis with one index set
        if not isinstance(axis, (tuple, list)) and not isinstance(indices, (tuple, list)):
            return self.__filter_axis(data, axis, indices)
        # filter a set of axes with a set of filter index sets
        if isinstance(axis, (tuple, list)) and isinstance(indices, (tuple, list)):
            if len(axis) == len(indices): 
                for _axis, _indices in zip(axis, indices):
                    data = self.__filter_axis(data, _axis, _indices)
            else: # not a 1:1 mapping, try 1:n broadcasting
                if len(axis) == 1: # broadcast axis to all filter indices
                    for _indices in indices:
                        data = self.__filter_axis(data, axis[0], _indices)
                elif len(indices) == 1: # broadcast filter indices to all axes
                    for _axis in axis:
                        data = self.__filter_axis(data, _axis, indices[0])
                else: # lengths not equal and cannot be broadcasted
                    raise ValueError("Number of axes and filter index sets must be equal or broadcastable")
        elif isinstance(axis, (tuple, list)): # a single filter index set, broadcast it to all axes
            for _axis in axis:
                data = self.__filter_axis(data, _axis, indices)
        elif isinstance(indices, (tuple, list)): # a single axis, broadcast it to all fitler index sets
            for _indices in indices:
                data = self.__filter_axis(data, axis, _indices)
        return data

    # Description:
    #   Filter multiple axes of "data" by applying the given multi-axis filter index set "indices"
    # Arguments:
    #   data - the data to be filtered
    #   axes - the target axes/dimensions of data for filtering
    #   indices - the multi-axis array of locations at which to pull values from data along the given axes
    # Requirements:
    #   data - ndarray
    #   axes - list of integers
    #   indices - ndarray
    def __filter_axes(self, data, axes, indices):
        if len(data.shape) < 2:
            raise ValueError("Data must be multi-dimensional, received data.shape=%s" % (data.shape))
        if len(axes) != len(indices.shape):
            raise ValueError("Number of axes and filter dimension must be equal, received axes=%s and indices.shape=%s" % (axes, indices.shape))
        # Perform filtering
        #   arange target axes to occupy right-most end
        data = np.moveaxis(data, axes, range(-len(axes), 0))
        #   filter target axes (now the last k=len(axes) dimensions) with indices expanded to dimension of data
        data = np.take_along_axis(
            data, 
            np.expand_dims(indices, tuple(range(len(data.shape) - len(axes)))), 
            -1
        )
        #   arrange target axes back into original positions
        data = np.moveaxis(data, range(-len(axes), 0), axes)
        return data
#        return np.moveaxis(data, range(-len(axes), 0), axes)

    def filter_axes(self, data, axes, indices):
        return self.__filter_axes(data, axes, indices)
    
    def __filter_axis_foreach(self, data, axis, indices):
        if isinstance(data, tuple):
            data = tuple(self.__filter_axis(_, axis, indices) for _ in data)
        elif isinstance(data, list):
            data = [self.__filter_axis(_, axis, indices) for _ in data]
        elif isinstance(data, dict):
            data = {key: self.__filter_axis(value, axis, indices) for key, value in data.items()}
        elif isinstance(data, np.ndarray) and issubclass(data.dtype.type, np.object):
            return np.reshape(
                np.array((self.filter_axis(_, axis, indices) for _ in np.reshape(data, -1)), dtype=object), 
                data.shape
            )
        else:
            raise NotImplementedError("Unknown type (%s) for data in __filter_axis_foreach()" % (str(type))) 
        return data
    
    def filter_axis_foreach(self, data, axis, indices):
        data = self.__filter_axis_foreach(data, axis, indices)
        return data

    def get_reduced_temporal_indices(self, temporal_selection, reduced_temporal_labels, reduced_n_temporal):
        mode = temporal_selection[0]
        implemented_modes = ["interval", "range", "literal"]
        if mode not in implemented_modes:
            raise NotImplementedError("Given mode \"%s\" but only modes \"%s\" are implemented" % (
                mode,
                ",".join(implemented_modes))
            )
        n_temporal_channels = reduced_temporal_labels.shape[0]
        temporal_indices = []
        if mode == "interval":
            for i in range(n_temporal_channels):
                start_idx = self.get_temporal_index(
                    temporal_selection[1],
                    reduced_temporal_labels[i,:reduced_n_temporal[i]]
                )
                end_idx = self.get_temporal_index(
                    temporal_selection[2],
                    reduced_temporal_labels[i,:reduced_n_temporal[i]]
                )
                temporal_indices.append(np.arange(start_idx, end_idx+1))
        elif mode == "range":
            raise NotImplementedError()
        elif mode == "literal":
            raise NotImplementedError()
        return temporal_indices
