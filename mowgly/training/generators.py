
from collections import OrderedDict

from mowgly.utils import check_random_state
from mowgly.sampling.sampler import WalkerAlias

import numpy as np

def take_sub_arrays(d, indices):
    """

    :param d: dict of arrays supporting __getitem__ with a 1d numpy.ndarray of type int32 as input argument.
    :type d: dict
    :param indices: dict of training indices.
    :type indices: dict of 1d numpy.ndarray of type int32
    :return: dict of sub-arrays.
    """
    out = {}
    for arr_name, arr in d.items():
        if isinstance(arr, dict):
            out[arr_name] = take_sub_arrays(arr, indices[arr_name])
        else:
            out[arr_name] = arr[indices[arr_name]]
    return out

class GeneratorFactory(object):

    def make_train_generator(self):
        """
        This method returns generator of training minibatches.
        """
        raise NotImplementedError

    def make_val_generator(self):
        """
        This method returns a generator of validation minibatches
        """
        raise NotImplementedError


class DefaultGeneratorFactory(GeneratorFactory):


    def __init__(self, arrays, train_indices, val_indices, train_minibatch_size, val_minibatch_size=None,
                 sampling_prob=None, num_threads=4, random_state=None):
        """
        :param arrays: dictionary of data-arrays to be sampled from. all arrays should support __getitem__
                        with a 1d numpy.ndarray of type int32 as input argument.
        :type arrays: dict
        :param train_indices: training indices or dict of training indices where keys in dict are keys in arrays.
        :type train_indices: 1d numpy.ndarray of type int32 or dict
        :param val_indices: validation indices or dict of training indices where keys in dict are keys in arrays.
        :type val_indices: 1d numpy.ndarray of type int32 or dict
        :param train_minibatch_size: number of samples in training minibatch.
        :type train_minibatch_size: int
        :param val_minibatch_size: number of samples in validation minibatch ( will default to train_minibatch_size * 2 )
        :type val_minibatch_size: int
        :param sampling_prob: sampling probability density function - if you want to over/under-sample some samples (default: None - will assume equal probability).
        :type sampling_prob: 1d numpy.ndarray of type float64 or dict
        :param num_threads: number of threads to use when sampling (default: 4)
        :type num_threads: int
        :param random_state: random number generator / seed (default: None).
        :type random_state: numpy.random.RandomState, int or None
        """
        assert isinstance(arrays, (dict, OrderedDict))
        self.arrays = arrays

        if isinstance(train_indices, (dict, OrderedDict)):
            for key, value in train_indices.items():
                assert key in self.arrays
                assert isinstance(value, np.ndarray)
                assert value.ndim == 1
                assert value.dtype == np.int32
        else:
            assert isinstance(train_indices, np.ndarray)
            assert train_indices.ndim == 1
            assert train_indices.dtype == np.int32
            train_indices = {key: train_indices for key in self.arrays.keys()}
        self.train_indices = train_indices
        self.train_size = {key: len(self.train_indices[key]) for key in self.arrays.keys()}
        self.max_train_size = 0
        self.max_train_size_key = None
        for key, train_size in self.train_size.items():
            if train_size > self.max_train_size:
                self.max_train_size = train_size
                self.max_train_size_key = key

        if isinstance(val_indices, (dict, OrderedDict)):
            for key, value in val_indices.items():
                assert key in self.arrays
                assert isinstance(value, np.ndarray)
                assert value.ndim == 1
                assert value.dtype == np.int32
        else:
            assert isinstance(val_indices, np.ndarray)
            assert val_indices.ndim == 1
            assert val_indices.dtype == np.int32
            val_indices = {key: val_indices for key in self.arrays.keys()}

        self.val_indices = val_indices
        self.val_size = {key: len(self.val_indices[key]) for key in self.arrays.keys()}
        self.max_val_size = 0
        self.max_val_size_key = None
        for key, val_size in self.val_size.items():
            if val_size > self.max_val_size:
                self.max_val_size = val_size
                self.max_val_size_key = key

        self.train_minibatch_size = train_minibatch_size
        self.val_minibatch_size = val_minibatch_size

        if isinstance(sampling_prob, (dict, OrderedDict)):
            for key, value in sampling_prob.keys():
                assert key in self.arrays
                assert value.dtype == np.float64
                assert value.ndim == 1
                assert np.isclose(value.sum(), 1)
                assert len(value) == self.train_size[key]
        else:
            if sampling_prob is None:
                sampling_prob = {key: np.ones(self.train_size[key])/self.train_size[key] for key in self.arrays.keys()}
            else:
                assert sampling_prob.dtype == np.float64
                assert sampling_prob.ndim == 1
                assert np.isclose(sampling_prob.sum(), 1)
                assert len(sampling_prob) == self.train_size
                sampling_prob = {key: sampling_prob for key in self.arrays.keys()}

        self.samplers = {key: WalkerAlias(sampling_prob[key], random_state=random_state) for key in self.arrays.keys()}


    def make_train_generator(self):
        """
        This method returns generator of training minibatches.
        """

        for i in range(0, self.max_train_size, self.train_minibatch_size):
            sub_indices = {key: self.train_indices[key][self.samplers[key](self.train_minibatch_size)]
                           for key in self.arrays.keys()}
            yield take_sub_arrays(self.arrays, sub_indices)

    def make_val_generator(self):
        """
        This method returns a generator of validation minibatches
        """
        for i in range(0, self.max_val_size, self.val_minibatch_size):
            sub_indices = {self.val_indices[key][i:i + self.val_minibatch_size] for key in self.arrays.keys()}
            yield take_sub_arrays(self.arrays, sub_indices)