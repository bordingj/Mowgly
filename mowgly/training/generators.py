


from mowgly.utils import check_random_state
from mowgly.sampling import random_choice_with_cdf

import numpy as np

def take_sub_arrays(d, indices):
    """

    :param d: dict of arrays supporting __getitem__ with a 1d numpy.ndarray of type int32 as input argument.
    :type d: dict
    :param indices: training indices.
    :type indices: 1d numpy.ndarray of type int32
    :return: dict of sub-arrays.
    """
    out = {}
    for arr_name, arr in d.items():
        if isinstance(arr, dict):
            out[arr_name] = take_sub_arrays(arr, indices)
        else:
            out[arr_name] = arr[indices]
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
                 pdf=None, num_threads=4, random_state=None):
        """
        :param arrays: dictionary of data-arrays to be sampled from. all arrays should support __getitem__
                        with a 1d numpy.ndarray of type int32 as input argument.
        :type arrays: dict
        :param train_indices: training indices.
        :type train_indices: 1d numpy.ndarray of type int32
        :param val_indices: validation indices.
        :type val_indices: 1d numpy.ndarray of type int32
        :param train_minibatch_size: number of samples in training minibatch.
        :type train_minibatch_size: int
        :param val_minibatch_size: number of samples in validation minibatch ( will default to train_minibatch_size * 2 )
        :type val_minibatch_size: int
        :param pdf: sampling probability density function (default: None - will assume equal probability).
        :type pdf: 1d numpy.ndarray of type float64
        :param num_threads: number of threads to use when sampling (default: 4)
        :type num_threads: int
        :param random_state: random number generator / seed (default: None).
        :type random_state: numpy.random.RandomState, int or None
        """
        assert isinstance(train_indices, np.ndarray)
        assert train_indices.ndim == 1
        assert train_indices.dtype == np.int32

        assert isinstance(val_indices, np.ndarray)
        assert val_indices.ndim == 1
        assert val_indices.dtype == np.int32

        self.arrays = arrays

        self.train_indices = train_indices
        self.train_size = len(self.train_indices)
        assert self.train_size > train_minibatch_size / 2, "train_minibatch_size must be atleast twice the size of the training data"
        self.train_minibatch_size = train_minibatch_size

        self.val_indices = val_indices
        self.val_size = len(self.val_indices)
        assert self.val_size < self.train_size / 2, "the number of validation indices must be atleast twice the number of training indices"
        if val_minibatch_size is None:
            self.val_minibatch_size = self.val_minibatch_size * 2
        else:
            self.val_minibatch_size = val_minibatch_size
            assert self.val_minibatch_size <= self.val_size, "val_minibatch_size must be less than or equal to the validation size"

        self.random_state = check_random_state(random_state)
        self.pdf = pdf
        self.num_threads = num_threads

        if self.pdf is None:
            self.cdf_ = None
        else:
            assert pdf.dtype == np.float64
            assert pdf.ndim == 1
            assert np.isclose(pdf.sum(), 1)
            assert len(pdf) == self.val_size + self.train_size
            pdf = self.pdf[self.train_indices]
            pdf /= pdf.sum()
            self.cdf_ = np.cumsum(pdf)

    def make_train_generator(self):
        """
        This method returns generator of training minibatches.
        """
        if self.cdf_ is None:
            random_indices = self.random_state.choice(self.train_indices, size=self.train_size, replace=True)
        else:
            random_indices = random_choice_with_cdf(self.train_indices, size=self.train_size, cdf=self.cdf_,
                                                    replace=True, num_threads=self.num_threads,
                                                    random_state=self.random_state)

        for i in range(0, self.train_size, self.train_minibatch_size):
            sub_indices = random_indices[i:( i +self.train_minibatch_size)]
            yield take_sub_arrays(self.arrays, sub_indices)

    def make_val_generator(self):
        """
        This method returns a generator of validation minibatches
        """
        for i in range(0, self.val_size, self.val_minibatch_size):
            sub_indices = self.val_indices[i:( i +self.val_minibatch_size)]
            yield take_sub_arrays(self.arrays, sub_indices)
