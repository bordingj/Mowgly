
from mowgly.utils import check_random_state

import numpy as np


def get_train_test_random_indices(num_samples, test_ratio, random_state=None):
    """
    This function generates random test and train indices.

    :param num_samples: total number of samples.
    :type num_samples: int
    :param test_ratio: percentage of num_samples that should be test indices.
    :type test_ratio: float
    :param random_state: random number generator / seed (default: None).
    :type random_state: numpy.random.RandomState, int or None
    :return: tuples of two 1d-arrays of train and test indices, respectively.
    """

    num_samples_test = int(num_samples * test_ratio)
    assert num_samples_test > 0
    indices = np.arange(num_samples)
    random_state = check_random_state(random_state)
    random_state.shuffle(indices)
    test_indices = indices[:num_samples_test]
    train_indices = indices[num_samples_test:]

    return train_indices, test_indices

