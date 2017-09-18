
from copy import deepcopy

from mowgly.utils import check_random_state
from mowgly.sampling import random_choice_with_cdf
from mowgly.training import loggers

import numpy as np


class DefaultTrainGeneratorFactory(object):

    def __init__(self, minibatchsize, cdf=None, num_threads=4, random_state=None):
        self.minibatchsize = minibatchsize
        self.random_state = check_random_state(random_state)
        self.cdf = cdf
        self.num_threads = num_threads

    def __call__(self, arrays, indices):

        if self.cdf is not None:
            random_indices = random_choice_with_cdf(indices, size=len(indices), cdf=self.cdf[indices],
                                   replace=True, num_threads=self.num_threads,
                                   random_state=self.random_state)
        else:
            random_indices = self.random_state.choice(indices, size=len(indices), replace=True)

        new_arrays = {}
        for arr_name, arr in arrays.items():
            new_arrays[arr_name] = arr.values if hasattr(arr, 'values') else arr

        for i in range(0,len(indices),self.minibatchsize):
            sub_indices = random_indices[i:(i+self.minibatchsize)]
            yield {arr_name: arr[sub_indices] for arr_name, arr in new_arrays.items()}


class DefaultValGeneratorFactory(object):

    def __init__(self, minibatchsize):
        self.minibatchsize = minibatchsize

    def __call__(self, arrays, indices):

        new_arrays = {}
        for arr_name, arr in arrays.items():
            new_arrays[arr_name] = arr.values if hasattr(arr, 'values') else arr

        for i in range(0, len(indices), self.minibatchsize):
            sub_indices = indices[i:(i + self.minibatchsize)]
            yield {arr_name: arr[sub_indices] for arr_name, arr in new_arrays.items()}


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


def early_stopping_train(arrays, model, train_generator_factory, val_generator_factory,
                         loss_function, score_function,
                         num_samples, val_ratio, starting_patience, max_epochs,
                         patience_increase_threshold=0.995, patience_multiplier=2.0,
                         logger=None, random_state=None):
    """
    This function performs early stopping training on a model.

    :param arrays: dict of data-arrays passed on to the train_generator_factory and the val_generator_factory
    :type arrays: dict
    :param model: model to be trained. should have class-methods score and train.
    :type model: mowgly.model.Model
    :param train_generator_factory: callable that should return a one-epoch generator for training minibatches
    :type train_generator_factory: callable
    :param val_generator_factory: callable that should return a generator for validation.
    :type val_generator_factory: callable
    :param loss_function: callable passed on to model.train
    :type loss_function: callable
    :param score_function: callable passed on to model.score.
    :type score_function: callable.
    :param num_samples: total number of samples in data.
    :type num_samples: int
    :param val_ratio: percentage of num_samples that should be used for early-stopping validation
    :type val_ratio: float
    :param starting_patience: minimum number of epochs.
    :type starting_patience: int
    :param max_epochs: maximum number of epochs.
    :type max_epochs: int
    :param patience_increase_threshold: threshold for patience increase.
    :type patience_increase_threshold: float
    :param patience_multiplier: patience multiplier when patience is to be increased.
    :type patience_multiplier: float
    :param logger: a logger class that has method log_value (default: mowgly.training.loggers.ConsoleLogger)
    :param random_state: random number generator / seed (default: None).
    :type random_state: numpy.random.RandomState, int or None
    :return: best model found during training.
    """

    if logger is None:
        logger = loggers.ConsoleLogger()

    random_state = check_random_state(random_state)

    # split in train and validation
    train_indices, val_indices = get_train_test_random_indices(num_samples=num_samples,
                                                               test_ratio=val_ratio,
                                                               random_state=random_state)

    # get a validation generator
    val_generator = val_generator_factory(arrays, val_indices)

    # get the current best model score
    best_score = model.score(val_generator, score_function)
    logger.log_value('best_val_score', best_score, 0)

    best_model = deepcopy(model)
    scores_at_latest_patience_increase = best_score
    patience = starting_patience

    num_epochs = 0
    finished = False

    for epoch_number in range(max_epochs):

        # get a train generator and train the model for one epoch
        train_generator = train_generator_factory(arrays, train_indices)
        train_loss = model.train(train_generator, loss_function)
        logger.log_value('train_loss', best_score, epoch_number + 1)

        # get a validation generator and compute validation score
        val_generator = val_generator_factory(arrays, val_indices)
        current_score = model.score(val_generator, score_function)
        logger.log_value('current_val_score', current_score, epoch_number + 1)

        if current_score > best_score:
            if patience_increase_threshold * current_score > scores_at_latest_patience_increase:
                patience = max( patience, patience_multiplier * num_epochs)
                scores_at_latest_patience_increase = current_score
            best_score = current_score
            best_model = deepcopy(model)

        logger.log_value('best_val_score', best_score, epoch_number + 1)

        if num_epochs >= patience:
            break

    return best_model
