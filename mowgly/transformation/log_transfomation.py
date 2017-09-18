
from mowgly.step import Step

import numpy as np


class LogTransformation(Step):
    """
    This step applies the natural log transformation
    """
    def __init__(self, input_name2feature_names, copy=False):
        """
        :param input_name2feature_names: mapping from input_name to iterable of feature_names
        :type input_name2feature_names: dict of iterables
        :param copy: whether to copy input or not (default: False)
        :type copy: bool
        """

        self.input_name2feature_names = input_name2feature_names
        self.copy = copy

    def fit(self, **inputs):
        return self

    def forward(self, **inputs):

        for input_name, feature_names in self.input_name2feature_names:
            input = inputs[input_name]
            if self.copy:
                input = input.copy()
            for feat_name in feature_names:
                input[feat_name] = np.log(input[feat_name])

        return inputs


