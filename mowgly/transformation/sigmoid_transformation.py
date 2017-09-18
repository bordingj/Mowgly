
from mowgly.step import Step

import numpy as np


class SigmoidTransformation(Step):
    """
    This step applies the natural sigmoid transformation
    """
    def __init__(self, input_name2feature_names, sigmoid_function='tanh', copy=False):
        """
        :param input_name2feature_names: mapping from input_name to iterable of feature_names
        :type input_name2feature_names: dict of iterables
        :param sigmoid_function: sigmoid function
        :type sigmoid_function: str
        :param copy: whether to copy input or not (default: False)
        :type copy: bool
        """
        self.sigmoid_function = sigmoid_function
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
                if self.sigmoid_function == 'tanh':
                    input[feat_name] = np.tanh(input[feat_name])
                else:
                    msg = "unsupported sigmoid function"
                    raise NotImplementedError(msg)

        return inputs


