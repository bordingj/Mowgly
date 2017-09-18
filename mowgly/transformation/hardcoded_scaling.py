

from mowgly.step import Step

import numpy as np


class HardcodedScaling(Step):
    """
    This step performs rescaling of inputs using harded offsets and denominators
    """
    def __init__(self, input_name2rescalings, copy=False):
        """
        :param input_name2rescalings: mapping from input_name to dict of feature_name->(offset,normalizer) mappings
        :type input_name2rescalings: dict of dicts
        :param copy: whether to copy input or not (default: False)
        :type copy: bool
        """
        self.input_name2rescalings = input_name2rescalings
        self.copy = copy

    def fit(self, **inputs):
        return self

    def forward(self, **inputs):

        for input_name, feature_names in self.input_name2rescalings:
            input = inputs[input_name]
            if self.copy:
                input = input.copy()
            for feat_name, rescaling in self.input_name2rescalings[input_name].items():
                offset, normalizer = rescaling
                input[feat_name] -= offset
                input[feat_name] /= normalizer

        return inputs

