

from mowgly.step import Step

import numpy as np


class RobustScaling(Step):
    """
    This step performs robust scaling of inputs using the median and interquantile range
    """
    def __init__(self, input_name2feature_names, quantile_range=(0.25, 0.75), copy=False):
        """
        :param input_name2feature_names: mapping from input_name to iterable of feature_names
        :type input_name2feature_names: dict of iterables
        :param quantile_range: interquantile range
        :type quantile_range: tuple or list of floats of length 2.
        :param copy: whether to copy input or not (default: False)
        :type copy: bool
        """
        self.input_name2feature_names = input_name2feature_names
        self.quantile_range = quantile_range
        self.copy = copy

    def fit(self, **inputs):

        self.input_name2rescalings = {}
        for input_name, feature_names in self.input_name2feature_names:
            input = inputs[input_name]
            rescaling = {}
            for feat_name in feature_names:
                median = np.nanmedian(input[feat_name])
                interq_range = ( np.nanpercentile(input[feat_name], self.quantile_range[1])
                                - np.nanpercentile(input[feat_name], self.quantile_range[0]) )
                rescaling[feat_name] = (median, interq_range)
            self.input_name2rescalings[input_name] = rescaling

        return self

    def forward(self, **inputs):

        for input_name, feature_names in self.input_name2feature_names:
            input = inputs[input_name]
            if self.copy:
                input = input.copy()
            rescaling = self.input_name2rescalings[input_name]
            for feat_name in feature_names:
                median, interq_range = rescaling[feat_name]
                input[feat_name] -= median
                input[feat_name] /= interq_range

        return inputs

