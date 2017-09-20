

from mowgly.step import Step

import numpy as np
import pandas as pd
import xarray as xr


class HardcodedNaNFilling(Step):
    """
    This step performs nan-filling of inputs using hardcoded fill values.
    """
    def __init__(self, input_name2fill_values, perform_check=True, copy=False):
        """
        :param input_name2fill_values: mapping from input_name to dict of feature_name->(offset,normalizer) mappings
        :type input_name2fill_values: dict of dicts
        :param perform_check: whether to perform isfinite checking or not (default: True)
        :type perform_check: bool
        :param copy: whether to copy input or not (default: False)
        :type copy: bool
        """
        self.input_name2fill_values = input_name2fill_values
        self.perform_check = perform_check
        self.copy = copy

    def fit(self, **inputs):
        return self

    def forward(self, **inputs):

        for input_name, feature_names in self.input_name2fill_values:
            input = inputs[input_name]
            if self.copy:
                input = input.copy()
            for feat_name, fillvalue in self.input_name2fill_values[input_name].items():
                input[feat_name] = input[feat_name].fillna(fillvalue)

                if self.perform_check and not np.all(np.isfinite(input[feat_name])):
                    msg = "feature {0} in input {1} still has non-finite numbers".format(feat_name, input_name)
                    raise ValueError(msg)

        return inputs


class TSNaNFilling(Step):
    """
    This step performs nan-filling of inputs using forward or backward filling
    """
    def __init__(self, input_name2feature_names, time_coord_name, method="ffill", perform_check=True, copy=False):
        """
        :param input_name2feature_names: mapping from input_name to iterable of feature_names
        :type input_name2feature_names: dict of iterables
        :param time_coord_name: name of the time coordinate (index / column )
        :type time_coord_name: str
        :param method: "ffill" or "bfill" (default: "bfill")
        :type method: str
        :param perform_check: whether to perform isfinite checking or not (default: True)
        :type perform_check: bool
        :param copy: whether to copy input or not (default: False)
        :type copy: bool
        """
        assert method in ('ffill', 'bfill')
        self.method = method
        self.input_name2feature_names = input_name2feature_names
        self.time_coord_name = time_coord_name
        self.perform_check = perform_check
        self.copy = copy

    def fit(self, **inputs):
        return self

    def forward(self, **inputs):

        for input_name, feature_names in self.input_name2feature_names:
            input = inputs[input_name]
            if self.copy:
                input = input.copy()
            for feat_name in feature_names:
                feat = input[feat_name]
                if isinstance(feat, xr.DataArray):
                    feat = feat.to_pandas()
                if not isinstance(feat, (pd.Series, pd.DataFrame)):
                    raise TypeError

                if feat.index.name == self.time_coord_name:
                    input[feat_name] = feat.fillna(method=self.method)
                elif feat.columns.name == self.time_coord_name:
                    input[feat_name] = feat.T.fillna(method=self.method).T
                else:
                    msg = "feature {0} in input {1} must have {2} as index or column".format(feat_name,
                                                                                             input_name,
                                                                                             self.time_coord_name)
                    raise ValueError(msg)

                if self.perform_check and not np.all(np.isfinite(feat)):
                    msg = "feature {0} in input {1} still has non-finite numbers".format(feat_name, input_name)
                    raise ValueError(msg)

        return inputs


class MedianNaNFilling(Step):
    """
    This step performs nan-filling of inputs using the median
    """
    def __init__(self, input_name2feature_names, perform_check=True, copy=False):
        """
        :param input_name2feature_names: mapping from input_name to iterable of feature_names
        :type input_name2feature_names: dict of iterables
        :param perform_check: whether to perform isfinite checking or not (default: True)
        :type perform_check: bool
        :param copy: whether to copy input or not (default: False)
        :type copy: bool
        """
        self.input_name2feature_names = input_name2feature_names
        self.perform_check = perform_check
        self.copy = copy

    def fit(self, **inputs):

        self.input_name2medians = {}
        for input_name, feature_names in self.input_name2feature_names:
            input = inputs[input_name]
            medians = {}
            for feat_name in feature_names:
                median = np.nanmedian(input[feat_name])
                medians[feat_name] = np.nanmedian(input[feat_name])
            self.input_name2medians[input_name] = medians

        return self

    def forward(self, **inputs):

        for input_name, feature_names in self.input_name2feature_names:
            input = inputs[input_name]
            if self.copy:
                input = input.copy()
            medians = self.input_name2medians[input_name]
            for feat_name in feature_names:
                input[feat_name] = input[feat_name].fillna(medians[feat_name])

                if self.perform_check and not np.all(np.isfinite(input[feat_name])):
                    msg = "feature {0} in input {1} still has non-finite numbers".format(feat_name, input_name)
                    raise ValueError(msg)

        return inputs
