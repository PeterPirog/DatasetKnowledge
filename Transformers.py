import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer


class QuantileTransformerDf(QuantileTransformer):
    """DataFrame Wrapper around QuantileTransformer
    Function based on: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
    """

    def __init__(self, n_quantiles=1000, output_distribution='uniform', ignore_implicit_zeros=False,
                 subsample=1e5, random_state=42, copy=True, dataframe_as_output=True):
        super(QuantileTransformerDf, self).__init__(n_quantiles=n_quantiles,
                                                    output_distribution=output_distribution,
                                                    ignore_implicit_zeros=ignore_implicit_zeros,
                                                    subsample=subsample,
                                                    random_state=random_state,
                                                    copy=copy)
        self.dataframe_as_output = dataframe_as_output

    def transform(self, X, y=None):
        z = super(QuantileTransformerDf, self).transform(X.values)
        if self.dataframe_as_output:
            return pd.DataFrame(z, index=X.index, columns=X.columns)
        else:
            return z
