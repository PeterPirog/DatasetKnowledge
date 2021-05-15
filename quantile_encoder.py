import pandas as pd

pd.set_option('display.max_columns', None)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.preprocessing import QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin



class QuantileTransformer2(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features='auto', drop=True, dtype=np.float64, n_quantiles=1000,
                 output_distribution='uniform', ignore_implicit_zeros=False, subsample=100000, random_state=None,
                 copy=True):
        super().__init__()
        self.numerical_features = numerical_features
        self.drop = drop
        # parameters for original QuantileTransformer https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy

        self.quantiles_dict = {}

        self.dtype = dtype
        self.new_categories = []

    def fit(self, X, y=None):
        X = X.copy()
        if self.numerical_features == 'auto':
            self.numerical_features = X.select_dtypes(include=np.number).columns.tolist()
            print('numerical features=', self.numerical_features)
        for feature in self.numerical_features:
            quantile_transformer = QuantileTransformer(n_quantiles=self.n_quantiles,
                                                       output_distribution=self.output_distribution,
                                                       ignore_implicit_zeros=self.ignore_implicit_zeros,
                                                       subsample=self.subsample,
                                                       random_state=self.random_state,
                                                       copy=self.copy)
            x = X[feature].to_frame(name=feature)  # convert to DataFrame prevent converting to Series
            quantile_transformer.fit(x)
            # define quantile encoder for every numerical feature
            self.quantiles_dict[feature] = quantile_transformer
        return self

    def transform(self, X, y=None):
        X=X.copy()
        pd.options.mode.chained_assignment = None  # default='warn' - turn off warning about overwrited data
        for feature in self.quantiles_dict:
            x = X[feature].to_frame(name=feature)  # convert to DataFrame prevent converting to Series

            quantile_transformer = self.quantiles_dict[feature]
            idx_nan = x.loc[pd.isnull(X[feature])].index
            data = quantile_transformer.transform(x)

            X[feature + '_qnt'] = pd.DataFrame(data=data, columns=[feature], dtype=self.dtype)
            X[feature + '_qnt'].loc[idx_nan] = np.nan  # prevent setting CDF value=1 for nan values
        pd.options.mode.chained_assignment = 'warn'  # - turn on warning about overwrited data
        if self.drop:
            X = X.drop(columns=self.numerical_features)  # drop encoded columns

        return X


if __name__ == '__main__':
    # load training data
    data = pd.read_csv('train.csv')

    # split training data into train and test

    X_train, X_test, y_train, y_test = train_test_split(data.drop(
        ['Id', 'SalePrice'], axis=1),
        data['SalePrice'],
        test_size=0.05,
        random_state=0,
        stratify=None)

    qtrans = QuantileTransformer2(numerical_features='auto', drop=True, dtype=np.float32, n_quantiles=1000,
                                  output_distribution='uniform', ignore_implicit_zeros=False, subsample=100000,
                                  random_state=42, copy=True)

    X_trans = qtrans.fit_transform(X_train)
    print(X_trans.info())

    # show X_train data
    X_trans.hist(bins=50, figsize=(10, 10))
    plt.show()
    X_train.hist(bins=50, figsize=(10, 10))
    plt.show()
    X_trans.to_excel("output.xlsx")
    print(X_train.skew(numeric_only=True),'\n')
    print(X_trans.skew(numeric_only=True))
