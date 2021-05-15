

import pandas as pd
import scipy.stats

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

from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import beta, skew
from scipy.optimize import minimize



if __name__ == '__main__':
    sample=np.array([1.0,0.2,1.4,2.5,0.3,3.7,0.02,3.0,3.0,3.0,3.0,5.0])
    ecdf=ECDF(sample)
    output=ecdf(sample)
    print(output)
    print(f'Skew={skew(output)}')


    a=1.85589332
    b=4.5617472
    out_norm=beta.pdf(output, a, b, loc=0, scale=1)
    print(f'Out norm={skew(out_norm)}')

    def objective(x):
        sample = np.array([1.0, 0.2, 1.4, 2.5, 0.3, 3.7, 0.02, 3.0, 3.0, 3.0, 3.0, 5.0])
        ecdf = ECDF(sample)
        output = ecdf(sample)
        output=output.flatten()
        out_norm=beta.pdf(output, x[0], x[1], loc=0, scale=1)
        out_norm=out_norm.flatten()
        if np.isnan(skew(out_norm)):
            return 20
        else:
            return np.abs(skew(out_norm))


    # perform the l-bfgs-b algorithm search
    bounds = [[0, np.inf], [0, np.inf]]
    result = minimize(objective, x0=[2,8],bounds=bounds, method='BFGS')

    print(f'result: {result}')