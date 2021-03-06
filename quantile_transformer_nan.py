#http://flennerhag.com/2017-01-08-Recursive-Override/
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer

#imports own classes
from Transformers import QuantileTransformerDf


pd.set_option('display.max_columns', None)

if __name__ == '__main__':

    columns = ["F1", "F2", "F3", "F4"]

    X = [[1.0, np.nan, 0.2, 1.1], [2.0, 0.35, np.nan, 1.12], [2.5, 3, 2.32, 8.2],
         [0.5, 1.2, 0.8, 2.2], [0.3, np.nan, 4.5, np.nan]]
    df = pd.DataFrame(data=X, columns=columns)

    print(df.head())
    q_trans=QuantileTransformerDf(n_quantiles=1000, output_distribution='uniform',ignore_implicit_zeros=False,
                                                     subsample=1e5, random_state=42, copy=True,dataframe_as_output=True)
    out=q_trans.fit_transform(df)

    print(out.head())

