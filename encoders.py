# https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-ef792bbb3260
# https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.distributions.empirical_distribution import ECDF

pd.set_option('display.max_columns', None)


class OneHotNanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categories='auto', drop=False, dtype=np.float64):
        super().__init__()
        self.categories = categories
        self.drop = drop
        self.dtype = dtype
        self.new_categories=[]

    def fit(self, X, y=None):
        X = X.copy()
        if self.categories == 'auto':
            self.categories = X.select_dtypes(include=['object']).columns.tolist()

        return self

    def transform(self, X, y=None):

        for category in self.categories:
            labels = X[category].unique().tolist()
            labels = [str(x) for x in labels]  # converting nan to 'nan
            try:
                labels.remove('nan')  # remove nan labels
            except:
                pass
            for label in labels:
                new_label = str(category) + '_' + str(label)
                self.new_categories.append(new_label)
                X[new_label] = np.where(X[category] == label, 1, 0)
                X.loc[X[category].isna(), new_label] = np.nan
        if self.drop:
            X = X.drop(columns=self.categories)  # drop encoded columns
        X[self.new_categories]=X[self.new_categories].astype(self.dtype)
        return X

class CDFEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features='auto', dtype=np.float64):
        super().__init__()
        self.numerical_features = numerical_features


        self.dtype = dtype
        self.new_categories=[]

    def fit(self, X, y=None):
        X = X.copy()
        if self.numerical_features == 'auto':
            self.numerical_features = X.select_dtypes(include=np.number).columns.tolist()

        for feature in self.numerical_features:
            x=X[feature]
            #idx_nan = x.loc[pd.isna(x[feature]), :].index
            #print(f'Feature: {feature}, nan idx: {idx_nan} ')

        print(f'self.numerical_features= {self.numerical_features}')
        return self

    def transform(self, X, y=None):


        return X

if __name__ == '__main__':

    columns = ["Number","Sex", "Country","Temperature"]
    X = [[1.0,np.nan, 'Germany',"Heat"], [2.0,'Male', np.nan,"Warm"], [2.0,'Female', 'Poland',"Cold"], [0.5,'Female', 'Brasil',"Unknown"], [0.3,np.nan, 'Poland',np.nan]]
    df = pd.DataFrame(data=X, columns=columns)
    print(df.head())

    ohne = OneHotNanEncoder(categories=["Number","Sex", "Country","Temperature"], drop=True, dtype=np.float32) #'auto'
    cdfe=CDFEncoder(numerical_features='auto', dtype=np.float64)
    ohne.fit(X=df)
    out=ohne.fit_transform(X=df)
    out=cdfe.fit_transform(out)



    print(out.info())
    print(out.head())
"""
    for feature in num_features:
        x=df[feature]
        idx=df.loc[pd.isna(df[feature]), :].index
        ecdf = ECDF(x)
        output=ecdf(x)
        output[idx]=np.nan
        print(f'idx={idx}')
        print(f' ECDF:{output}')"""