# https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-ef792bbb3260
# https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.means_ = None
        self.std_ = None

    def fit(self, X, y=None):
        X = X.to_numpy()
        self.means_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True)

        return self

    def transform(self, X, y=None):
        X[:] = (X.to_numpy() - self.means_) / self.std_

        return X


class CdfEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categories='auto', drop=False, dtype=np.float64):
        super().__init__()
        self.categories = categories
        self.drop = drop
        self.dtype = dtype

    def fit(self, X, y=None):
        X = X.copy()
        if self.categories == 'auto':
            self.categories = X.select_dtypes(include=['object']).columns.tolist()
        print(self.categories)

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
                X[new_label] = np.where(X[category] == label, 1, 0)
                X.loc[X[category].isna(), new_label] = np.nan
        if self.drop:
            X = X.drop(columns=self.categories)  # drop encoded columns
        X=X[self.categories].astype(self.dtype)
        return X


if __name__ == '__main__':

    columns = ["Sex", "Country"]
    X = [[np.nan, 'Germany'], ['Male', np.nan], ['Female', 'Poland'], ['Female', 'Brasil'], [np.nan, 'Brasil']]
    df = pd.DataFrame(data=X, columns=columns)
    print(df.head())

    cdf = CdfEncoder(categories='auto', drop=True, dtype=np.float64) #'auto'
    cdf.fit(X=df)
    out=cdf.fit_transform(X=df)
    print(out.info())
