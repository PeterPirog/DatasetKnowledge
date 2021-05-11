# https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-ef792bbb3260
# https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin



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
                self.new_categories.append(new_label)
                X[new_label] = np.where(X[category] == label, 1, 0)
                X.loc[X[category].isna(), new_label] = np.nan
        if self.drop:
            X = X.drop(columns=self.categories)  # drop encoded columns
        X[self.new_categories]=X[self.new_categories].astype(self.dtype)
        return X


if __name__ == '__main__':

    columns = ["Sex", "Country"]
    X = [[np.nan, 'Germany'], ['Male', np.nan], ['Female', 'Poland'], ['Female', 'Brasil'], [np.nan, 'Brasil']]
    df = pd.DataFrame(data=X, columns=columns)
    print(df.head())

    cdf = OneHotNanEncoder(categories='auto', drop=False, dtype=np.float32) #'auto'
    cdf.fit(X=df)
    out=cdf.fit_transform(X=df)
    print(out.info())
    print(out.head())
