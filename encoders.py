# https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-ef792bbb3260
# https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156

import pandas as pd
#import modin.pandas as pd
#import ray
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.distributions.empirical_distribution import ECDF
from feature_engine.encoding import RareLabelEncoder

pd.set_option('display.max_columns', None)


class OneHotNanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categories='auto', drop=False, dtype=np.float64):
        super().__init__()
        self.categories = categories
        self.drop = drop
        self.dtype = dtype
        self.new_categories = []

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
        X[self.new_categories] = X[self.new_categories].astype(self.dtype)
        return X


class CDFEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features='auto', drop=True, dtype=np.float64):
        super().__init__()
        self.numerical_features = numerical_features
        self.drop = drop
        self.ecdf_dict = {}

        self.dtype = dtype
        self.new_categories = []

    def fit(self, X, y=None):
        X = X.copy()
        if self.numerical_features == 'auto':
            self.numerical_features = X.select_dtypes(include=np.number).columns.tolist()

        for feature in self.numerical_features:
            self.ecdf_dict[feature] = ECDF(
                X[feature])  # train experimental cuu=mulative distribution function for each feature

        return self

    def transform(self, X, y=None):
        pd.options.mode.chained_assignment = None  # default='warn' - turn off warning about overwrited data
        for key in self.ecdf_dict:
            x = X[key].copy()
            experimental_cdf = self.ecdf_dict[key]
            idx_nan = x.loc[pd.isnull(x)].index  # find nan values in analyzed feature column
            X[key + '_cdf'] = experimental_cdf(x)
            X[key + '_cdf'].loc[idx_nan] = np.nan  # prevent setting CDF value=1 for nan values
        pd.options.mode.chained_assignment = 'warn'  # - turn on warning about overwrited data
        if self.drop:
            X = X.drop(columns=self.numerical_features)  # drop encoded columns

        return X

class RareLabelNanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categories=None,tol=0.05, n_categories=10, max_n_categories=None, replace_with='Rare',impute_missing_label=False):
        super().__init__()
        self.categories = categories
        self.impute_missing_label=impute_missing_label
        #original RareLabelEncoder parameters
        self.tol=tol
        self.n_categories=n_categories
        self.max_n_categories=max_n_categories
        self.replace_with=replace_with


        self.new_categories = []

    def fit(self, X, y=None):
        X = X.copy()
        if self.categories is None:
            self.categories = X.select_dtypes(include=['object']).columns.tolist()

        return self
    def transform(self, X, y=None):
        #pd.options.mode.chained_assignment = None  # default='warn' - turn off warning about overwrited data
        for category in self.categories:
            x = X[category].copy() #not use copy to intentionaly change value
            idx_nan = x.loc[pd.isnull(x)].index  # find nan values in analyzed feature column
            print(f'idx_nan for {category} is {idx_nan}')
            #replace missing values
            x[idx_nan]='MISS'
            encoder=RareLabelEncoder(tol=self.tol, n_categories=self.n_categories,max_n_categories=self.max_n_categories,
                           replace_with=self.replace_with)


            x=x.to_frame(name=category)  # convert pd.series to dataframe
            x = encoder.fit_transform(x)
            X[category] = x
            if not self.impute_missing_label:
                X[category].loc[idx_nan] = np.nan
            print('x type is ', type(x))

        return X


if __name__ == '__main__':
    #ray.init()
    columns = ["Number", "Sex", "Country", "Temperature"]
    """
    Number - numeric feature
    Sex,Country - categorical nominal feature
    Temperature - categorical ordinal feature
    """
    X = [[1.0, np.nan, 'Germany', "Heat"], [2.0, 'Male', np.nan, "Warm"], [2.0, 'Female', 'Poland', "Cold"],
         [0.5, 'Female', 'Brasil', "Unknown"], [0.3, np.nan, 'Poland', np.nan]]
    df = pd.DataFrame(data=X, columns=columns)

    df = pd.read_csv('train.csv')

    rl_enc=RareLabelNanEncoder(categories=None,tol=0.05, n_categories=10, max_n_categories=None,
                               replace_with='Rare',impute_missing_label=False)

    print('ORIGINAL HEAD \n', df.head())

    out=rl_enc.fit_transform(df)

    print('MODIFIED HEAD \n', out.head())
    """
    ohne = OneHotNanEncoder(categories='auto', drop=True, dtype=np.float32)  # 'auto' categories=["Sex", "Country"]
    cdfe = CDFEncoder(numerical_features='auto', dtype=np.float32)

    print('ORIGINAL HEAD \n', df.head())
    out = ohne.fit_transform(X=df)
    out = cdfe.fit_transform(out)

    print('INFO \n', out.info())
    #print('HEAD \n', out.head())

    # explicitly require this experimental feature
    from sklearn.experimental import enable_iterative_imputer  # noqa
    # now you can import normally from sklearn.impute
    from sklearn.impute import IterativeImputer

    imp_mean = IterativeImputer(min_value=0,
                                max_value=1,
                                random_state=0,
                                max_iter=100,
                                tol=1e-3,
                                verbose=2)

    df_imputed=imp_mean.fit_transform(out)
    df_imputed=pd.DataFrame(data=df_imputed)

    df_imputed.to_csv(path_or_buf='train_imputed.csv')
    
    """