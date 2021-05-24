import pandas as pd
import numpy as np

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer


class IterativeImputerDf(IterativeImputer):
    """DataFrame Wrapper around QuantileTransformer
    Function based on: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
    """

    def __init__(self, min_value=-np.inf,  # values from 0 to 1 for categorical for numeric
                 max_value=np.inf,
                 random_state=42,
                 max_iter=10,
                 tol=1e-3,
                 verbose=1, dataframe_as_output=True):
        super(IterativeImputerDf, self).__init__(min_value=min_value,
                                                 max_value=max_value,
                                                 random_state=random_state,
                                                 max_iter=max_iter,
                                                 tol=tol,
                                                 verbose=verbose)
        self.dataframe_as_output = dataframe_as_output

    def transform(self, X, y=None):
        z = super(IterativeImputerDf, self).transform(X.values)
        if self.dataframe_as_output:
            return pd.DataFrame(z, index=X.index, columns=X.columns)
        else:
            return z


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    columns = ["Number", "Sex", "Country"]
    cat_features = ["Sex", "Country"]
    num_features = ["Number"]

    X = [[1.0, np.nan, 'Germany'], [1.0, 'Male', np.nan], [1.99, 'Female', 'Poland'], [np.nan, 'Female', 'Brasil'],
         [3.0, np.nan, 'Brasil']]
    df = pd.DataFrame(data=X, columns=columns)
    df_num = df[num_features]
    print(df_num.head())

    drop_columns = True
    for cat_feature in cat_features:
        labels = df[cat_feature].unique().tolist()
        labels = [str(x) for x in labels]  # converting nan to 'nan
        try:
            labels.remove('nan')  # remove nan labels
        except:
            pass
        for label in labels:
            new_label = str(cat_feature) + '_' + str(label)
            print(f"New label: {new_label} Column: {cat_features}, label: {label}")
            df[new_label] = np.where(df[cat_feature] == label, 1, 0)
            df.loc[df[cat_feature].isna(), new_label] = np.nan
    if drop_columns:
        df = df.drop(columns=columns)  # drop encoded columns

    df = pd.concat([df_num, df])
    print(df.head())
    new_columns = df.columns.to_list()
    print(new_columns)

    columns = ["F1", "F2", "F3", "F4"]

    X = [[1.0, np.nan, 0.2, 1.1], [2.0, 0.35, np.nan, 1.12], [2.5, 3, 2.32, 8.2],
         [0.5, 1.2, 0.8, 2.2], [0.3, np.nan, 4.5, np.nan]]
    df = pd.DataFrame(data=X, columns=columns)

    print(df.head())
    imp_mean = IterativeImputerDf(min_value=-np.inf,  # values from 0 to 1 for categorical for numeric
                                  max_value=np.inf,
                                  random_state=0,
                                  max_iter=100,
                                  tol=1e-6,
                                  verbose=2, dataframe_as_output=True)

    df_imputed = imp_mean.fit_transform(df)
    # df_imputed=pd.DataFrame(df_imputed,columns=new_columns)
    print(df_imputed.head())
