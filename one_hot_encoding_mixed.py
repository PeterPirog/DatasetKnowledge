import pandas as pd
import numpy as np
import joblib

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from Transformers import QuantileTransformerDf, IterativeImputerDf, OneHotNanEncoder, RareLabelNanEncoder
from category_encoders import OneHotEncoder

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    # Prepare data example with numerical and categorical values with missing values
    columns = ["C1", "N1", "N2", "N3", "N4"]

    X = [['A', 1.0, np.nan, 0.2, 1.1], ['B', 2.0, 0.35, np.nan, 1.12], [np.nan, 2.5, 3, 2.32, 8.2],
         ['A', 0.5, 1.2, 0.8, 2.2], ['B', 0.3, np.nan, 4.5, np.nan]]
    df = pd.DataFrame(data=X, columns=columns)

    print(df.head())
    # STEP 1 -  categorical features rare labels encoding
    rle = RareLabelNanEncoder(categories=None, tol=0.05, minimum_occurrences=None, n_categories=10,
                              max_n_categories=None,
                              replace_with='Rare', impute_missing_label=False, additional_categories_list=None)

    # STEP 2 - categorical features one hot encoding
    ohe = OneHotNanEncoder(categories='auto', drop=True, dtype=np.float64)

    ohe2=OneHotEncoder(verbose=0, cols=None, drop_invariant=False, return_df=True,
                 handle_missing='return_nan', #options are 'error', 'return_nan', 'value', and 'indicator'.
                       handle_unknown='return_nan',#options are 'error', 'return_nan', 'value', and 'indicator'
                       use_cat_names=False)

    # STEP 3 - numerical values quantile transformation with skewness removing
    q_trans = QuantileTransformerDf(n_quantiles=1000, output_distribution='uniform', ignore_implicit_zeros=False,
                                    subsample=1e5, random_state=42, copy=True, dataframe_as_output=True)

    # STEP 4 - missing values multivariate imputation
    imp = IterativeImputerDf(min_value=0,  # values from 0 to 1 for categorical for numeric
                             max_value=1,
                             random_state=42,
                             max_iter=10,
                             tol=1e-3,
                             verbose=1, dataframe_as_output=True)

    pipe = Pipeline([
        ('rare_lab', rle),
        ('one_hot', ohe2),
        #('q_trans', q_trans),
        #('imputer', imp)
    ])

    pipe.fit(df)
    df_imputed = pipe.fit_transform(df)
    print(f'\ndf_original=\n{df}')
    print(f'df_imputed=\n{df_imputed}')

    #SAVE PIPELINE
    joblib.dump(pipe, 'pipe.pkl')

    #LOAD PIPELINE
    loaded_pipe=joblib.load('pipe.pkl', mmap_mode=None)

    #Prepare test data
    X_test=[['A', 1.1, np.nan, 0.3, 1.5],[np.nan, 1.0, np.nan, 0.5, 2.5]]
    df_test = pd.DataFrame(data=X_test, columns=columns)
    df_test_imputed=loaded_pipe.transform(df_test)

    print(f'df_test_original=\n{df_test}')
    print(f'df_test_imputed=\n{df_test_imputed}')
